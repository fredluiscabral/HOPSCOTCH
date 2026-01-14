// hopscotch2d_hib_sem_barrier.cpp
// Hopscotch 2D (equação do calor) — HÍBRIDO MPI + OpenMP + semáforos POSIX
// - Sincronização local vizinho-a-vizinho por fase (semáforos) + barreiras globais (OpenMP)
//   para proteger as trocas de halos (MPI_THREAD_FUNNELED).
// - A cada fase, após o handshake local, o thread mestre (tid==0) faz a troca de halos MPI;
//   em seguida há uma barreira OpenMP garantindo que todos vejam os halos atualizados.
// - A cada passo, há barreira global (OpenMP) para alinhar a paridade global (m_shared).
// - Apenas o rank 0 escreve um único "output.txt" (amostrado a cada 16 pontos).
// - Em stdout: APENAS "Tempo : <segundos> s".
//
// Compilar (exemplo):
//   mpicxx -std=c++17 -O3 -fopenmp -pthread -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1 \
//          hopscotch2d_hib_sem_barrier.cpp -o hopscotch2d_hib_sem_barrier -lm

#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <cctype>
#include <semaphore.h>

// --------------------- util de parsing ---------------------
static inline std::string ltrim(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
        [](unsigned char ch){ return !std::isspace(ch); }));
    return s;
}
static inline std::string rtrim(std::string s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
        [](unsigned char ch){ return !std::isspace(ch); }).base(), s.end());
    return s;
}
static inline std::string trim(std::string s) { return rtrim(ltrim(s)); }
static inline std::string tolower_str(std::string s){
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

static bool load_params_strict(const std::string& fname,
                               int& N, double& alpha, int& T, int& TILE)
{
    std::ifstream fin(fname);
    if (!fin) {
        std::cerr << "Erro: não foi possível abrir " << fname << " (é obrigatório).\n";
        return false;
    }

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string key = trim(line.substr(0, pos));
        std::string val = trim(line.substr(pos+1));
        kv[tolower_str(key)] = val;
    }

    try {
        if (!kv.count("n"))     { std::cerr << "Erro: parâmetro obrigatório 'n' ausente.\n"; return false; }
        if (!kv.count("alpha")) { std::cerr << "Erro: parâmetro obrigatório 'alpha' ausente.\n"; return false; }
        if (!kv.count("t"))     { std::cerr << "Erro: parâmetro obrigatório 't' ausente.\n"; return false; }
        if (!kv.count("tile"))  { std::cerr << "Erro: parâmetro obrigatório 'tile' ausente.\n"; return false; }
        N     = std::stoi(kv.at("n"));
        alpha = std::stod(kv.at("alpha"));
        T     = std::stoi(kv.at("t"));
        TILE  = std::stoi(kv.at("tile"));
    } catch (const std::exception& e) {
        std::cerr << "Erro: falha ao converter parâmetros de " << fname << ": " << e.what() << "\n";
        return false;
    }

    if (N < 3)          { std::cerr << "Erro: N >= 3.\n"; return false; }
    if (alpha <= 0.0)   { std::cerr << "Erro: alpha > 0.\n"; return false; }
    if (T < 0)          { std::cerr << "Erro: T >= 0.\n"; return false; }
    if (TILE < 1 || TILE > N-2) {
        std::cerr << "Erro: tile deve satisfazer 1 <= tile <= N-2.\n";
        return false;
    }
    return true;
}

// --------------------- troca de halos (MPI, funneled) ---------------------
static void exchange_halos(double* buf, int N, int local_n,
                           int up, int down, int tag, MPI_Comm comm)
{
    // buf layout: (local_n+2) x N, com linhas 0 e local_n+1 como halos
    MPI_Request reqs[4]; int rcount = 0;

    if (up != MPI_PROC_NULL) {
        MPI_Irecv(&buf[0 * (size_t)N], N, MPI_DOUBLE, up,   tag, comm, &reqs[rcount++]);
        MPI_Isend(&buf[1 * (size_t)N], N, MPI_DOUBLE, up,   tag, comm, &reqs[rcount++]);
    }
    if (down != MPI_PROC_NULL) {
        MPI_Irecv(&buf[(size_t)(local_n+1) * N], N, MPI_DOUBLE, down, tag, comm, &reqs[rcount++]);
        MPI_Isend(&buf[(size_t)(local_n)   * N], N, MPI_DOUBLE, down, tag, comm, &reqs[rcount++]);
    }
    if (rcount) MPI_Waitall(rcount, reqs, MPI_STATUSES_IGNORE);
}

// --------------------- programa principal ---------------------
int main(int argc, char** argv)
{
    // MPI init (funneled: só o thread mestre chama MPI dentro da região OpenMP)
    int prov = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);

    MPI_Comm COMM = MPI_COMM_WORLD;
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(COMM, &rank);
    MPI_Comm_size(COMM, &nprocs);

    // ---- Parâmetros globais ----
    int N=0, T=0, TILE=0; double alpha=0.0;
    if (!load_params_strict("param.txt", N, alpha, T, TILE)) {
        MPI_Abort(COMM, 1);
        return 1;
    }

    // ---- Discretização ----
    const double h   = 1.0 / (N - 1);
    const double dt  = 0.90 * (h*h) / (4.0 * alpha);
    const double lam = alpha * dt / (h*h);
    const double denom = 1.0 + 4.0 * lam;

    // ---- Decomposição por linhas (somente interior 1..N-2) ----
    const int interior = (N - 2);
    if (interior <= 0) {
        if (rank==0) std::cerr << "Erro: N muito pequeno.\n";
        MPI_Abort(COMM, 1);
        return 1;
    }
    const int base = interior / nprocs;
    const int rem  = interior % nprocs;

    const int local_n = base + ((rank < rem) ? 1 : 0);
    const int start_g = 1 + rank*base + std::min(rank, rem);
    const int end_g   = start_g + local_n - 1;

    const int up   = (rank > 0)        ? rank-1   : MPI_PROC_NULL;
    const int down = (rank < nprocs-1) ? rank+1   : MPI_PROC_NULL;

    // ---- Arrays locais com halos ----
    const size_t LROWS = (size_t)(local_n + 2);
    std::vector<double> U_new(LROWS * (size_t)N, 0.0), U_old(LROWS * (size_t)N, 0.0);
    auto L = [&](int il, int j)->size_t { return (size_t)il * (size_t)N + (size_t)j; };

    // ---- Condição inicial (somente interior real: il=1..local_n, j=1..N-2) ----
    const double D  = 100.0;
    const double x0 = 0.5;
    const double y0 = 0.5;
    for (int il = 1; il <= local_n; ++il) {
        const int ig = start_g + (il - 1);
        const double x = ig * h;
        for (int j = 1; j <= N-2; ++j) {
            const double y = j * h;
            U_new[L(il,j)] = std::exp(-D*((x-x0)*(x-x0) + (y-y0)*(y-y0)));
        }
    }

    // Contornos Dirichlet 0 em j=0 e j=N-1
    for (int il = 0; il <= local_n+1; ++il) {
        U_new[L(il,0)]   = 0.0; U_new[L(il,N-1)] = 0.0;
        U_old[L(il,0)]   = 0.0; U_old[L(il,N-1)] = 0.0;
    }
    // Bordas globais i=0 e i=N-1 (halos) quando aplicável
    if (up == MPI_PROC_NULL) {
        for (int j=0; j<N; ++j) { U_new[L(0,j)] = 0.0; U_old[L(0,j)] = 0.0; }
    }
    if (down == MPI_PROC_NULL) {
        for (int j=0; j<N; ++j) { U_new[L(local_n+1,j)] = 0.0; U_old[L(local_n+1,j)] = 0.0; }
    }

    // Sincroniza antes de cronometrar (tempo global)
    MPI_Barrier(COMM);
    const double t0 = MPI_Wtime();

    // Variável compartilhada de paridade global (por rank)
    int m_shared = 0;

    // ---------------- OpenMP região paralela ----------------
    std::vector<sem_t> sem_left, sem_right;

    #pragma omp parallel default(none) \
        shared(N, T, TILE, lam, denom, U_new, U_old, L, local_n, start_g, up, down, COMM, \
               sem_left, sem_right, m_shared)
    {
        const int tid = omp_get_thread_num();
        const int NT = omp_get_num_threads();

        // Inicializa semáforos (apenas uma thread)
        #pragma omp single
        {
            sem_left.resize(NT);
            sem_right.resize(NT);
            for (int t = 0; t < NT; ++t) {
                sem_init(&sem_left[t],  0, 0);
                sem_init(&sem_right[t], 0, 0);
            }
        }
        #pragma omp barrier

        // Particionamento em faixas locais (linhas 1..local_n)
        const int H = local_n;
        const int baseT = H / NT;
        const int sobraT = H % NT;
        int iLocal = 1 + tid*baseT + std::min(tid, sobraT);
        int fLocal = iLocal + baseT - 1;
        if (tid < sobraT) fLocal += 1;

        auto signal_done = [&](int t){
            sem_post(&sem_left[t]);
            sem_post(&sem_right[t]);
        };
        auto wait_for_neighbors = [&](int t){
            if (t > 0)     sem_wait(&sem_right[t-1]);
            if (t < NT-1)  sem_wait(&sem_left[t+1]);
        };

        for (int step = 0; step < T; ++step) {

            // --- Troca de halo U_new (necessária para Fase 1)
            #pragma omp master
            {
                exchange_halos(U_new.data(), N, local_n, up, down, 100 + 4*(step%1000000) + 0, COMM);
            }
            #pragma omp barrier

            int m_local = m_shared;

            // ---------------- FASE 1 (explícita): U_old a partir de U_new nos pontos pares
            {
                for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                    const int i_end = std::min(fLocal, ii + TILE - 1);
                    for (int jj = 1; jj <= N-2; jj += TILE) {
                        const int j_end = std::min(N-2, jj + TILE - 1);
                        for (int i = ii; i <= i_end; ++i) {
                            const int ig = (start_g - 1) + i; // i global
                            for (int j = jj; j <= j_end; ++j) {
                                if ( ((ig + j + m_local) & 1) == 0 ) {
                                    U_old[L(i,j)] = U_new[L(i,j)] +
                                        lam * ( U_new[L(i+1,j)] + U_new[L(i-1,j)]
                                              + U_new[L(i,j+1)] + U_new[L(i,j-1)]
                                              - 4.0 * U_new[L(i,j)] );
                                } else {
                                    U_old[L(i,j)] = U_new[L(i,j)];
                                }
                            }
                        }
                    }
                }
                signal_done(tid);
                wait_for_neighbors(tid);
            }

            // --- Troca de halo U_old (após Fase 1)
            #pragma omp master
            {
                exchange_halos(U_old.data(), N, local_n, up, down, 100 + 4*(step%1000000) + 1, COMM);
            }
            #pragma omp barrier

            // ---------------- FASE 2 (semi-implícita): U_old nos pontos ímpares
            {
                for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                    const int i_end = std::min(fLocal, ii + TILE - 1);
                    for (int jj = 1; jj <= N-2; jj += TILE) {
                        const int j_end = std::min(N-2, jj + TILE - 1);
                        for (int i = ii; i <= i_end; ++i) {
                            const int ig = (start_g - 1) + i;
                            for (int j = jj; j <= j_end; ++j) {
                                if ( ((ig + j + m_local) & 1) == 1 ) {
                                    U_old[L(i,j)] =
                                        ( U_new[L(i,j)]
                                        + lam*( U_old[L(i+1,j)] + U_old[L(i-1,j)]
                                              + U_old[L(i,j+1)] + U_old[L(i,j-1)] ) ) / denom;
                                }
                            }
                        }
                    }
                }
                signal_done(tid);
                wait_for_neighbors(tid);
                m_local++;
            }

            // --- Troca de halo U_old (após Fase 2)
            #pragma omp master
            {
                exchange_halos(U_old.data(), N, local_n, up, down, 100 + 4*(step%1000000) + 2, COMM);
            }
            #pragma omp barrier

            // ---------------- FASE 3 (explícita): U_new a partir de U_old nos pontos pares
            {
                for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                    const int i_end = std::min(fLocal, ii + TILE - 1);
                    for (int jj = 1; jj <= N-2; jj += TILE) {
                        const int j_end = std::min(N-2, jj + TILE - 1);
                        for (int i = ii; i <= i_end; ++i) {
                            const int ig = (start_g - 1) + i;
                            for (int j = jj; j <= j_end; ++j) {
                                if ( ((ig + j + m_local) & 1) == 0 ) {
                                    U_new[L(i,j)] = U_old[L(i,j)] +
                                        lam * ( U_old[L(i+1,j)] + U_old[L(i-1,j)]
                                              + U_old[L(i,j+1)] + U_old[L(i,j-1)]
                                              - 4.0 * U_old[L(i,j)] );
                                } else {
                                    U_new[L(i,j)] = U_old[L(i,j)];
                                }
                            }
                        }
                    }
                }
                signal_done(tid);
                wait_for_neighbors(tid);
            }

            // --- Troca de halo U_new (após Fase 3)
            #pragma omp master
            {
                exchange_halos(U_new.data(), N, local_n, up, down, 100 + 4*(step%1000000) + 3, COMM);
            }
            #pragma omp barrier

            // ---------------- FASE 4 (semi-implícita): U_new nos pontos ímpares
            {
                for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                    const int i_end = std::min(fLocal, ii + TILE - 1);
                    for (int jj = 1; jj <= N-2; jj += TILE) {
                        const int j_end = std::min(N-2, jj + TILE - 1);
                        for (int i = ii; i <= i_end; ++i) {
                            const int ig = (start_g - 1) + i;
                            for (int j = jj; j <= j_end; ++j) {
                                if ( ((ig + j + m_local) & 1) == 1 ) {
                                    U_new[L(i,j)] =
                                        ( U_old[L(i,j)]
                                        + lam*( U_new[L(i+1,j)] + U_new[L(i-1,j)]
                                              + U_new[L(i,j+1)] + U_new[L(i,j-1)] ) ) / denom;
                                }
                            }
                        }
                    }
                }
                signal_done(tid);
                wait_for_neighbors(tid);
                m_local++;
            }

            // --- Barreira global por passo: mantém índice temporal alinhado entre threads
            #pragma omp barrier
            #pragma omp single
            {
                m_shared += 2;
            }
            #pragma omp barrier
        } // step

        // Destrói semáforos
        #pragma omp single
        {
            for (int t = 0; t < NT; ++t) {
                sem_destroy(&sem_left[t]);
                sem_destroy(&sem_right[t]);
            }
        }
    } // fim região paralela

    MPI_Barrier(COMM);
    const double t1 = MPI_Wtime();
    double secs = t1 - t0;

    // Tempo global (máximo entre ranks)
    double secs_max = 0.0;
    MPI_Reduce(&secs, &secs_max, 1, MPI_DOUBLE, MPI_MAX, 0, COMM);

    if (rank == 0) {
        std::cout << "Tempo : " << secs_max << " s\n";
    }

    // ---------------- Saída única (rank 0) ----------------
    // Amostragem de 16 em 16, ordenada por i crescente.
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(8);

    // Rank 0 adiciona linha i=0
    if (rank == 0) {
        if (0 % 16 == 0) {
            const int i = 0;
            const double x = i * h;
            for (int j=0; j<N; j+=16) {
                const double y = j * h;
                oss << x << " " << y << " " << 0.0 << "\n";
            }
        }
    }

    // Linhas interiores deste rank (i=start_g..end_g) múltiplas de 16
    int first_ig = ((start_g + 15) / 16) * 16;
    for (int ig = first_ig; ig <= end_g; ig += 16) {
        const int il = (ig - start_g) + 1; // 1..local_n
        const double x = ig * h;
        for (int j = 0; j < N; j += 16) {
            const double y = j * h;
            double v = (j == 0 || j == N-1) ? 0.0 : U_new[L(il,j)];
            oss << x << " " << y << " " << v << "\n";
        }
    }

    // Último rank adiciona linha i=N-1
    if (rank == nprocs-1) {
        if ((N-1) % 16 == 0) {
            const int i = N-1;
            const double x = i * h;
            for (int j=0; j<N; j+=16) {
                const double y = j * h;
                oss << x << " " << y << " " << 0.0 << "\n";
            }
        }
    }

    std::string local_txt = oss.str();
    int local_len = (int)local_txt.size();

    std::vector<int> recvcounts, displs;
    if (rank == 0) recvcounts.resize(nprocs);
    MPI_Gather(&local_len, 1, MPI_INT,
               rank==0 ? recvcounts.data() : nullptr, 1, MPI_INT,
               0, COMM);

    std::vector<char> allbuf;
    if (rank == 0) {
        displs.resize(nprocs);
        int total = 0;
        for (int p=0; p<nprocs; ++p) { displs[p] = total; total += recvcounts[p]; }
        allbuf.resize((size_t)total);
        MPI_Gatherv(local_txt.data(), local_len, MPI_CHAR,
                    allbuf.data(), recvcounts.data(), displs.data(), MPI_CHAR,
                    0, COMM);

        std::ofstream fout("output.txt");
        if (!fout) {
            std::cerr << "Erro ao abrir output.txt\n";
        } else {
            fout.write(allbuf.data(), (std::streamsize)allbuf.size());
            fout.close();
        }
    } else {
        MPI_Gatherv(local_txt.data(), local_len, MPI_CHAR,
                    nullptr, nullptr, nullptr, MPI_CHAR,
                    0, COMM);
    }

    MPI_Finalize();
    return 0;
}
