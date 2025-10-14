// Hopscotch 2D (equação do calor) — HÍBRIDO MPI + OpenMP + semáforos POSIX
// - Decomposição 1D em linhas entre processos MPI
// - Dentro de cada processo: divisão em faixas por thread com sincronização local (semáforos)
// - SEM barreira global de OpenMP nas fases (apenas barreiras pontuais para halos MPI)
// - Lê param.txt (obrigatório)
// - Em stdout imprime SOMENTE: "Tempo : <s> s" (apenas no rank 0)
// - Rank 0 escreve um único "output.txt" (amostrado a cada 16 pontos)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cctype>
#include <semaphore.h>
#include <omp.h>
#include <mpi.h>

// ---------- Utils ----------
static inline std::string ltrim(std::string s){
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
        [](unsigned char ch){ return !std::isspace(ch); }));
    return s;
}
static inline std::string rtrim(std::string s){
    s.erase(std::find_if(s.rbegin(), s.rend(),
        [](unsigned char ch){ return !std::isspace(ch); }).base(), s.end());
    return s;
}
static inline std::string trim(std::string s){ return rtrim(ltrim(s)); }
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
        std::cerr << "Erro: não foi possível abrir " << fname << " (obrigatório).\n";
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
        if (!kv.count("n"))     { std::cerr << "Erro: falta 'n'.\n"; return false; }
        if (!kv.count("alpha")) { std::cerr << "Erro: falta 'alpha'.\n"; return false; }
        if (!kv.count("t"))     { std::cerr << "Erro: falta 't'.\n"; return false; }
        if (!kv.count("tile"))  { std::cerr << "Erro: falta 'tile'.\n"; return false; }
        N     = std::stoi(kv["n"]);
        alpha = std::stod(kv["alpha"]);
        T     = std::stoi(kv["t"]);
        TILE  = std::stoi(kv["tile"]);
    } catch (const std::exception& e) {
        std::cerr << "Erro: conversão de parâmetros: " << e.what() << "\n";
        return false;
    }
    if (N < 3)            { std::cerr << "Erro: N>=3.\n"; return false; }
    if (alpha <= 0.0)     { std::cerr << "Erro: alpha>0.\n"; return false; }
    if (T < 0)            { std::cerr << "Erro: T>=0.\n"; return false; }
    if (TILE < 1 || TILE > N-2) {
        std::cerr << "Erro: 1<=TILE<=N-2.\n"; return false;
    }
    return true;
}

// ---------- Troca de halos (linhas superior e inferior) ----------
static inline void exchange_halos(double* U, int N, int local_n,
                                  int up, int down, int tag_base, MPI_Comm comm)
{
    MPI_Request reqs[4];
    int rq = 0;

    if (up   != MPI_PROC_NULL)
        MPI_Irecv(&U[0 * (size_t)N], N, MPI_DOUBLE, up,   tag_base+1, comm, &reqs[rq++]);
    if (down != MPI_PROC_NULL)
        MPI_Irecv(&U[(size_t)(local_n+1)*N], N, MPI_DOUBLE, down, tag_base+0, comm, &reqs[rq++]);

    if (up   != MPI_PROC_NULL)
        MPI_Isend(&U[1 * (size_t)N], N, MPI_DOUBLE, up,   tag_base+0, comm, &reqs[rq++]);
    if (down != MPI_PROC_NULL)
        MPI_Isend(&U[(size_t)local_n * N], N, MPI_DOUBLE, down, tag_base+1, comm, &reqs[rq++]);

    if (rq) MPI_Waitall(rq, reqs, MPI_STATUSES_IGNORE);
}

int main(int argc, char** argv){
    // ---- MPI init (FUNNELED: só thread master chama MPI) ----
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int world_size = 1, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // DUPLICA o comunicador para evitar o símbolo global ompi_mpi_comm_world na região OpenMP
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    // ---- Parâmetros globais ----
    int N=0, T=0, TILE=0; double alpha=0.0;
    if (!load_params_strict("param.txt", N, alpha, T, TILE)) {
        MPI_Abort(comm, 1);
        return 1;
    }
    const double h   = 1.0 / (N - 1);
    const double dt  = 0.90 * (h*h) / (4.0 * alpha); // estável explícito 2D
    const double lam = alpha * dt / (h*h);
    const double denom = 1.0 + 4.0 * lam;

    // ---- Decomposição 1D (linhas interiores 1..N-2) entre ranks ----
    const int interior_glob = (N - 2);
    const int base = (world_size ? interior_glob / world_size : interior_glob);
    const int rem  = (world_size ? interior_glob % world_size : 0);

    const int local_n = base + (world_rank < rem ? 1 : 0); // linhas interiores locais
    const int start_g = 1 + world_rank * base + std::min(world_rank, rem); // linha global inicial (interior)

    // Vizinhos MPI (Proc null nas bordas)
    const int up   = (world_rank > 0) ? world_rank - 1 : MPI_PROC_NULL;
    const int down = (world_rank < world_size-1) ? world_rank + 1 : MPI_PROC_NULL;

    // ---- Campos locais com halos ----
    const size_t rows_loc = (size_t)local_n + 2; // inclui 2 halos
    const size_t NN_loc   = rows_loc * (size_t)N;
    std::vector<double> U_new(NN_loc, 0.0), U_old(NN_loc, 0.0);
    auto idx = [N](int i, int j) -> size_t { return (size_t)i * (size_t)N + (size_t)j; };

    // ---- Condição inicial (gaussiano no centro, somente interior) ----
    const double D=100.0, x0=0.5, y0=0.5;
    for (int li = 1; li <= local_n; ++li) {
        const int gi = start_g + (li - 1); // índice global da linha
        const double x = gi * h;
        for (int j = 1; j <= N-2; ++j) {
            const double y = j * h;
            U_new[idx(li,j)] = std::exp(-D*((x-x0)*(x-x0)+(y-y0)*(y-y0)));
        }
    }
    // Contorno Dirichlet 0 (colunas borda)
    for (int li = 0; li <= local_n+1; ++li) {
        U_new[idx(li,0)]   = 0.0; U_new[idx(li,N-1)] = 0.0;
        U_old[idx(li,0)]   = 0.0; U_old[idx(li,N-1)] = 0.0;
    }
    // Halos físicos (topo/bottom) são 0 nas bordas globais
    if (up   == MPI_PROC_NULL) { std::fill_n(&U_new[idx(0,0)], N, 0.0); std::fill_n(&U_old[idx(0,0)], N, 0.0); }
    if (down == MPI_PROC_NULL) { std::fill_n(&U_new[idx(local_n+1,0)], N, 0.0); std::fill_n(&U_old[idx(local_n+1,0)], N, 0.0); }

    // ---- Semáforos por thread (sincronização local) ----
    std::vector<sem_t> sem_left, sem_right;       // por fase
    std::vector<sem_t> step_left, step_right;     // rendezvous por passo

    // ---- Medição do laço principal ----
    MPI_Barrier(comm); // alinhar antes de começar
    const double t0 = MPI_Wtime();

    // Região paralela (OpenMP) — cada rank trabalha nas suas linhas locais
    #pragma omp parallel default(none) \
        shared(N, T, TILE, lam, denom, U_new, U_old, idx, \
               sem_left, sem_right, step_left, step_right, \
               local_n, up, down, comm)
    {
        const int nt  = omp_get_num_threads();
        const int tid = omp_get_thread_num();

        // Inicialização dos semáforos (uma vez por processo)
        #pragma omp single
        {
            sem_left.resize(nt);  sem_right.resize(nt);
            step_left.resize(nt); step_right.resize(nt);
            for (int t = 0; t < nt; ++t) {
                sem_init(&sem_left[t],  0, 0);
                sem_init(&sem_right[t], 0, 0);
                sem_init(&step_left[t],  0, 0);
                sem_init(&step_right[t], 0, 0);
            }
        }
        #pragma omp barrier

        // Particionamento das linhas locais (1..local_n) por faixas contíguas
        const int baseT = (nt ? local_n / nt : local_n);
        const int remT  = (nt ? local_n % nt : 0);
        int iLocal = 1 + tid * baseT + std::min(tid, remT);
        int mySize = baseT + (tid < remT ? 1 : 0);
        int fLocal = iLocal + mySize - 1;

        // Auxiliares de sincronização local por fase
        auto signal_done_phase = [&](int t){
            sem_post(&sem_left[t]);
            sem_post(&sem_right[t]);
        };
        auto wait_neighbors_phase = [&](int t){
            if (t > 0)     sem_wait(&sem_right[t-1]); // esquerda
            if (t < nt-1)  sem_wait(&sem_left[t+1]);  // direita
        };
        auto announce_step = [&](int t){
            sem_post(&step_left[t]);
            sem_post(&step_right[t]);
        };
        auto wait_step_neighbors = [&](int t){
            if (t > 0)     sem_wait(&step_right[t-1]);
            if (t < nt-1)  sem_wait(&step_left[t+1]);
        };

        // Troca inicial de halos de U_new (estado t=0)
        #pragma omp master
        {
            exchange_halos(U_new.data(), N, local_n, up, down, /*tag=*/100, comm);
        }
        #pragma omp barrier

        for (int s = 0; s < T; ++s) {
            if (s > 0) wait_step_neighbors(tid); // rendezvous local
            int mloc = 2*s;

            // ===== FASE 1 (explícita): U_old a partir de U_new nos pontos (par) =====
            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + mloc) & 1) == 0) {
                                U_old[idx(i,j)] = U_new[idx(i,j)] +
                                  lam*( U_new[idx(i+1,j)] + U_new[idx(i-1,j)]
                                      + U_new[idx(i,j+1)] + U_new[idx(i,j-1)]
                                      - 4.0 * U_new[idx(i,j)] );
                            } else {
                                U_old[idx(i,j)] = U_new[idx(i,j)];
                            }
                        }
                    }
                }
            }
            signal_done_phase(tid);
            wait_neighbors_phase(tid);

            // Troca de halos de U_old para FASE 2
            #pragma omp master
            {
                exchange_halos(U_old.data(), N, local_n, up, down, /*tag=*/100 + 4*s + 1, comm);
            }
            #pragma omp barrier

            // ===== FASE 2 (semi-implícita): U_old nos pontos (ímpar) =====
            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + mloc) & 1) == 1) {
                                U_old[idx(i,j)] =
                                  ( U_new[idx(i,j)]
                                  + lam*( U_old[idx(i+1,j)] + U_old[idx(i-1,j)]
                                        + U_old[idx(i,j+1)] + U_old[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            signal_done_phase(tid);
            wait_neighbors_phase(tid);
            ++mloc;

            // ===== FASE 3 (explícita): U_new a partir de U_old (par) =====
            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + mloc) & 1) == 0) {
                                U_new[idx(i,j)] = U_old[idx(i,j)] +
                                  lam*( U_old[idx(i+1,j)] + U_old[idx(i-1,j)]
                                      + U_old[idx(i,j+1)] + U_old[idx(i,j-1)]
                                      - 4.0 * U_old[idx(i,j)] );
                            } else {
                                U_new[idx(i,j)] = U_old[idx(i,j)];
                            }
                        }
                    }
                }
            }
            signal_done_phase(tid);
            wait_neighbors_phase(tid);

            // Troca de halos de U_new para FASE 4
            #pragma omp master
            {
                exchange_halos(U_new.data(), N, local_n, up, down, /*tag=*/100 + 4*s + 2, comm);
            }
            #pragma omp barrier

            // ===== FASE 4 (semi-implícita): U_new (ímpar) =====
            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + mloc) & 1) == 1) {
                                U_new[idx(i,j)] =
                                  ( U_old[idx(i,j)]
                                  + lam*( U_new[idx(i+1,j)] + U_new[idx(i-1,j)]
                                        + U_new[idx(i,j+1)] + U_new[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            signal_done_phase(tid);
            wait_neighbors_phase(tid);
            ++mloc;

            // Rendezvous local de PASSO
            announce_step(tid);
        } // passos

        // Destrói semáforos (uma vez por processo)
        #pragma omp single
        {
            for (int t = 0; t < nt; ++t) {
                sem_destroy(&sem_left[t]);
                sem_destroy(&sem_right[t]);
                sem_destroy(&step_left[t]);
                sem_destroy(&step_right[t]);
            }
        }
    } // região paralela

    const double t1 = MPI_Wtime();
    double elapsed_local = t1 - t0, elapsed_max = 0.0;
    MPI_Allreduce(&elapsed_local, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    // ---- Coleta e escrita do resultado final (apenas rank 0) ----
    const int sendcount = local_n * N;

    int world_size_chk = world_size; // para calcular counts/displs
    const int base_chk = (world_size_chk ? (N-2) / world_size_chk : (N-2));
    const int rem_chk  = (world_size_chk ? (N-2) % world_size_chk : 0);

    std::vector<int> counts, displs;
    std::vector<double> U_global;

    if (world_rank == 0) {
        counts.resize(world_size);
        displs.resize(world_size);
        for (int r = 0; r < world_size; ++r) {
            const int ln = base_chk + (r < rem_chk ? 1 : 0);
            const int sg = 1 + r * base_chk + std::min(r, rem_chk);
            counts[r] = ln * N;
            displs[r] = sg * N;
        }
        U_global.assign((size_t)N * (size_t)N, 0.0);
    }

    MPI_Gatherv( (sendcount ? &U_new[(size_t)1 * N] : nullptr), sendcount, MPI_DOUBLE,
                 (world_rank==0 ? U_global.data() : nullptr),
                 (world_rank==0 ? counts.data() : nullptr),
                 (world_rank==0 ? displs.data() : nullptr),
                 MPI_DOUBLE, 0, comm);

    if (world_rank == 0) {
        std::ofstream fout("output.txt");
        if (!fout) {
            std::cerr << "Erro: não foi possível abrir output.txt para escrita\n";
            MPI_Abort(comm, 2);
        }
        fout.setf(std::ios::fixed); fout.precision(8);
        for (int i = 0; i < N; i += 16) {
            const double x = i * h;
            for (int j = 0; j < N; j += 16) {
                const double y = j * h;
                fout << x << " " << y << " " << U_global[(size_t)i * N + j] << "\n";
            }
        }
        fout.close();

        // SOMENTE o tempo
        std::cout << "Tempo : " << elapsed_max << " s\n";
    }

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}
