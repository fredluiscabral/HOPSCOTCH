// hopscotch2d_hib_sem_nobarrier.cpp
// Hopscotch 2D (equação do calor) — HÍBRIDO MPI + OpenMP + semáforos POSIX
// - Sincronização local vizinho-a-vizinho por fase (semáforos).
// - "Rendezvous" local por passo (sem barreira global).
// - Troca de halos MPI assíncrona, apenas tid==0 (MPI_THREAD_FUNNELED).
// - Apenas o rank 0 escreve um único "output.txt" (amostrado a cada 16 pontos).
// - Em stdout: APENAS "Tempo : <segundos> s".
//
// Compilar (exemplo):
//   mpicxx -std=c++17 -O3 -fopenmp -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1 \
//          hopscotch2d_hib_sem_nobarrier.cpp -o hopscotch2d_hib_sem_nobarrier -lm

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
#include <chrono>
#include <cctype>
#include <atomic>
#include <thread>
#include <semaphore.h>

#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
  static inline void spin_pause() noexcept { _mm_pause(); }
#else
  static inline void spin_pause() noexcept { std::this_thread::yield(); }
#endif

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
    if (!fin) { std::cerr << "Erro: não foi possível abrir " << fname << ".\n"; return false; }

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
        N     = std::stoi(kv.at("n"));
        alpha = std::stod(kv.at("alpha"));
        T     = std::stoi(kv.at("t"));
        TILE  = std::stoi(kv.at("tile"));
    } catch (const std::exception& e) {
        std::cerr << "Erro: conversão de parâmetros: " << e.what() << "\n";
        return false;
    }

    if (N < 3)             { std::cerr << "Erro: N>=3.\n"; return false; }
    if (alpha <= 0.0)      { std::cerr << "Erro: alpha>0.\n"; return false; }
    if (T < 0)             { std::cerr << "Erro: T>=0.\n"; return false; }
    if (TILE < 1 || TILE > N-2) {
        std::cerr << "Erro: 1<=tile<=N-2.\n"; return false;
    }
    return true;
}

// --------------------- troca de halos (MPI, funneled) ---------------------
static void exchange_halos(double* buf, int N, int local_n,
                           int up, int down, int tag, MPI_Comm comm)
{
    // buf layout: (local_n+2) x N, com linhas 0 e local_n+1 como halos
    // Envia linha 1 ao 'up' e recebe no halo 0; envia linha local_n ao 'down' e recebe no halo local_n+1.
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
    // MPI init (funneled: só uma thread chama MPI)
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

    const double h   = 1.0 / (N - 1);
    const double dt  = 0.90 * (h*h) / (4.0 * alpha);
    const double lam = alpha * dt / (h*h);
    const double denom = 1.0 + 4.0 * lam;

    // ---- Decomposição por linhas (somente interior 1..N-2) ----
    const int interior = (N - 2);
    if (interior <= 0) {
        if (rank==0) std::cerr << "Erro: N muito pequeno.\n";
        MPI_Abort(COMM, 1);
    }
    const int base = interior / nprocs;
    const int rem  = interior % nprocs;

    const int local_n = base + ((rank < rem) ? 1 : 0);         // nº de linhas interiores locais
    const int start_g = 1 + rank*base + std::min(rank, rem);   // primeira linha interior global do rank
    const int end_g   = start_g + local_n - 1;                 // última linha interior global do rank

    const int up   = (rank > 0)          ? rank-1 : MPI_PROC_NULL;
    const int down = (rank < nprocs-1)   ? rank+1 : MPI_PROC_NULL;

    // ---- Arrays locais com halos ----
    const size_t LROWS = (size_t)(local_n + 2); // inclui 2 halos
    std::vector<double> U_new(LROWS * (size_t)N, 0.0), U_old(LROWS * (size_t)N, 0.0);
    auto L = [&](int il, int j)->size_t { return (size_t)il * N + j; }; // il: 0..local_n+1 ; j: 0..N-1

    // ---- Condição inicial (somente interior real: il=1..local_n , j=1..N-2) ----
    const double D=100.0, x0=0.5, y0=0.5;
    for (int il = 1; il <= local_n; ++il) {
        const int ig = start_g + (il - 1);
        const double x = ig * h;
        for (int j = 1; j <= N-2; ++j) {
            const double y = j * h;
            U_new[L(il,j)] = std::exp(-D*((x-x0)*(x-x0) + (y-y0)*(y-y0)));
        }
    }
    // Contornos 0 (j=0, j=N-1) + bordas topo/fundo se aplicável
    for (int il = 0; il <= local_n+1; ++il) {
        U_new[L(il,0)]   = 0.0; U_new[L(il,N-1)] = 0.0;
        U_old[L(il,0)]   = 0.0; U_old[L(il,N-1)] = 0.0;
    }
    if (up == MPI_PROC_NULL)   { // bordo global i=0
        for (int j=0;j<N;++j){ U_new[L(0,j)]=0.0; U_old[L(0,j)]=0.0; }
    }
    if (down == MPI_PROC_NULL) { // bordo global i=N-1
        for (int j=0;j<N;++j){ U_new[L(local_n+1,j)]=0.0; U_old[L(local_n+1,j)]=0.0; }
    }

    // Sincroniza antes de cronometrar (tempo global)
    MPI_Barrier(COMM);
    auto t0 = std::chrono::high_resolution_clock::now();

    // ---------------- OpenMP região paralela ----------------
    // Vetores de semáforos para sincronização local por fase e por passo:
    std::vector<sem_t> sem_left, sem_right;     // sinalização por FASE
    std::vector<sem_t> step_left, step_right;   // rendezvous por PASSO

    // Gates atômicos para halos (só tiles de borda esperam)
    static std::atomic<int> unew_up_ready{1}, unew_down_ready{1};
    static std::atomic<int> uold_up_ready{1}, uold_down_ready{1};

    // Latch de inicialização (evita barreira global explícita)
    static std::atomic<int> init_done{0};

    #pragma omp parallel default(none) \
        shared(N, T, TILE, lam, denom, U_new, U_old, L, local_n, start_g, up, down, COMM, \
               sem_left, sem_right, step_left, step_right, init_done, \
               unew_up_ready, unew_down_ready, uold_up_ready, uold_down_ready)
    {
        const int nt  = omp_get_num_threads();
        const int tid = omp_get_thread_num();

        // --------- Inicialização sem barreira explícita ---------
        #pragma omp single nowait
        {
            sem_left.resize(nt);  sem_right.resize(nt);
            step_left.resize(nt); step_right.resize(nt);
            for (int t = 0; t < nt; ++t) {
                sem_init(&sem_left[t],  0, 0);
                sem_init(&sem_right[t], 0, 0);
                sem_init(&step_left[t],  0, 0);
                sem_init(&step_right[t], 0, 0);
            }
            init_done.store(1, std::memory_order_release);
        }
        while (init_done.load(std::memory_order_acquire) == 0) { spin_pause(); }

        // Particionamento em faixas locais (linhas 1..local_n)
        const int baseSize = (local_n) / nt;
        const int sobra    = (local_n) % nt;
        int iLocal = 1 + tid*baseSize + std::min(tid, sobra);
        int mySize = baseSize + (tid < sobra ? 1 : 0);
        int fLocal = iLocal + mySize - 1;

        // Auxiliares de fase (semáforos vizinho-a-vizinho)
        auto signal_done_phase = [&](int t) {
            sem_post(&sem_left[t]);
            sem_post(&sem_right[t]);
        };
        auto wait_neighbors_phase = [&](int t) {
            if (t > 0)     sem_wait(&sem_right[t-1]); // espera esquerda
            if (t < nt-1)  sem_wait(&sem_left[t+1]);  // espera direita
        };
        // Rendezvous local por passo
        auto announce_step = [&](int t) {
            sem_post(&step_left[t]);
            sem_post(&step_right[t]);
        };
        auto wait_step_neighbors = [&](int t) {
            if (t > 0)     sem_wait(&step_right[t-1]);
            if (t < nt-1)  sem_wait(&step_left[t+1]);
        };

        // Helpers: esperar halo apenas quando o tile toca a borda
        auto wait_top_unew  = [&](){ if (up   != MPI_PROC_NULL) while (unew_up_ready.load(std::memory_order_acquire)==0)  spin_pause(); };
        auto wait_down_unew = [&](){ if (down != MPI_PROC_NULL) while (unew_down_ready.load(std::memory_order_acquire)==0) spin_pause(); };
        auto wait_top_uold  = [&](){ if (up   != MPI_PROC_NULL) while (uold_up_ready.load(std::memory_order_acquire)==0)  spin_pause(); };
        auto wait_down_uold = [&](){ if (down != MPI_PROC_NULL) while (uold_down_ready.load(std::memory_order_acquire)==0) spin_pause(); };

        for (int s=0; s<T; ++s) {
            // Mantém passos sincronizados SOMENTE com vizinhos do time
            if (s > 0) { wait_step_neighbors(tid); }

            // =================== Troca de halo U_new (para FASE 1) ===================
            if (tid == 0) {
                unew_up_ready.store( (up   == MPI_PROC_NULL)?1:0, std::memory_order_release);
                unew_down_ready.store((down == MPI_PROC_NULL)?1:0, std::memory_order_release);
                exchange_halos(U_new.data(), N, local_n, up, down, 100 + 4*(s%1000000) + 0, COMM);
                unew_up_ready.store(1, std::memory_order_release);
                unew_down_ready.store(1, std::memory_order_release);
            }

            // ------------------- FASE 1 (explícita) -------------------
            // U_old <- U_new (par), U_old <- U_new (cópia) (ímpar)
            for (int ii=iLocal; ii<=fLocal; ii+=TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                if (ii == iLocal)      wait_top_unew();
                if (i_end == fLocal)   wait_down_unew();

                for (int jj=1; jj<=N-2; jj+=TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i=ii; i<=i_end; ++i) {
                        for (int j=jj; j<=j_end; ++j) {
                            if ( ((i + (start_g-1) + j + 2*s) & 1) == 0 ) { // mloc=2*s
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
            signal_done_phase(tid);
            wait_neighbors_phase(tid);

            // =================== Troca de halo U_old (após FASE 1) ===================
            if (tid == 0) {
                uold_up_ready.store( (up   == MPI_PROC_NULL)?1:0, std::memory_order_release);
                uold_down_ready.store((down == MPI_PROC_NULL)?1:0, std::memory_order_release);
                exchange_halos(U_old.data(), N, local_n, up, down, 100 + 4*(s%1000000) + 1, COMM);
                uold_up_ready.store(1, std::memory_order_release);
                uold_down_ready.store(1, std::memory_order_release);
            }

            // ------------------- FASE 2 (semi-implícita) -------------------
            // U_old (ímpar) usa vizinhos U_old
            for (int ii=iLocal; ii<=fLocal; ii+=TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                if (ii == iLocal)      wait_top_uold();
                if (i_end == fLocal)   wait_down_uold();

                for (int jj=1; jj<=N-2; jj+=TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i=ii; i<=i_end; ++i) {
                        for (int j=jj; j<=j_end; ++j) {
                            if ( ((i + (start_g-1) + j + 2*s) & 1) == 1 ) {
                                U_old[L(i,j)] = ( U_new[L(i,j)]
                                  + lam*( U_old[L(i+1,j)] + U_old[L(i-1,j)]
                                        + U_old[L(i,j+1)] + U_old[L(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            signal_done_phase(tid);
            wait_neighbors_phase(tid);
            // mloc++ => (2*s + 1) efetivo nas próximas fases

            // =================== Troca de halo U_old (após FASE 2) ===================
            if (tid == 0) {
                uold_up_ready.store( (up   == MPI_PROC_NULL)?1:0, std::memory_order_release);
                uold_down_ready.store((down == MPI_PROC_NULL)?1:0, std::memory_order_release);
                exchange_halos(U_old.data(), N, local_n, up, down, 100 + 4*(s%1000000) + 2, COMM);
                uold_up_ready.store(1, std::memory_order_release);
                uold_down_ready.store(1, std::memory_order_release);
            }

            // ------------------- FASE 3 (explícita) -------------------
            // U_new (par relativo a mloc=2*s+1) a partir de U_old
            for (int ii=iLocal; ii<=fLocal; ii+=TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                if (ii == iLocal)      wait_top_uold();
                if (i_end == fLocal)   wait_down_uold();

                for (int jj=1; jj<=N-2; jj+=TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i=ii; i<=i_end; ++i) {
                        for (int j=jj; j<=j_end; ++j) {
                            if ( ((i + (start_g-1) + j + (2*s+1)) & 1) == 0 ) {
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
            signal_done_phase(tid);
            wait_neighbors_phase(tid);

            // =================== Troca de halo U_new (após FASE 3) ===================
            if (tid == 0) {
                unew_up_ready.store( (up   == MPI_PROC_NULL)?1:0, std::memory_order_release);
                unew_down_ready.store((down == MPI_PROC_NULL)?1:0, std::memory_order_release);
                exchange_halos(U_new.data(), N, local_n, up, down, 100 + 4*(s%1000000) + 3, COMM);
                unew_up_ready.store(1, std::memory_order_release);
                unew_down_ready.store(1, std::memory_order_release);
            }

            // ------------------- FASE 4 (semi-implícita) -------------------
            // U_new (ímpar relativo a mloc=2*s+1) usa vizinhos U_new
            for (int ii=iLocal; ii<=fLocal; ii+=TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                if (ii == iLocal)      wait_top_unew();
                if (i_end == fLocal)   wait_down_unew();

                for (int jj=1; jj<=N-2; jj+=TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i=ii; i<=i_end; ++i) {
                        for (int j=jj; j<=j_end; ++j) {
                            if ( ((i + (start_g-1) + j + (2*s+1)) & 1) == 1 ) {
                                U_new[L(i,j)] = ( U_old[L(i,j)]
                                  + lam*( U_new[L(i+1,j)] + U_new[L(i-1,j)]
                                        + U_new[L(i,j+1)] + U_new[L(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            signal_done_phase(tid);
            wait_neighbors_phase(tid);
            // completa o passo
            announce_step(tid);
        } // s
        // Destrói semáforos (uma vez)
        #pragma omp single nowait
        {
            for (int t=0; t<nt; ++t) {
                sem_destroy(&sem_left[t]);
                sem_destroy(&sem_right[t]);
                sem_destroy(&step_left[t]);
                sem_destroy(&step_right[t]);
            }
        }
    } // fim região paralela

    MPI_Barrier(COMM);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    // Tempo global (máximo entre ranks)
    double secs_max = 0.0;
    MPI_Reduce(&secs, &secs_max, 1, MPI_DOUBLE, MPI_MAX, 0, COMM);

    // ---- Apenas o tempo em stdout ----
    if (rank == 0) {
        std::cout << "Tempo : " << secs_max << " s\n";
    }

    // ---------------- Saída única (rank 0) ----------------
    // Amostragem de 16 em 16, ordenada por i crescente.
    // Cada rank envia somente suas linhas interiores (i=start_g..end_g) múltiplas de 16.
    // Rank 0 adiciona linha i=0; último rank adiciona linha i=N-1, se múltiplas de 16.
    std::ostringstream oss;
    oss.setf(std::ios::fixed); oss.precision(8);

    // Se este rank possui a linha global 0 (rank 0), adiciona i=0
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

    // Linhas interiores deste rank
    for (int ig = start_g; ig <= end_g; ig += 16) {
        if (ig % 16 != 0) continue;
        const int il = (ig - start_g) + 1;
        const double x = ig * h;
        for (int j=0; j<N; j+=16) {
            const double y = j * h;
            double v = 0.0;
            if (j==0 || j==N-1) {
                v = 0.0;
            } else {
                v = U_new[(size_t)il * N + j];
            }
            oss << x << " " << y << " " << v << "\n";
        }
    }

    // Se este é o último rank, adiciona i=N-1
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

    // Coletar comprimentos
    std::vector<int> recvcounts, displs;
    if (rank == 0) { recvcounts.resize(nprocs); }
    MPI_Gather(&local_len, 1, MPI_INT,
               rank==0 ? recvcounts.data() : nullptr, 1, MPI_INT,
               0, COMM);

    // Coletar texto
    std::vector<char> allbuf;
    if (rank == 0) {
        displs.resize(nprocs);
        int total = 0;
        for (int p=0; p<nprocs; ++p) { displs[p] = total; total += recvcounts[p]; }
        allbuf.resize(total);
        MPI_Gatherv(local_txt.data(), local_len, MPI_CHAR,
                    allbuf.data(), recvcounts.data(), displs.data(), MPI_CHAR,
                    0, COMM);

        // Concatena na ordem dos ranks (já corresponde a i crescente)
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
