// hopscotch2d_hib_busywait_nobarrier.cpp
// Hopscotch 2D (equação do calor) — HÍBRIDO MPI + OpenMP (busy-wait) SEM barreira global.
// - Decomposição 1D por linhas (interior 1..N-2) entre ranks.
// - Troca de halos MPI assíncrona, feita somente por tid==0 (MPI_THREAD_FUNNELED).
// - Coordenação local entre threads via "progress[]" e espera ocupada (vizinho-a-vizinho).
// - Saída única em "output.txt" (amostrada a cada 16 pontos), escrita pelo rank 0.
// - Stdout: apenas "Tempo : <segundos> s".
//
// Compilar (exemplo):
//   mpicxx -std=c++17 -O3 -fopenmp -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1 \
//          hopscotch2d_hib_busywait_nobarrier.cpp -o hopscotch2d_hib_busywait_nobarrier -lm

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
#include <memory>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
  static inline void spin_pause() noexcept { _mm_pause(); }
#else
  static inline void spin_pause() noexcept { std::this_thread::yield(); }
#endif

// --------- Utils de parsing ---------
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
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    return s;
}

// Leitura estrita de param.txt (todos obrigatórios)
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

// ---- Troca de halos (MPI, funneled) ----
// buf: (local_n+2) x N (linhas 0 e local_n+1 são halos)
static void exchange_halos(double* buf, int N, int local_n,
                           int up, int down, int tag, MPI_Comm comm)
{
    MPI_Request reqs[4]; int rcount = 0;
    if (up != MPI_PROC_NULL) {
        MPI_Irecv(&buf[0*(size_t)N], N, MPI_DOUBLE, up,   tag, comm, &reqs[rcount++]);
        MPI_Isend(&buf[1*(size_t)N], N, MPI_DOUBLE, up,   tag, comm, &reqs[rcount++]);
    }
    if (down != MPI_PROC_NULL) {
        MPI_Irecv(&buf[(size_t)(local_n+1)*N], N, MPI_DOUBLE, down, tag, comm, &reqs[rcount++]);
        MPI_Isend(&buf[(size_t)local_n*N],     N, MPI_DOUBLE, down, tag, comm, &reqs[rcount++]);
    }
    if (rcount) MPI_Waitall(rcount, reqs, MPI_STATUSES_IGNORE);
}

int main(int argc, char** argv)
{
    // ---- MPI init (funneled) ----
    int prov = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);

    MPI_Comm COMM = MPI_COMM_WORLD;
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(COMM, &rank);
    MPI_Comm_size(COMM, &nprocs);

    // ---- Parâmetros globais ----
    int N=0, T=0, TILE=0; double alpha=0.0;
    if (!load_params_strict("param.txt", N, alpha, T, TILE)) {
        MPI_Abort(COMM, 1); return 1;
    }

    // ---- Discretização ----
    const double h   = 1.0 / (N - 1);
    const double dt  = 0.90 * (h*h) / (4.0 * alpha); // 90% do limite estável
    const double lam = alpha * dt / (h*h);
    const double denom = 1.0 + 4.0 * lam;

    // ---- Decomposição por linhas (interior 1..N-2) ----
    const int interior = (N - 2);
    if (interior <= 0) {
        if (rank==0) std::cerr << "Erro: N muito pequeno.\n";
        MPI_Abort(COMM, 1);
    }
    const int base = interior / nprocs;
    const int rem  = interior % nprocs;

    const int local_n = base + ((rank < rem) ? 1 : 0);
    const int start_g = 1 + rank*base + std::min(rank, rem); // 1ª linha interior global do rank
    const int end_g   = start_g + local_n - 1;

    const int up   = (rank > 0)        ? rank-1     : MPI_PROC_NULL;
    const int down = (rank < nprocs-1) ? rank+1     : MPI_PROC_NULL;

    // ---- Arrays locais com halos ----
    const size_t LROWS = (size_t)(local_n + 2);
    std::vector<double> U_new(LROWS * (size_t)N, 0.0), U_old(LROWS * (size_t)N, 0.0);
    auto L = [&](int il, int j)->size_t { return (size_t)il * N + j; }; // il: 0..local_n+1

    // ---- Condição inicial (interior real) ----
    const double D=100.0, x0=0.5, y0=0.5;
    for (int il = 1; il <= local_n; ++il) {
        const int ig = start_g + (il - 1);
        const double x = ig * h;
        for (int j = 1; j <= N-2; ++j) {
            const double y = j * h;
            U_new[L(il,j)] = std::exp(-D*((x-x0)*(x-x0) + (y-y0)*(y-y0)));
        }
    }
    // Contornos 0 (colunas) + topo/fundo se na borda global
    for (int il = 0; il <= local_n+1; ++il) {
        U_new[L(il,0)] = U_new[L(il,N-1)] = 0.0;
        U_old[L(il,0)] = U_old[L(il,N-1)] = 0.0;
    }
    if (up == MPI_PROC_NULL) {
        for (int j=0;j<N;++j){ U_new[L(0,j)]=0.0; U_old[L(0,j)]=0.0; }
    }
    if (down == MPI_PROC_NULL) {
        for (int j=0;j<N;++j){ U_new[L(local_n+1,j)]=0.0; U_old[L(local_n+1,j)]=0.0; }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // ---- Flags atômicas de halos (gates) ----
    std::atomic<int> unew_up_ready{1},   unew_down_ready{1};
    std::atomic<int> uold_up_ready{1},   uold_down_ready{1};

    // ---- Latch de inicialização (evita barreira global) ----
    std::atomic<int> init_done{0};

    // ---- Vetor de progresso por thread (para busy-wait vizinho-a-vizinho) ----
    std::unique_ptr<std::atomic<int>[]> progress;

    // ---- Região paralela OpenMP ----
    #pragma omp parallel default(none) shared(N,T,TILE,lam,denom,U_new,U_old,L,local_n,start_g,up,down,COMM,progress,unew_up_ready,unew_down_ready,uold_up_ready,uold_down_ready,init_done)
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();

        // Divisão de [1..local_n] em nt faixas contíguas
        const int H = (local_n)/nt;
        const int R = (local_n)%nt;
        int iLocal = 1 + tid*H + std::min(tid, R);
        int fLocal = iLocal + H - 1 + (tid < R ? 1 : 0);

        // progress[] compartilhado
        #pragma omp single
        {
            progress.reset(new std::atomic<int>[nt]);
            for (int p=0; p<nt; ++p) progress[p].store(0, std::memory_order_relaxed);
            init_done.store(1, std::memory_order_release);
        }
        while (init_done.load(std::memory_order_acquire) == 0) spin_pause();

        auto wait_neighbors_atleast = [&](int expected){
            if (tid>0)     { while (progress[tid-1].load(std::memory_order_acquire) < expected) spin_pause(); }
            if (tid<nt-1)  { while (progress[tid+1].load(std::memory_order_acquire) < expected) spin_pause(); }
        };

        // Waits de halo apenas quando o tile toca topo/fundo
        auto wait_top_unew  = [&](){ if (up   != MPI_PROC_NULL) while (unew_up_ready.load(std::memory_order_acquire)==0)  spin_pause(); };
        auto wait_down_unew = [&](){ if (down != MPI_PROC_NULL) while (unew_down_ready.load(std::memory_order_acquire)==0) spin_pause(); };
        auto wait_top_uold  = [&](){ if (up   != MPI_PROC_NULL) while (uold_up_ready.load(std::memory_order_acquire)==0)  spin_pause(); };
        auto wait_down_uold = [&](){ if (down != MPI_PROC_NULL) while (uold_down_ready.load(std::memory_order_acquire)==0) spin_pause(); };

        // Auxiliar: o "m" global efetivo em cada fase (sem compartilhar variável)
        auto m_for_phase = [&](int step, int add)->int { return 2*step + add; };

        for (int step=0; step<T; ++step) {

            // Alinhamento local do início do passo (evita avanço do tid==0 antes do restante)
            wait_neighbors_atleast(4*step + 0);


            // ===== Halo U_new para Fase 1 (feito por tid==0 assim que o passo começa) =====
            if (tid == 0) {
                // reseta gates (se houver vizinhos MPI)
                if (up   != MPI_PROC_NULL) unew_up_ready.store(0,   std::memory_order_release);
                if (down != MPI_PROC_NULL) unew_down_ready.store(0, std::memory_order_release);
                exchange_halos(U_new.data(), N, local_n, up, down, 100 + 4*(step%1000000) + 0, COMM);
                if (up   != MPI_PROC_NULL) unew_up_ready.store(1,   std::memory_order_release);
                if (down != MPI_PROC_NULL) unew_down_ready.store(1, std::memory_order_release);
            }

            // ====== FASE 1 (explícita): U_old <- U_new (par) ======
            {
                const int mloc = m_for_phase(step, 0); // 2*step
                for (int ii=iLocal; ii<=fLocal; ii+=TILE){
                    const int i_end = std::min(fLocal, ii + TILE - 1);
                    if (ii == iLocal)    wait_top_unew();
                    if (i_end == fLocal) wait_down_unew();
                    for (int jj=1; jj<=N-2; jj+=TILE){
                        const int j_end = std::min(N-2, jj + TILE - 1);
                        for (int i=ii; i<=i_end; ++i){
                            for (int j=jj; j<=j_end; ++j){
                                // global i = (i + start_g - 1)
                                if ((((i + (start_g-1) + j + mloc) & 1) == 0)){
                                    U_old[L(i,j)] = U_new[L(i,j)] +
                                      lam*( U_new[L(i+1,j)] + U_new[L(i-1,j)]
                                          + U_new[L(i,j+1)] + U_new[L(i,j-1)]
                                          - 4.0*U_new[L(i,j)] );
                                } else {
                                    U_old[L(i,j)] = U_new[L(i,j)];
                                }
                            }
                        }
                    }
                }
            }
            progress[tid].store(4*step + 1, std::memory_order_release);

            // ===== Halo U_old após Fase 1 (quando bordas terminaram F1) =====
            if (tid == 0) {
                // aguarda threads que possuem as linhas 1 e local_n concluírem Fase 1
                while (progress[0].load(std::memory_order_acquire) < 4*step+1) spin_pause();
                if (nt>1) while (progress[nt-1].load(std::memory_order_acquire) < 4*step+1) spin_pause();
                if (up   != MPI_PROC_NULL) uold_up_ready.store(0,   std::memory_order_release);
                if (down != MPI_PROC_NULL) uold_down_ready.store(0, std::memory_order_release);
                exchange_halos(U_old.data(), N, local_n, up, down, 100 + 4*(step%1000000) + 1, COMM);
                if (up   != MPI_PROC_NULL) uold_up_ready.store(1,   std::memory_order_release);
                if (down != MPI_PROC_NULL) uold_down_ready.store(1, std::memory_order_release);
            }

            // ====== FASE 2 (semi-implícita): U_old (ímpar) usando U_old ======
            wait_neighbors_atleast(4*step + 1);
            {
                const int mloc = m_for_phase(step, 0); // 2*step
                for (int ii=iLocal; ii<=fLocal; ii+=TILE){
                    const int i_end = std::min(fLocal, ii + TILE - 1);
                    if (ii == iLocal)    wait_top_uold();
                    if (i_end == fLocal) wait_down_uold();
                    for (int jj=1; jj<=N-2; jj+=TILE){
                        const int j_end = std::min(N-2, jj + TILE - 1);
                        for (int i=ii; i<=i_end; ++i){
                            for (int j=jj; j<=j_end; ++j){
                                if ((((i + (start_g-1) + j + mloc) & 1) == 1)){
                                    U_old[L(i,j)] = ( U_new[L(i,j)]
                                      + lam*( U_old[L(i+1,j)] + U_old[L(i-1,j)]
                                            + U_old[L(i,j+1)] + U_old[L(i,j-1)] ) ) / denom;
                                }
                            }
                        }
                    }
                }
            }
            progress[tid].store(4*step + 2, std::memory_order_release);

            // ===== Halo U_old após Fase 2 (para Fase 3) =====
            if (tid == 0) {
                while (progress[0].load(std::memory_order_acquire) < 4*step+2) spin_pause();
                if (nt>1) while (progress[nt-1].load(std::memory_order_acquire) < 4*step+2) spin_pause();
                if (up   != MPI_PROC_NULL) uold_up_ready.store(0,   std::memory_order_release);
                if (down != MPI_PROC_NULL) uold_down_ready.store(0, std::memory_order_release);
                exchange_halos(U_old.data(), N, local_n, up, down, 100 + 4*(step%1000000) + 2, COMM);
                if (up   != MPI_PROC_NULL) uold_up_ready.store(1,   std::memory_order_release);
                if (down != MPI_PROC_NULL) uold_down_ready.store(1, std::memory_order_release);
            }

            // ====== FASE 3 (explícita): U_new (par relativo a 2*step+1) a partir de U_old ======
            wait_neighbors_atleast(4*step + 2);
            {
                const int mloc = m_for_phase(step, 1); // 2*step + 1
                for (int ii=iLocal; ii<=fLocal; ii+=TILE){
                    const int i_end = std::min(fLocal, ii + TILE - 1);
                    if (ii == iLocal)    wait_top_uold();
                    if (i_end == fLocal) wait_down_uold();
                    for (int jj=1; jj<=N-2; jj+=TILE){
                        const int j_end = std::min(N-2, jj + TILE - 1);
                        for (int i=ii; i<=i_end; ++i){
                            for (int j=jj; j<=j_end; ++j){
                                if ((((i + (start_g-1) + j + mloc) & 1) == 0)){
                                    U_new[L(i,j)] = U_old[L(i,j)] +
                                      lam*( U_old[L(i+1,j)] + U_old[L(i-1,j)]
                                          + U_old[L(i,j+1)] + U_old[L(i,j-1)]
                                          - 4.0*U_old[L(i,j)] );
                                } else {
                                    U_new[L(i,j)] = U_old[L(i,j)];
                                }
                            }
                        }
                    }
                }
            }
            progress[tid].store(4*step + 3, std::memory_order_release);

            // ===== Halo U_new após Fase 3 (para Fase 4) =====
            if (tid == 0) {
                while (progress[0].load(std::memory_order_acquire) < 4*step+3) spin_pause();
                if (nt>1) while (progress[nt-1].load(std::memory_order_acquire) < 4*step+3) spin_pause();
                if (up   != MPI_PROC_NULL) unew_up_ready.store(0,   std::memory_order_release);
                if (down != MPI_PROC_NULL) unew_down_ready.store(0, std::memory_order_release);
                exchange_halos(U_new.data(), N, local_n, up, down, 100 + 4*(step%1000000) + 3, COMM);
                if (up   != MPI_PROC_NULL) unew_up_ready.store(1,   std::memory_order_release);
                if (down != MPI_PROC_NULL) unew_down_ready.store(1, std::memory_order_release);
            }

            // ====== FASE 4 (semi-implícita): U_new (ímpar relativo a 2*step+1) usando U_new ======
            wait_neighbors_atleast(4*step + 3);
            {
                const int mloc = m_for_phase(step, 1); // 2*step + 1
                for (int ii=iLocal; ii<=fLocal; ii+=TILE){
                    const int i_end = std::min(fLocal, ii + TILE - 1);
                    if (ii == iLocal)    wait_top_unew();
                    if (i_end == fLocal) wait_down_unew();
                    for (int jj=1; jj<=N-2; jj+=TILE){
                        const int j_end = std::min(N-2, jj + TILE - 1);
                        for (int i=ii; i<=i_end; ++i){
                            for (int j=jj; j<=j_end; ++j){
                                if ((((i + (start_g-1) + j + mloc) & 1) == 1)){
                                    U_new[L(i,j)] = ( U_old[L(i,j)]
                                      + lam*( U_new[L(i+1,j)] + U_new[L(i-1,j)]
                                            + U_new[L(i,j+1)] + U_new[L(i,j-1)] ) ) / denom;
                                }
                            }
                        }
                    }
                }
            }
            progress[tid].store(4*step + 4, std::memory_order_release);
            // SEM barreira global — início do próximo passo usará apenas vizinhos.
        } // step
    } // parallel

    MPI_Barrier(COMM);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    // Tempo global (máximo entre ranks)
    double secs_max = 0.0;
    MPI_Reduce(&secs, &secs_max, 1, MPI_DOUBLE, MPI_MAX, 0, COMM);

    // ---- Apenas rank 0 imprime o tempo ----
    if (rank == 0) {
        std::cout << "Tempo : " << secs_max << " s\n";
    }

    // ---- Saída única (rank 0) — amostrada a cada 16 pontos ----
    // Monta fragmento local em ordem crescente de i global.
    std::ostringstream oss;
    oss.setf(std::ios::fixed); oss.precision(8);

    // i=0 (borda) apenas no rank 0, se múltiplo de 16 (sempre é)
    if (rank == 0) {
        const int i = 0; const double x = i * h;
        for (int j=0; j<N; j+=16) {
            const double y = j*h;
            oss << x << " " << y << " " << 0.0 << "\n";
        }
    }

    // Interiores deste rank: alinhar primeiro múltiplo de 16 dentro de [start_g..end_g]
    int ig_first = ((start_g + 15) / 16) * 16; // primeiro i múltiplo de 16 >= start_g
    for (int ig = ig_first; ig <= end_g; ig += 16) {
        const int il = (ig - start_g) + 1;
        const double x = ig * h;
        for (int j=0; j<N; j+=16) {
            const double y = j * h;
            double v = (j==0 || j==N-1) ? 0.0 : U_new[(size_t)il * N + j];
            oss << x << " " << y << " " << v << "\n";
        }
    }

    // i=N-1 (borda) apenas no último rank
    if (rank == nprocs-1) {
        const int i = N-1; const double x = i * h;
        for (int j=0; j<N; j+=16) {
            const double y = j*h;
            oss << x << " " << y << " " << 0.0 << "\n";
        }
    }

    std::string local_txt = oss.str();
    int local_len = (int)local_txt.size();

    // Coleta lengths
    std::vector<int> recvcounts, displs;
    if (rank == 0) recvcounts.resize(nprocs);
    MPI_Gather(&local_len, 1, MPI_INT,
               rank==0 ? recvcounts.data() : nullptr, 1, MPI_INT,
               0, COMM);

    // Coleta texto
    if (rank == 0) {
        displs.resize(nprocs);
        int total = 0;
        for (int p=0; p<nprocs; ++p) { displs[p] = total; total += recvcounts[p]; }
        std::vector<char> allbuf(total);
        MPI_Gatherv(local_txt.data(), local_len, MPI_CHAR,
                    allbuf.data(), recvcounts.data(), displs.data(), MPI_CHAR,
                    0, COMM);
        std::ofstream fout("output.txt");
        if (fout) { fout.write(allbuf.data(), (std::streamsize)allbuf.size()); }
    } else {
        MPI_Gatherv(local_txt.data(), local_len, MPI_CHAR,
                    nullptr, nullptr, nullptr, MPI_CHAR,
                    0, COMM);
    }

    MPI_Finalize();
    return 0;
}
