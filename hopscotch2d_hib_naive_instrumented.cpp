// hopscotch2d_hib_naive_instrumented.cpp
// Equação do calor 2D (Hopscotch) — versão HÍBRIDA MPI+OpenMP com block tiling (naive).
//
// Instrumentação para validação do modelo (sec:validacao-metodologia):
//  - Por fase φ=1..4 e por passo: T_phi (parede), T_step (parede)
//  - Por thread e fase: C_t^(φ) (tempo de computação do trecho do stencil; exclui barreiras)
//    -> reporta max/min/mean por fase.
//  - Comunicação (híbrido): T_comm^(φ) ao redor de exchange_halo() (MPI_Wtime).
//  - Estimativa observável do custo de barreira por fase (naive, sem sobreposição):
//        Cbar_hat^(φ) = T_phi - max_t C_t^(φ) - T_comm^(φ)
//
// Saída de métricas:
//  - Um arquivo JSONL por rank.
//  - Diretório via env METRICS_PATH (default: .)
//    Arquivo: <METRICS_PATH>/hib_naive.rank<rank>.jsonl
//  - Variáveis úteis:
//      METRICS=0/1            (default 1)
//      METRICS_DETAIL=0/1     (default 0)  -> log por passo e por fase (muito mais linhas)
//      DELAY_US (>=0)         (default 0)  -> atraso artificial (microsegundos) para E1 (beta)
//      DELAY_TID (>=0)        (default 0)  -> thread alvo do atraso
//      DELAY_PHASE (0..4)     (default 0)  -> fase alvo (0 desliga; 1..4 ativa)
//
// Compilar:
//   mpicxx -O3 -march=native -fopenmp -pthread -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1 \
//          hopscotch2d_hib_naive_instrumented.cpp -o hopscotch2d_hib_naive_instrumented -lm
//
// Executar (exemplo OpenMPI):
//   export OMP_PLACES=cores OMP_PROC_BIND=spread OMP_NUM_THREADS=16
//   export METRICS_PATH=./metrics
//   mpirun -np 2 --bind-to core --map-by ppr:1:socket:PE=16 ./hopscotch2d_hib_naive_instrumented
//
// Observação: a saída principal (stdout) e output.txt seguem o comportamento do código original.

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
#include <cstdint>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

// ----------------- Utils -----------------
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

static inline int getenv_int(const char* name, int defv) {
    const char* s = std::getenv(name);
    if (!s || !*s) return defv;
    try { return std::stoi(s); } catch (...) { return defv; }
}
static inline std::string getenv_str(const char* name, const std::string& defv) {
    const char* s = std::getenv(name);
    if (!s || !*s) return defv;
    return std::string(s);
}
static inline void spin_delay_us(int delay_us) {
    if (delay_us <= 0) return;
    const double t0 = omp_get_wtime();
    const double dt = static_cast<double>(delay_us) * 1e-6;
    while ((omp_get_wtime() - t0) < dt) { /* busy wait */ }
}

// índice linear local (com halo) — L = (nj+2)
static inline size_t idx2(int i, int j, int L) {
    return static_cast<size_t>(i) * static_cast<size_t>(L) + static_cast<size_t>(j);
}

static bool load_params_strict_rank0(const std::string& fname,
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

// Divisão 1D equilibrada
static inline void split_range(int total, int parts, int coord, int& start, int& count){
    const int base = total / parts;
    const int rem  = total % parts;
    count = base + (coord < rem ? 1 : 0);
    start = coord * base + std::min(coord, rem);
}

// Troca de halos (Sendrecv). A tem dimensões locais (ni+2) x (nj+2)
static void exchange_halo(std::vector<double>& A, int ni, int nj,
                          int nbr_north, int nbr_south, int nbr_west, int nbr_east,
                          MPI_Datatype col_t, MPI_Comm comm)
{
    const int L = nj + 2;
    MPI_Status status;

    // Norte: i=1 <-> halo i=0
    if (nbr_north != MPI_PROC_NULL) {
        MPI_Sendrecv(&A[idx2(1,     1, L)], nj, MPI_DOUBLE, nbr_north, 10,
                     &A[idx2(0,     1, L)], nj, MPI_DOUBLE, nbr_north, 11,
                     comm, &status);
    }
    // Sul: i=ni <-> halo i=ni+1
    if (nbr_south != MPI_PROC_NULL) {
        MPI_Sendrecv(&A[idx2(ni,    1, L)], nj, MPI_DOUBLE, nbr_south, 11,
                     &A[idx2(ni+1,  1, L)], nj, MPI_DOUBLE, nbr_south, 10,
                     comm, &status);
    }
    // Oeste: j=1 <-> halo j=0
    if (nbr_west != MPI_PROC_NULL) {
        MPI_Sendrecv(&A[idx2(1, 1, L)], 1, col_t, nbr_west, 20,
                     &A[idx2(1, 0, L)], 1, col_t, nbr_west, 21,
                     comm, &status);
    }
    // Leste: j=nj <-> halo j=nj+1
    if (nbr_east != MPI_PROC_NULL) {
        MPI_Sendrecv(&A[idx2(1, nj,   L)], 1, col_t, nbr_east, 21,
                     &A[idx2(1, nj+1, L)], 1, col_t, nbr_east, 20,
                     comm, &status);
    }
}

static inline void json_write_array4(std::ostream& os, const double a[4]) {
    os << "[" << a[0] << "," << a[1] << "," << a[2] << "," << a[3] << "]";
}
static inline void json_write_array4_i(std::ostream& os, const int a[4]) {
    os << "[" << a[0] << "," << a[1] << "," << a[2] << "," << a[3] << "]";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank=0, world_size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Rank 0 lê parâmetros e difunde
    int N=0, T=0, TILE=0; double alpha=0.0;
    if (world_rank==0) {
        if (!load_params_strict_rank0("param.txt", N, alpha, T, TILE)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&N, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&TILE, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Instrumentação: flags
    const int metrics_on = getenv_int("METRICS", 1);
    const int metrics_detail = getenv_int("METRICS_DETAIL", 0);

    // Delay artificial (E1 - beta)
    const int delay_us   = std::max(0, getenv_int("DELAY_US", 0));
    const int delay_tid  = std::max(0, getenv_int("DELAY_TID", 0));
    const int delay_phi  = std::max(0, getenv_int("DELAY_PHASE", 0)); // 0..4

    // Discretização
    const double h   = 1.0 / (N - 1);
    const double dt  = 0.90 * (h*h) / (4.0 * alpha);
    const double lam = alpha * dt / (h*h);
    const double denom = 1.0 + 4.0 * lam;

    // Grade cartesiana 2D de ranks
    int dims[2] = {0,0};
    MPI_Dims_create(world_size, 2, dims); // PxQ equilibrado
    int periods[2] = {0,0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/1, &cart);

    int coords[2];
    MPI_Cart_coords(cart, world_rank, 2, coords);
    const int Px = dims[0], Qy = dims[1];
    const int px = coords[0], qy = coords[1];

    // Vizinhos (N,S,W,E)
    int nbr_north, nbr_south, nbr_west, nbr_east;
    MPI_Cart_shift(cart, 0, +1, &nbr_north, &nbr_south);
    MPI_Cart_shift(cart, 1, +1, &nbr_west,  &nbr_east);

    // Decomposição do domínio interior [1..N-2] x [1..N-2]
    const int interior = N-2;
    int istart_local, icount_local, jstart_local, jcount_local;
    split_range(interior, Px, px, istart_local, icount_local);
    split_range(interior, Qy, qy, jstart_local, jcount_local);

    // Índices globais iniciais da minha submalha interior
    const int ig0 = 1 + istart_local;
    const int jg0 = 1 + jstart_local;

    const int ni = icount_local;
    const int nj = jcount_local;

    // Alocação local com halo: (ni+2) x (nj+2)
    const int L = nj + 2;
    const size_t NN_local = static_cast<size_t>(ni+2) * static_cast<size_t>(nj+2);
    std::vector<double> U_new(NN_local, 0.0), U_old(NN_local, 0.0);

    // Condição inicial (gaussiana) no interior global
    const double D=100.0, x0=0.5, y0=0.5;
    for (int i=1; i<=ni; ++i){
        const int ig = ig0 + (i-1);
        const double x = ig * h;
        for (int j=1; j<=nj; ++j){
            const int jg = jg0 + (j-1);
            const double y = jg * h;
            U_new[idx2(i,j,L)] = std::exp(-D*((x-x0)*(x-x0)+(y-y0)*(y-y0)));
        }
    }

    // Datatype de coluna (halo vertical)
    MPI_Datatype COL_T;
    MPI_Type_vector(ni, /*blocklen=*/1, /*stride=*/L, MPI_DOUBLE, &COL_T);
    MPI_Type_commit(&COL_T);

    // Halo inicial de U_new (para o passo 0)
    exchange_halo(U_new, ni, nj, nbr_north, nbr_south, nbr_west, nbr_east, COL_T, cart);

    // ===== Log de métricas (por rank) =====
    std::ofstream mlog;
    std::string metrics_dir = getenv_str("METRICS_PATH", ".");
    std::string omp_places = getenv_str("OMP_PLACES", "");
    std::string omp_bind   = getenv_str("OMP_PROC_BIND", "");

    if (metrics_on) {
        try {
            fs::create_directories(metrics_dir);
        } catch (...) {
            if (world_rank==0) {
                std::cerr << "Aviso: não consegui criar METRICS_PATH='" << metrics_dir
                          << "'. Usando diretório atual.\n";
            }
            metrics_dir = ".";
        }
        std::ostringstream fn;
        fn << metrics_dir << "/hib_naive.rank" << world_rank << ".jsonl";
        mlog.open(fn.str(), std::ios::out);
        if (!mlog) {
            std::cerr << "Erro: não foi possível abrir arquivo de métricas '" << fn.str() << "'.\n";
            MPI_Abort(cart, 2);
        }
        mlog.setf(std::ios::fixed);
        mlog << std::setprecision(9);

        mlog << "{"
             << "\"type\":\"run\","
             << "\"variant\":\"hib_naive\","
             << "\"rank\":" << world_rank << ",\"ranks\":" << world_size
             << ",\"Px\":" << Px << ",\"Qy\":" << Qy
             << ",\"N\":" << N << ",\"alpha\":" << alpha << ",\"T\":" << T << ",\"tile\":" << TILE
             << ",\"ni\":" << ni << ",\"nj\":" << nj
             << ",\"dt\":" << dt << ",\"lam\":" << lam
             << ",\"omp_places\":\"" << omp_places << "\","
             << "\"omp_proc_bind\":\"" << omp_bind << "\","
             << "\"delay_us\":" << delay_us << ",\"delay_tid\":" << delay_tid << ",\"delay_phase\":" << delay_phi
             << "}\n";
    }

    // Integração temporal
    int m = 0; // paridade global

    MPI_Barrier(cart);
    const double t0 = MPI_Wtime();

    const int max_threads = std::max(1, omp_get_max_threads());
    std::vector<double> comp(4 * static_cast<size_t>(max_threads), 0.0);

    // agregadores (somatórios em steps)
    double Tphi_sum[4] = {0,0,0,0};
    double comm_sum[4] = {0,0,0,0};
    double compmax_sum[4] = {0,0,0,0};
    double compmin_sum[4] = {0,0,0,0};
    double compmean_sum[4] = {0,0,0,0};
    double deltaimb_sum[4] = {0,0,0,0};
    double cbarhat_sum[4] = {0,0,0,0};
    double Tstep_sum = 0.0;
    int actual_nt = 0;

    
    // Variáveis de timing compartilhadas (single/master podem ser executados por threads diferentes)
    double step_t0_s = 0.0;
    double phi_t0_s  = 0.0;
    double comm_phi_s = 0.0;


    #pragma omp parallel default(none) \
        shared(N,T,TILE,lam,denom,U_new,U_old,m,ni,nj,ig0,jg0,cart, \
               nbr_north,nbr_south,nbr_west,nbr_east,COL_T,L, \
               metrics_on,metrics_detail,mlog,comp,max_threads, \
               Tphi_sum,comm_sum,compmax_sum,compmin_sum,compmean_sum,deltaimb_sum,cbarhat_sum, \
               Tstep_sum,actual_nt,world_rank,world_size,Px,Qy,alpha,dt,delay_us,delay_tid,delay_phi, \
               step_t0_s,phi_t0_s,comm_phi_s)
    {
        const int tid = omp_get_thread_num();

        for (int step=0; step<T; ++step) {

            #pragma omp single
            {
                step_t0_s = omp_get_wtime();
                if (actual_nt == 0) actual_nt = omp_get_num_threads();
            }

            // ----------------- FASE 1 -----------------
            #pragma omp single
            { phi_t0_s = omp_get_wtime(); }

            double c0 = omp_get_wtime();
            #pragma omp for collapse(2) schedule(static) nowait
            for (int ii=1; ii<=ni; ii+=TILE){
                for (int jj=1; jj<=nj; jj+=TILE){
                    const int i_end = std::min(ni, ii+TILE-1);
                    const int j_end = std::min(nj, jj+TILE-1);
                    for (int i=ii; i<=i_end; ++i){
                        const int ig = ig0 + (i-1);
                        for (int j=jj; j<=j_end; ++j){
                            const int jg = jg0 + (j-1);
                            if ( ((ig + jg + m) & 1) == 0 ){
                                U_old[idx2(i,j,L)] = U_new[idx2(i,j,L)] +
                                  lam*( U_new[idx2(i+1,j,  L)] + U_new[idx2(i-1,j,  L)]
                                      + U_new[idx2(i,  j+1,L)] + U_new[idx2(i,  j-1,L)]
                                      - 4.0*U_new[idx2(i,j,L)] );
                            } else {
                                U_old[idx2(i,j,L)] = U_new[idx2(i,j,L)];
                            }
                        }
                    }
                }
            }
            if (delay_us > 0 && delay_phi == 1 && tid == delay_tid) spin_delay_us(delay_us);
            double c1 = omp_get_wtime();
            comp[0*max_threads + tid] = c1 - c0;

            #pragma omp barrier
            #pragma omp master
            {
                const double cs = MPI_Wtime();
                exchange_halo(U_old, ni, nj, nbr_north, nbr_south, nbr_west, nbr_east, COL_T, cart);
                const double ce = MPI_Wtime();
                comm_phi_s = ce - cs;
            }
            #pragma omp barrier

            #pragma omp single
            {
                const double phi_t1 = omp_get_wtime();
                const int nt = omp_get_num_threads();

                double cmax = -1e300, cmin = 1e300, csum=0.0;
                for (int t=0; t<nt; ++t) {
                    const double ct = comp[0*max_threads + t];
                    cmax = std::max(cmax, ct);
                    cmin = std::min(cmin, ct);
                    csum += ct;
                }
                const double cmean = csum / nt;
                const double Tphi = phi_t1 - phi_t0_s;
                const double delta = cmax - cmin;
                const double cbar_hat = Tphi - cmax - comm_phi_s;

                Tphi_sum[0] += Tphi;
                comm_sum[0] += comm_phi_s;
                compmax_sum[0] += cmax;
                compmin_sum[0] += cmin;
                compmean_sum[0] += cmean;
                deltaimb_sum[0] += delta;
                cbarhat_sum[0] += cbar_hat;

                if (metrics_on && metrics_detail) {
                    mlog << "{"
                         << "\"type\":\"phase\",\"rank\":" << world_rank
                         << ",\"step\":" << step << ",\"phi\":1"
                         << ",\"Tphi\":" << Tphi
                         << ",\"comm\":" << comm_phi_s
                         << ",\"comp_max\":" << cmax
                         << ",\"comp_min\":" << cmin
                         << ",\"comp_mean\":" << cmean
                         << ",\"delta_imb\":" << delta
                         << ",\"Cbar_hat\":" << cbar_hat
                         << "}\n";
                }
            }

            // ----------------- FASE 2 -----------------
            #pragma omp single
            { phi_t0_s = omp_get_wtime(); }

            c0 = omp_get_wtime();
            #pragma omp for collapse(2) schedule(static) nowait
            for (int ii=1; ii<=ni; ii+=TILE){
                for (int jj=1; jj<=nj; jj+=TILE){
                    const int i_end = std::min(ni, ii+TILE-1);
                    const int j_end = std::min(nj, jj+TILE-1);
                    for (int i=ii; i<=i_end; ++i){
                        const int ig = ig0 + (i-1);
                        for (int j=jj; j<=j_end; ++j){
                            const int jg = jg0 + (j-1);
                            if ( ((ig + jg + m) & 1) == 1 ){
                                U_old[idx2(i,j,L)] = ( U_new[idx2(i,j,L)]
                                  + lam*( U_old[idx2(i+1,j,  L)] + U_old[idx2(i-1,j,  L)]
                                        + U_old[idx2(i,  j+1,L)] + U_old[idx2(i,  j-1,L)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            if (delay_us > 0 && delay_phi == 2 && tid == delay_tid) spin_delay_us(delay_us);
            c1 = omp_get_wtime();
            comp[1*max_threads + tid] = c1 - c0;

            #pragma omp barrier
            #pragma omp master
            {
                const double cs = MPI_Wtime();
                exchange_halo(U_old, ni, nj, nbr_north, nbr_south, nbr_west, nbr_east, COL_T, cart);
                const double ce = MPI_Wtime();
                comm_phi_s = ce - cs;
            }
            #pragma omp barrier

            #pragma omp single
            {
                const double phi_t1 = omp_get_wtime();
                const int nt = omp_get_num_threads();

                double cmax = -1e300, cmin = 1e300, csum=0.0;
                for (int t=0; t<nt; ++t) {
                    const double ct = comp[1*max_threads + t];
                    cmax = std::max(cmax, ct);
                    cmin = std::min(cmin, ct);
                    csum += ct;
                }
                const double cmean = csum / nt;
                const double Tphi = phi_t1 - phi_t0_s;
                const double delta = cmax - cmin;
                const double cbar_hat = Tphi - cmax - comm_phi_s;

                Tphi_sum[1] += Tphi;
                comm_sum[1] += comm_phi_s;
                compmax_sum[1] += cmax;
                compmin_sum[1] += cmin;
                compmean_sum[1] += cmean;
                deltaimb_sum[1] += delta;
                cbarhat_sum[1] += cbar_hat;

                if (metrics_on && metrics_detail) {
                    mlog << "{"
                         << "\"type\":\"phase\",\"rank\":" << world_rank
                         << ",\"step\":" << step << ",\"phi\":2"
                         << ",\"Tphi\":" << Tphi
                         << ",\"comm\":" << comm_phi_s
                         << ",\"comp_max\":" << cmax
                         << ",\"comp_min\":" << cmin
                         << ",\"comp_mean\":" << cmean
                         << ",\"delta_imb\":" << delta
                         << ",\"Cbar_hat\":" << cbar_hat
                         << "}\n";
                }
            }

            // ----------------- FASE 3 -----------------
            #pragma omp single
            { phi_t0_s = omp_get_wtime(); }

            c0 = omp_get_wtime();
            #pragma omp for collapse(2) schedule(static) nowait
            for (int ii=1; ii<=ni; ii+=TILE){
                for (int jj=1; jj<=nj; jj+=TILE){
                    const int i_end = std::min(ni, ii+TILE-1);
                    const int j_end = std::min(nj, jj+TILE-1);
                    for (int i=ii; i<=i_end; ++i){
                        const int ig = ig0 + (i-1);
                        for (int j=jj; j<=j_end; ++j){
                            const int jg = jg0 + (j-1);
                            if ( ((ig + jg + m) & 1) == 0 ){
                                U_new[idx2(i,j,L)] = U_old[idx2(i,j,L)] +
                                  lam*( U_old[idx2(i+1,j,  L)] + U_old[idx2(i-1,j,  L)]
                                      + U_old[idx2(i,  j+1,L)] + U_old[idx2(i,  j-1,L)]
                                      - 4.0*U_old[idx2(i,j,L)] );
                            } else {
                                U_new[idx2(i,j,L)] = U_old[idx2(i,j,L)];
                            }
                        }
                    }
                }
            }
            if (delay_us > 0 && delay_phi == 3 && tid == delay_tid) spin_delay_us(delay_us);
            c1 = omp_get_wtime();
            comp[2*max_threads + tid] = c1 - c0;

            #pragma omp barrier
            #pragma omp master
            {
                const double cs = MPI_Wtime();
                exchange_halo(U_new, ni, nj, nbr_north, nbr_south, nbr_west, nbr_east, COL_T, cart);
                const double ce = MPI_Wtime();
                comm_phi_s = ce - cs;
            }
            #pragma omp barrier

            #pragma omp single
            {
                const double phi_t1 = omp_get_wtime();
                const int nt = omp_get_num_threads();

                double cmax = -1e300, cmin = 1e300, csum=0.0;
                for (int t=0; t<nt; ++t) {
                    const double ct = comp[2*max_threads + t];
                    cmax = std::max(cmax, ct);
                    cmin = std::min(cmin, ct);
                    csum += ct;
                }
                const double cmean = csum / nt;
                const double Tphi = phi_t1 - phi_t0_s;
                const double delta = cmax - cmin;
                const double cbar_hat = Tphi - cmax - comm_phi_s;

                Tphi_sum[2] += Tphi;
                comm_sum[2] += comm_phi_s;
                compmax_sum[2] += cmax;
                compmin_sum[2] += cmin;
                compmean_sum[2] += cmean;
                deltaimb_sum[2] += delta;
                cbarhat_sum[2] += cbar_hat;

                if (metrics_on && metrics_detail) {
                    mlog << "{"
                         << "\"type\":\"phase\",\"rank\":" << world_rank
                         << ",\"step\":" << step << ",\"phi\":3"
                         << ",\"Tphi\":" << Tphi
                         << ",\"comm\":" << comm_phi_s
                         << ",\"comp_max\":" << cmax
                         << ",\"comp_min\":" << cmin
                         << ",\"comp_mean\":" << cmean
                         << ",\"delta_imb\":" << delta
                         << ",\"Cbar_hat\":" << cbar_hat
                         << "}\n";
                }
            }

            // ----------------- FASE 4 -----------------
            #pragma omp single
            { phi_t0_s = omp_get_wtime(); }

            c0 = omp_get_wtime();
            #pragma omp for collapse(2) schedule(static) nowait
            for (int ii=1; ii<=ni; ii+=TILE){
                for (int jj=1; jj<=nj; jj+=TILE){
                    const int i_end = std::min(ni, ii+TILE-1);
                    const int j_end = std::min(nj, jj+TILE-1);
                    for (int i=ii; i<=i_end; ++i){
                        const int ig = ig0 + (i-1);
                        for (int j=jj; j<=j_end; ++j){
                            const int jg = jg0 + (j-1);
                            if ( ((ig + jg + m) & 1) == 1 ){
                                U_new[idx2(i,j,L)] = ( U_old[idx2(i,j,L)]
                                  + lam*( U_new[idx2(i+1,j,  L)] + U_new[idx2(i-1,j,  L)]
                                        + U_new[idx2(i,  j+1,L)] + U_new[idx2(i,  j-1,L)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            if (delay_us > 0 && delay_phi == 4 && tid == delay_tid) spin_delay_us(delay_us);
            c1 = omp_get_wtime();
            comp[3*max_threads + tid] = c1 - c0;

            #pragma omp barrier
            #pragma omp master
            {
                const double cs = MPI_Wtime();
                exchange_halo(U_new, ni, nj, nbr_north, nbr_south, nbr_west, nbr_east, COL_T, cart);
                const double ce = MPI_Wtime();
                comm_phi_s = ce - cs;
            }
            #pragma omp barrier

            #pragma omp single
            {
                const double phi_t1 = omp_get_wtime();
                const int nt = omp_get_num_threads();

                double cmax = -1e300, cmin = 1e300, csum=0.0;
                for (int t=0; t<nt; ++t) {
                    const double ct = comp[3*max_threads + t];
                    cmax = std::max(cmax, ct);
                    cmin = std::min(cmin, ct);
                    csum += ct;
                }
                const double cmean = csum / nt;
                const double Tphi = phi_t1 - phi_t0_s;
                const double delta = cmax - cmin;
                const double cbar_hat = Tphi - cmax - comm_phi_s;

                Tphi_sum[3] += Tphi;
                comm_sum[3] += comm_phi_s;
                compmax_sum[3] += cmax;
                compmin_sum[3] += cmin;
                compmean_sum[3] += cmean;
                deltaimb_sum[3] += delta;
                cbarhat_sum[3] += cbar_hat;

                if (metrics_on && metrics_detail) {
                    mlog << "{"
                         << "\"type\":\"phase\",\"rank\":" << world_rank
                         << ",\"step\":" << step << ",\"phi\":4"
                         << ",\"Tphi\":" << Tphi
                         << ",\"comm\":" << comm_phi_s
                         << ",\"comp_max\":" << cmax
                         << ",\"comp_min\":" << cmin
                         << ",\"comp_mean\":" << cmean
                         << ",\"delta_imb\":" << delta
                         << ",\"Cbar_hat\":" << cbar_hat
                         << "}\n";
                }
            }

            // Alterna paridade (mesma lógica do original)
            #pragma omp single
            { m++; }
            #pragma omp single
            { m++; }

            // Tempo do passo
            #pragma omp single
            {
                const double step_t1 = omp_get_wtime();
                const double Tstep = step_t1 - step_t0_s;
                Tstep_sum += Tstep;

                if (metrics_on && metrics_detail) {
                    mlog << "{"
                         << "\"type\":\"step\",\"rank\":" << world_rank
                         << ",\"step\":" << step
                         << ",\"Tstep\":" << Tstep
                         << "}\n";
                }
            }
        } // step
    } // parallel
    


    const double t1 = MPI_Wtime();
    const double elapsed_local = t1 - t0;

    // mantém stdout igual ao original (max rank)
    double elapsed_max = 0.0;
    MPI_Reduce(&elapsed_local, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    MPI_Bcast(&elapsed_max, 1, MPI_DOUBLE, 0, cart);

    if (world_rank==0){
        std::cout << "Tempo: " << std::setprecision(6) << std::fixed << elapsed_max << " s\n";
    }

    // ===== summary por rank (médias por passo) =====
    if (metrics_on) {
        const double invT = (T > 0) ? 1.0 / static_cast<double>(T) : 0.0;

        double Tphi_mean[4], comm_mean[4], compmax_mean[4], compmin_mean[4], compmean_mean[4], deltaimb_mean[4], cbarhat_mean[4];
        for (int k=0; k<4; ++k) {
            Tphi_mean[k] = Tphi_sum[k] * invT;
            comm_mean[k] = comm_sum[k] * invT;
            compmax_mean[k] = compmax_sum[k] * invT;
            compmin_mean[k] = compmin_sum[k] * invT;
            compmean_mean[k] = compmean_sum[k] * invT;
            deltaimb_mean[k] = deltaimb_sum[k] * invT;
            cbarhat_mean[k] = cbarhat_sum[k] * invT;
        }
        const double Tstep_mean = Tstep_sum * invT;

        mlog << "{"
             << "\"type\":\"summary\","
             << "\"variant\":\"hib_naive\","
             << "\"rank\":" << world_rank << ",\"ranks\":" << world_size
             << ",\"nt\":" << actual_nt
             << ",\"elapsed_local\":" << elapsed_local
             << ",\"elapsed_max\":" << elapsed_max
             << ",\"Tstep_mean\":" << Tstep_mean
             << ",\"Tphi_mean\":"; json_write_array4(mlog, Tphi_mean);
        mlog << ",\"Tcomm_mean\":"; json_write_array4(mlog, comm_mean);
        mlog << ",\"comp_max_mean\":"; json_write_array4(mlog, compmax_mean);
        mlog << ",\"comp_min_mean\":"; json_write_array4(mlog, compmin_mean);
        mlog << ",\"comp_mean_mean\":"; json_write_array4(mlog, compmean_mean);
        mlog << ",\"delta_imb_mean\":"; json_write_array4(mlog, deltaimb_mean);
        mlog << ",\"Cbar_hat_mean\":"; json_write_array4(mlog, cbarhat_mean);
        mlog << "}\n";

        mlog.close();
    }

    // ===================== SAÍDA ÚNICA (output.txt) =====================
    // Cada rank prepara suas amostras interiores cujo índice global é múltiplo de 16.
    // O rank 0 coleta (Gatherv), ordena por (i,j) global e escreve:
    std::vector<int> ig_local, jg_local;
    std::vector<double> val_local;

    // Primeiro índice local que cai num múltiplo de 16 no global
    auto first_on_stride = [](int global_start)->int {
        int r = (16 - (global_start % 16)) % 16;
        return 1 + r; // converte de global para local (i=1..)
    };
    int i_first = first_on_stride(ig0);
    int j_first = first_on_stride(jg0);

    for (int i = i_first; i <= ni; i += 16) {
        int ig = ig0 + (i-1);
        for (int j = j_first; j <= nj; j += 16) {
            int jg = jg0 + (j-1);
            // Só interior (bordas 0/N-1 serão escritas como 0.0 pelo rank 0)
            if (ig>=1 && ig<=N-2 && jg>=1 && jg<=N-2) {
                ig_local.push_back(ig);
                jg_local.push_back(jg);
                val_local.push_back(U_new[idx2(i,j,L)]);
            }
        }
    }

    int mloc = static_cast<int>(ig_local.size());
    std::vector<int> counts, displs;
    int gtot = 0;

    if (world_rank==0) counts.resize(world_size);
    MPI_Gather(&mloc, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, cart);

    if (world_rank==0) {
        displs.resize(world_size, 0);
        for (int r=0; r<world_size; ++r) {
            displs[r] = gtot;
            gtot += counts[r];
        }
    }

    std::vector<int> ig_all, jg_all;
    std::vector<double> val_all;
    if (world_rank==0) {
        ig_all.resize(gtot);
        jg_all.resize(gtot);
        val_all.resize(gtot);
    }

    MPI_Gatherv(ig_local.data(),  mloc, MPI_INT,
                ig_all.data(), counts.data(), displs.data(), MPI_INT, 0, cart);
    MPI_Gatherv(jg_local.data(),  mloc, MPI_INT,
                jg_all.data(), counts.data(), displs.data(), MPI_INT, 0, cart);
    MPI_Gatherv(val_local.data(), mloc, MPI_DOUBLE,
                val_all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, cart);

    if (world_rank==0) {
        std::unordered_map<uint64_t,double> mapv;
        mapv.reserve(static_cast<size_t>(gtot)*2u + 32u);

        auto key = [](int ig, int jg)->uint64_t {
            return (static_cast<uint64_t>(static_cast<uint32_t>(ig))<<32)
                 |  static_cast<uint32_t>(jg);
        };
        for (int k=0; k<gtot; ++k) {
            mapv[key(ig_all[k], jg_all[k])] = val_all[k];
        }

        std::ofstream fout("output.txt");
        if (!fout) {
            std::cerr << "Erro: não foi possível abrir output.txt para escrita\n";
            MPI_Abort(cart, 2);
        }
        fout.setf(std::ios::fixed);
        fout.precision(8);

        for (int i=0; i<N; i+=16) {
            const double x = i * h;
            for (int j=0; j<N; j+=16) {
                const double y = j * h;
                double v = 0.0;
                if (i>=1 && i<=N-2 && j>=1 && j<=N-2) {
                    auto it = mapv.find(key(i,j));
                    if (it != mapv.end()) v = it->second;
                }
                fout << x << " " << y << " " << v << "\n";
            }
        }
        fout.close();
    }

    MPI_Type_free(&COL_T);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
