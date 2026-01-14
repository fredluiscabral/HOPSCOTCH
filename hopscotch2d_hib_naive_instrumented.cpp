// hopscotch2d_hib_naive_instrumented.cpp
// Equação do calor 2D (Hopscotch) — versão HÍBRIDA MPI+OpenMP com block tiling.
// + Instrumentação de métricas para identificação de parâmetros do modelo:
//   - Por fase (φ=1..4) e por passo (step)
//   - Por rank (MPI) e usando todos os threads (OpenMP)
//   - Log JSONL: metrics_hib_naive_rank<rank>.jsonl
//
// Compilar:
//   mpicxx -O3 -march=native -fopenmp -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1 \
//          hopscotch2d_hib_naive_instrumented.cpp -o hopscotch2d_hib_naive_instrumented
//
// Executar (exemplo OpenMPI):
//   mpirun -np 2 --map-by ppr:1:socket:PE=48 --bind-to core \
//          -x OMP_NUM_THREADS=48 -x OMP_PLACES=cores -x OMP_PROC_BIND=spread \
//          ./hopscotch2d_hib_naive_instrumented

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
#include <limits>

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

// índice linear local (com halo) — L = (nj+2)
static inline size_t idx2(int i, int j, int L) {
    return static_cast<size_t>(i) * L + j;
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
// (instrumentação de tempo será feita fora, por quem chama)
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
    // Bordas externas permanecem 0.0 (Dirichlet)

    // Datatype de coluna (halo vertical)
    MPI_Datatype COL_T;
    MPI_Type_vector(ni, /*blocklen=*/1, /*stride=*/L, MPI_DOUBLE, &COL_T);
    MPI_Type_commit(&COL_T);

    // Halo inicial de U_new (para o passo 0)
    exchange_halo(U_new, ni, nj, nbr_north, nbr_south, nbr_west, nbr_east, COL_T, cart);

    // Arquivo de métricas (um por rank)
    std::ostringstream mname;
    mname << "metrics_hib_naive_rank" << world_rank << ".jsonl";
    std::ofstream mlog(mname.str());
    if (!mlog) {
        std::cerr << "Erro: não foi possível abrir " << mname.str() << " para escrita\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    mlog.setf(std::ios::fixed);
    mlog << std::setprecision(6);

    // Cabeçalho de execução
    mlog << "{"
         << "\"type\":\"run\",\"variant\":\"hib_naive\",\"rank\":" << world_rank
         << ",\"ranks\":" << world_size << ",\"Px\":" << Px << ",\"Qy\":" << Qy
         << ",\"N\":" << N << ",\"alpha\":" << alpha << ",\"T\":" << T
         << ",\"TILE\":" << TILE << "}\n";

    // Integração temporal + instrumentação
    int m = 0; // paridade global
    MPI_Barrier(cart);
    const double t0 = MPI_Wtime();

    // Alocação para coletar tempos de computação por thread e fase
    const int max_threads = omp_get_max_threads();
    std::vector<std::vector<double>> comp_by_thread(4, std::vector<double>(max_threads, 0.0));

    #pragma omp parallel default(none) \
        shared(N,T,TILE,lam,denom,U_new,U_old,m,ni,nj,ig0,jg0,cart, \
               nbr_north,nbr_south,nbr_west,nbr_east,COL_T,L,mlog,comp_by_thread, \
               world_rank,world_size,Px,Qy,alpha)
    {
        const int tid = omp_get_thread_num();

        for (int step=0; step<T; ++step){
            // --- Acumuladores por passo (só usados em single)
            double step_halo_sum = 0.0;
            double step_cbar_sum = 0.0;
            double step_compmax_sum = 0.0;
            double step_start = 0.0;

            #pragma omp single
            {
                step_start = MPI_Wtime();
                // zera buffers por fase
                for (int ph=0; ph<4; ++ph)
                    std::fill(comp_by_thread[ph].begin(), comp_by_thread[ph].end(), 0.0);
            }

            // =========================================================
            // ===== Fase 1 (explícita): U_old <- U_new em (i+j+m) par
            double phi1_start = MPI_Wtime();
            double tcomp0 = omp_get_wtime();

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
            double tcomp1 = omp_get_wtime();
            comp_by_thread[0][tid] = tcomp1 - tcomp0;

            // Garante término do for antes do halo
            #pragma omp barrier

            double halo1 = 0.0;
            #pragma omp master
            {
                double hs = MPI_Wtime();
                exchange_halo(U_old, ni, nj, nbr_north, nbr_south, nbr_west, nbr_east, COL_T, cart);
                double he = MPI_Wtime();
                halo1 = he - hs;
            }
            #pragma omp barrier
            double phi1_end = MPI_Wtime();

            #pragma omp single
            {
                const int nt = omp_get_num_threads();
                double cmax=-1e300, cmin=1e300, csum=0.0;
                for (int t=0; t<nt; ++t){
                    cmax = std::max(cmax, comp_by_thread[0][t]);
                    cmin = std::min(cmin, comp_by_thread[0][t]);
                    csum += comp_by_thread[0][t];
                }
                double cmean = csum / nt;
                double phase_wall = phi1_end - phi1_start;
                double cbar_hat = phase_wall - cmax - halo1;
                step_halo_sum += halo1;
                step_cbar_sum += cbar_hat;
                step_compmax_sum += cmax;

                mlog << "{"
                     << "\"type\":\"phase\",\"variant\":\"hib_naive\",\"rank\":" << world_rank
                     << ",\"ranks\":" << world_size << ",\"nt\":" << nt
                     << ",\"Px\":" << Px << ",\"Qy\":" << Qy
                     << ",\"N\":" << N << ",\"alpha\":" << alpha << ",\"TILE\":" << TILE
                     << ",\"step\":" << step << ",\"phi\":1"
                     << ",\"t_wall\":" << phase_wall
                     << ",\"comp_max\":" << cmax << ",\"comp_min\":" << cmin
                     << ",\"comp_mean\":" << cmean
                     << ",\"halo\":" << halo1
                     << ",\"cbar_hat\":" << cbar_hat
                     << "}\n";
            }

            // =========================================================
            // ===== Fase 2 (semi-implícita): U_old em (i+j+m) ímpar
            double phi2_start = MPI_Wtime();
            tcomp0 = omp_get_wtime();

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
            tcomp1 = omp_get_wtime();
            comp_by_thread[1][tid] = tcomp1 - tcomp0;

            #pragma omp barrier
            double halo2 = 0.0;
            #pragma omp master
            {
                double hs = MPI_Wtime();
                exchange_halo(U_old, ni, nj, nbr_north, nbr_south, nbr_west, nbr_east, COL_T, cart);
                double he = MPI_Wtime();
                halo2 = he - hs;
            }
            #pragma omp barrier
            double phi2_end = MPI_Wtime();

            #pragma omp single
            {
                const int nt = omp_get_num_threads();
                double cmax=-1e300, cmin=1e300, csum=0.0;
                for (int t=0; t<nt; ++t){
                    cmax = std::max(cmax, comp_by_thread[1][t]);
                    cmin = std::min(cmin, comp_by_thread[1][t]);
                    csum += comp_by_thread[1][t];
                }
                double cmean = csum / nt;
                double phase_wall = phi2_end - phi2_start;
                double cbar_hat = phase_wall - cmax - halo2;
                step_halo_sum += halo2;
                step_cbar_sum += cbar_hat;
                step_compmax_sum += cmax;

                mlog << "{"
                     << "\"type\":\"phase\",\"variant\":\"hib_naive\",\"rank\":" << world_rank
                     << ",\"ranks\":" << world_size << ",\"nt\":" << nt
                     << ",\"Px\":" << Px << ",\"Qy\":" << Qy
                     << ",\"N\":" << N << ",\"alpha\":" << alpha << ",\"TILE\":" << TILE
                     << ",\"step\":" << step << ",\"phi\":2"
                     << ",\"t_wall\":" << phase_wall
                     << ",\"comp_max\":" << cmax << ",\"comp_min\":" << cmin
                     << ",\"comp_mean\":" << cmean
                     << ",\"halo\":" << halo2
                     << ",\"cbar_hat\":" << cbar_hat
                     << "}\n";
            }

            // =========================================================
            // ===== Fase 3 (explícita): U_new <- U_old em (i+j+m) par
            double phi3_start = MPI_Wtime();
            tcomp0 = omp_get_wtime();

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
            tcomp1 = omp_get_wtime();
            comp_by_thread[2][tid] = tcomp1 - tcomp0;

            #pragma omp barrier
            double halo3 = 0.0;
            #pragma omp master
            {
                double hs = MPI_Wtime();
                exchange_halo(U_new, ni, nj, nbr_north, nbr_south, nbr_west, nbr_east, COL_T, cart);
                double he = MPI_Wtime();
                halo3 = he - hs;
            }
            #pragma omp barrier
            double phi3_end = MPI_Wtime();

            #pragma omp single
            {
                const int nt = omp_get_num_threads();
                double cmax=-1e300, cmin=1e300, csum=0.0;
                for (int t=0; t<nt; ++t){
                    cmax = std::max(cmax, comp_by_thread[2][t]);
                    cmin = std::min(cmin, comp_by_thread[2][t]);
                    csum += comp_by_thread[2][t];
                }
                double cmean = csum / nt;
                double phase_wall = phi3_end - phi3_start;
                double cbar_hat = phase_wall - cmax - halo3;
                step_halo_sum += halo3;
                step_cbar_sum += cbar_hat;
                step_compmax_sum += cmax;

                mlog << "{"
                     << "\"type\":\"phase\",\"variant\":\"hib_naive\",\"rank\":" << world_rank
                     << ",\"ranks\":" << world_size << ",\"nt\":" << nt
                     << ",\"Px\":" << Px << ",\"Qy\":" << Qy
                     << ",\"N\":" << N << ",\"alpha\":" << alpha << ",\"TILE\":" << TILE
                     << ",\"step\":" << step << ",\"phi\":3"
                     << ",\"t_wall\":" << phase_wall
                     << ",\"comp_max\":" << cmax << ",\"comp_min\":" << cmin
                     << ",\"comp_mean\":" << cmean
                     << ",\"halo\":" << halo3
                     << ",\"cbar_hat\":" << cbar_hat
                     << "}\n";
            }

            // =========================================================
            // ===== Fase 4 (semi-implícita): U_new em (i+j+m) ímpar
            double phi4_start = MPI_Wtime();
            tcomp0 = omp_get_wtime();

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
            tcomp1 = omp_get_wtime();
            comp_by_thread[3][tid] = tcomp1 - tcomp0;

            #pragma omp barrier
            double halo4 = 0.0;
            #pragma omp master
            {
                double hs = MPI_Wtime();
                exchange_halo(U_new, ni, nj, nbr_north, nbr_south, nbr_west, nbr_east, COL_T, cart);
                double he = MPI_Wtime();
                halo4 = he - hs;
            }
            #pragma omp barrier
            double phi4_end = MPI_Wtime();

            #pragma omp single
            {
                const int nt = omp_get_num_threads();
                double cmax=-1e300, cmin=1e300, csum=0.0;
                for (int t=0; t<nt; ++t){
                    cmax = std::max(cmax, comp_by_thread[3][t]);
                    cmin = std::min(cmin, comp_by_thread[3][t]);
                    csum += comp_by_thread[3][t];
                }
                double cmean = csum / nt;
                double phase_wall = phi4_end - phi4_start;
                double cbar_hat = phase_wall - cmax - halo4;
                step_halo_sum += halo4;
                step_cbar_sum += cbar_hat;
                step_compmax_sum += cmax;

                mlog << "{"
                     << "\"type\":\"phase\",\"variant\":\"hib_naive\",\"rank\":" << world_rank
                     << ",\"ranks\":" << world_size << ",\"nt\":" << nt
                     << ",\"Px\":" << Px << ",\"Qy\":" << Qy
                     << ",\"N\":" << N << ",\"alpha\":" << alpha << ",\"TILE\":" << TILE
                     << ",\"step\":" << step << ",\"phi\":4"
                     << ",\"t_wall\":" << phase_wall
                     << ",\"comp_max\":" << cmax << ",\"comp_min\":" << cmin
                     << ",\"comp_mean\":" << cmean
                     << ",\"halo\":" << halo4
                     << ",\"cbar_hat\":" << cbar_hat
                     << "}\n";
            }

            // ===== Alterna paridade (compatível com o código original)
            #pragma omp single
            { m++; }
            #pragma omp single
            { m++; } // completa o passo (duas alternâncias por passo)

            // ===== Log do passo
            #pragma omp single
            {
                double step_end = MPI_Wtime();
                double t_step = step_end - step_start;
                mlog << "{"
                     << "\"type\":\"step\",\"variant\":\"hib_naive\",\"rank\":" << world_rank
                     << ",\"step\":" << step
                     << ",\"t_wall\":" << t_step
                     << ",\"halo_sum\":" << step_halo_sum
                     << ",\"cbar_hat_sum\":" << step_cbar_sum
                     << ",\"comp_max_sum\":" << step_compmax_sum
                     << "}\n";
            }
        } // end for step
    } // end parallel

    const double t1 = MPI_Wtime();
    double elapsed = t1 - t0;
    double elapsed_max = 0.0;
    MPI_Reduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (world_rank==0){
        std::cout << "Tempo total (max rank): " << std::setprecision(6) << std::fixed
                  << elapsed_max << " s\n";
        std::cout << "Métricas por rank em: metrics_hib_naive_rank<R>.jsonl\n";
    }

    // ===================== SAÍDA ÚNICA (output.txt) =====================
    // (idêntica ao original)
    std::vector<int> ig_local, jg_local;
    std::vector<double> val_local;

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
        mapv.reserve(static_cast<size_t>(gtot)*1.3);
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
            MPI_Abort(MPI_COMM_WORLD, 2);
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
    // ====================================================================

    MPI_Type_free(&COL_T);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
