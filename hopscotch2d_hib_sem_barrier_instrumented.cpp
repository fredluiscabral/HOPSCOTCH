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

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>

namespace fs = std::filesystem;

static inline int getenv_int(const char* name, int defv) {
    const char* v = std::getenv(name);
    if (!v || !*v) return defv;
    try { return std::stoi(v); } catch (...) { return defv; }
}
static inline std::string getenv_str(const char* name, const std::string& defv) {
    const char* v = std::getenv(name);
    if (!v) return defv;
    return std::string(v);
}
static inline int64_t ns_from_sec(double s) {
    return static_cast<int64_t>(std::llround(s * 1.0e9));
}
static inline double sec_from_ns(int64_t ns) {
    return static_cast<double>(ns) * 1.0e-9;
}
static inline void spin_delay_us(int delay_us) {
    if (delay_us <= 0) return;
    const double t0 = omp_get_wtime();
    const double target = static_cast<double>(delay_us) * 1.0e-6;
    while ((omp_get_wtime() - t0) < target) { /* spin */ }
}
static inline void atomic_max_i64(std::atomic<int64_t>& a, int64_t v) {
    int64_t cur = a.load(std::memory_order_relaxed);
    while (v > cur && !a.compare_exchange_weak(cur, v, std::memory_order_release, std::memory_order_relaxed)) {}
}
static inline void atomic_min_i64(std::atomic<int64_t>& a, int64_t v) {
    int64_t cur = a.load(std::memory_order_relaxed);
    while (v < cur && !a.compare_exchange_weak(cur, v, std::memory_order_release, std::memory_order_relaxed)) {}
}

static inline void json_write_array4(std::ostream& os, const char* key, const double v[4]) {
    os << ",\"" << key << "\":[";
    for (int i = 0; i < 4; ++i) { if (i) os << ","; os << std::setprecision(17) << v[i]; }
    os << "]";
}

struct PhaseAgg {
    std::atomic<int> gen[4];
    std::atomic<int> count[4];
    std::atomic<int64_t> comp_sum_ns[4], comp_max_ns[4], comp_min_ns[4], tphi_max_ns[4];
    std::atomic<int64_t> comm_ns[4];

    std::atomic<int64_t> sum_Tphi_ns[4], sum_comm_ns[4], sum_compmax_ns[4], sum_compmin_ns[4],
                         sum_compmean_ns[4], sum_delta_ns[4], sum_cbar_ns[4];

    PhaseAgg() {
        for (int p = 0; p < 4; ++p) {
            gen[p].store(-1);
            count[p].store(0);
            comp_sum_ns[p].store(0);
            comp_max_ns[p].store(0);
            comp_min_ns[p].store(std::numeric_limits<int64_t>::max());
            tphi_max_ns[p].store(0);
            comm_ns[p].store(0);

            sum_Tphi_ns[p].store(0);
            sum_comm_ns[p].store(0);
            sum_compmax_ns[p].store(0);
            sum_compmin_ns[p].store(0);
            sum_compmean_ns[p].store(0);
            sum_delta_ns[p].store(0);
            sum_cbar_ns[p].store(0);
        }
    }

    void reset_if_new(int phi, int step) {
        const int p = phi - 1;
        int expected = gen[p].load(std::memory_order_relaxed);
        if (expected == step) return;
        if (gen[p].compare_exchange_strong(expected, step, std::memory_order_acq_rel)) {
            count[p].store(0, std::memory_order_release);
            comp_sum_ns[p].store(0, std::memory_order_release);
            comp_max_ns[p].store(0, std::memory_order_release);
            comp_min_ns[p].store(std::numeric_limits<int64_t>::max(), std::memory_order_release);
            tphi_max_ns[p].store(0, std::memory_order_release);
            comm_ns[p].store(0, std::memory_order_release);
        }
    }

    void set_comm_ns(int phi, int64_t cns) {
        comm_ns[phi - 1].store(cns, std::memory_order_release);
    }

    void end_phase(int phi, int step, int nt,
                   int64_t tphi_ns, int64_t comp_ns,
                   std::ofstream& mlog, int rank,
                   bool detail, const char* variant) {
        const int p = phi - 1;
        comp_sum_ns[p].fetch_add(comp_ns, std::memory_order_acq_rel);
        atomic_max_i64(comp_max_ns[p], comp_ns);
        atomic_min_i64(comp_min_ns[p], comp_ns);
        atomic_max_i64(tphi_max_ns[p], tphi_ns);

        const int prev = count[p].fetch_add(1, std::memory_order_acq_rel);
        if (prev != nt - 1) return; // não é o último thread desta fase neste passo

        const int64_t csum = comp_sum_ns[p].load(std::memory_order_acquire);
        const int64_t cmax = comp_max_ns[p].load(std::memory_order_acquire);
        const int64_t cmin = comp_min_ns[p].load(std::memory_order_acquire);
        const int64_t tphi = tphi_max_ns[p].load(std::memory_order_acquire);
        const int64_t comm = comm_ns[p].load(std::memory_order_acquire);
        const int64_t cmean = (nt > 0) ? (csum / nt) : 0;
        const int64_t delta = cmax - cmin;
        int64_t cbar = tphi - cmax - comm;
        if (cbar < 0) cbar = 0;

        sum_Tphi_ns[p].fetch_add(tphi, std::memory_order_acq_rel);
        sum_comm_ns[p].fetch_add(comm, std::memory_order_acq_rel);
        sum_compmax_ns[p].fetch_add(cmax, std::memory_order_acq_rel);
        sum_compmin_ns[p].fetch_add(cmin, std::memory_order_acq_rel);
        sum_compmean_ns[p].fetch_add(cmean, std::memory_order_acq_rel);
        sum_delta_ns[p].fetch_add(delta, std::memory_order_acq_rel);
        sum_cbar_ns[p].fetch_add(cbar, std::memory_order_acq_rel);

        if (detail) {
            const double comm_s = sec_from_ns(comm);
            const double cmax_s = sec_from_ns(cmax);
            const double cmin_s = sec_from_ns(cmin);
            const double cmean_s = sec_from_ns(cmean);
            const double tphi_s = sec_from_ns(tphi);
            const double delta_s = sec_from_ns(delta);
            const double cbar_s = sec_from_ns(cbar);

            #pragma omp critical(metrics_io)
            {
                mlog << "{\"type\":\"phase\",\"variant\":\"" << variant << "\""
                     << ",\"rank\":" << rank
                     << ",\"step\":" << step
                     << ",\"phi\":" << phi
                     << ",\"Tphi\":" << std::setprecision(17) << tphi_s
                     << ",\"comm\":" << std::setprecision(17) << comm_s
                     << ",\"comp_max\":" << std::setprecision(17) << cmax_s
                     << ",\"comp_min\":" << std::setprecision(17) << cmin_s
                     << ",\"comp_mean\":" << std::setprecision(17) << cmean_s
                     << ",\"delta_imb\":" << std::setprecision(17) << delta_s
                     << ",\"Cbar_hat\":" << std::setprecision(17) << cbar_s
                     << "}\n";
            }
        }
    }
};

struct StepAgg {
    std::atomic<int> gen;
    std::atomic<int> count;
    std::atomic<int64_t> tstep_max_ns;
    std::atomic<int64_t> sum_tstep_ns;

    StepAgg() {
        gen.store(-1);
        count.store(0);
        tstep_max_ns.store(0);
        sum_tstep_ns.store(0);
    }

    void reset_if_new(int step) {
        int expected = gen.load(std::memory_order_relaxed);
        if (expected == step) return;
        if (gen.compare_exchange_strong(expected, step, std::memory_order_acq_rel)) {
            count.store(0, std::memory_order_release);
            tstep_max_ns.store(0, std::memory_order_release);
        }
    }

    void end_step(int step, int nt, int64_t tstep_ns,
                  std::ofstream& mlog, int rank,
                  bool detail, const char* variant) {
        atomic_max_i64(tstep_max_ns, tstep_ns);
        const int prev = count.fetch_add(1, std::memory_order_acq_rel);
        if (prev != nt - 1) return;

        const int64_t tmax = tstep_max_ns.load(std::memory_order_acquire);
        sum_tstep_ns.fetch_add(tmax, std::memory_order_acq_rel);

        if (detail) {
            #pragma omp critical(metrics_io)
            {
                mlog << "{\"type\":\"step\",\"variant\":\"" << variant << "\""
                     << ",\"rank\":" << rank
                     << ",\"step\":" << step
                     << ",\"Tstep\":" << std::setprecision(17) << sec_from_ns(tmax)
                     << "}\n";
            }
        }
    }
};

static constexpr const char* VARIANT = "hib_sem_barrier";

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

    
    // ---------------- Instrumentação (JSONL) ----------------
    const bool metrics_on = (getenv_int("METRICS", 0) != 0);
    const bool metrics_detail = (getenv_int("METRICS_DETAIL", 0) != 0);
    const int delay_us = getenv_int("DELAY_US", 0);
    const int delay_tid = getenv_int("DELAY_TID", 0);
    const int delay_phi = getenv_int("DELAY_PHI", 0);

    const std::string metrics_dir = getenv_str("METRICS_PATH", "./metrics");
    std::ofstream mlog;
    if (metrics_on) {
        fs::create_directories(metrics_dir);
        std::ostringstream fn;
        fn << metrics_dir << "/" << VARIANT << ".rank" << rank << ".jsonl";
        mlog.open(fn.str(), std::ios::out | std::ios::trunc);
        if (!mlog) {
            std::cerr << "[WARN] Não foi possível abrir arquivo de métricas: " << fn.str() << std::endl;
        } else {
            const std::string omp_places = getenv_str("OMP_PLACES", "");
            const std::string omp_proc_bind = getenv_str("OMP_PROC_BIND", "");
            mlog << "{\"type\":\"run\",\"variant\":\"" << VARIANT << "\""
                 << ",\"rank\":" << rank
                 << ",\"ranks\":" << nprocs
                 << ",\"Px\":" << nprocs
                 << ",\"Qy\":1"
                 << ",\"N\":" << N
                 << ",\"alpha\":" << std::setprecision(17) << alpha
                 << ",\"T\":" << T
                 << ",\"tile\":" << TILE
                 << ",\"ni\":" << local_n
                 << ",\"nj\":" << N
                 << ",\"dt\":" << std::setprecision(17) << dt
                 << ",\"lam\":" << std::setprecision(17) << lam
                 << ",\"omp_places\":\"" << omp_places << "\""
                 << ",\"omp_proc_bind\":\"" << omp_proc_bind << "\""
                 << ",\"delay_us\":" << delay_us
                 << ",\"delay_tid\":" << delay_tid
                 << ",\"delay_phase\":" << delay_phi
                 << "}\n";
        }
    }

    PhaseAgg phaseAgg;
    StepAgg stepAgg;
    int actual_nt = 0;

// Sincroniza antes de cronometrar (tempo global)
    MPI_Barrier(COMM);
    const double t0 = MPI_Wtime();

    // Variável compartilhada de paridade global (por rank)
    int m_shared = 0;

    // ---------------- OpenMP região paralela ----------------
    std::vector<sem_t> sem_left, sem_right;

    #pragma omp parallel default(none) \
        shared(N, T, TILE, lam, denom, U_new, U_old, L, local_n, start_g, up, down, COMM, \
               sem_left, sem_right, m_shared, phaseAgg, stepAgg, mlog, metrics_on, metrics_detail, delay_us, delay_tid, delay_phi, actual_nt)
    {
        const int tid = omp_get_thread_num();
        #pragma omp single
        {
            actual_nt = omp_get_num_threads();
        }

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
            if (metrics_on) phaseAgg.reset_if_new(1, step);
            #pragma omp master
            {
                const double cs = MPI_Wtime();
                exchange_halos(U_new.data(), N, local_n, up, down, 100 + 4*(step%1000000) + 0, COMM);
                const double ce = MPI_Wtime();
                if (metrics_on) phaseAgg.set_comm_ns(1, ns_from_sec(ce - cs));
            }
            #pragma omp barrier

            int m_local = m_shared;

            // ---------------- FASE 1 (explícita): U_old a partir de U_new nos pontos pares
            {
                double phi_t0 = 0.0;
                double c0 = 0.0;
                int64_t comp_ns = 0;
                if (metrics_on) { phi_t0 = omp_get_wtime(); c0 = phi_t0; }
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
                                if (delay_us > 0 && delay_phi == 1 && tid == delay_tid) spin_delay_us(delay_us);
                if (metrics_on) { comp_ns = ns_from_sec(omp_get_wtime() - c0); }

signal_done(tid);
                wait_for_neighbors(tid);
                if (metrics_on) {
                    const int64_t tphi_ns = ns_from_sec(omp_get_wtime() - phi_t0);
                    phaseAgg.end_phase(1, step, actual_nt, tphi_ns, comp_ns, mlog, rank, metrics_detail, VARIANT);
                }

            }

            // --- Troca de halo U_old (após Fase 1)
            if (metrics_on) phaseAgg.reset_if_new(2, step);
            #pragma omp master
            {
                const double cs = MPI_Wtime();
                exchange_halos(U_old.data(), N, local_n, up, down, 100 + 4*(step%1000000) + 1, COMM);
                const double ce = MPI_Wtime();
                if (metrics_on) phaseAgg.set_comm_ns(2, ns_from_sec(ce - cs));
            }
            #pragma omp barrier

            // ---------------- FASE 2 (semi-implícita): U_old nos pontos ímpares
            {
                double phi_t0 = 0.0;
                double c0 = 0.0;
                int64_t comp_ns = 0;
                if (metrics_on) { phi_t0 = omp_get_wtime(); c0 = phi_t0; }
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
                                if (delay_us > 0 && delay_phi == 2 && tid == delay_tid) spin_delay_us(delay_us);
                if (metrics_on) { comp_ns = ns_from_sec(omp_get_wtime() - c0); }

signal_done(tid);
                wait_for_neighbors(tid);
                if (metrics_on) {
                    const int64_t tphi_ns = ns_from_sec(omp_get_wtime() - phi_t0);
                    phaseAgg.end_phase(2, step, actual_nt, tphi_ns, comp_ns, mlog, rank, metrics_detail, VARIANT);
                }

                m_local++;
            }

            // --- Troca de halo U_old (após Fase 2)
            if (metrics_on) phaseAgg.reset_if_new(3, step);
            #pragma omp master
            {
                const double cs = MPI_Wtime();
                exchange_halos(U_old.data(), N, local_n, up, down, 100 + 4*(step%1000000) + 2, COMM);
                const double ce = MPI_Wtime();
                if (metrics_on) phaseAgg.set_comm_ns(3, ns_from_sec(ce - cs));
            }
            #pragma omp barrier

            // ---------------- FASE 3 (explícita): U_new a partir de U_old nos pontos pares
            {
                double phi_t0 = 0.0;
                double c0 = 0.0;
                int64_t comp_ns = 0;
                if (metrics_on) { phi_t0 = omp_get_wtime(); c0 = phi_t0; }
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
                                if (delay_us > 0 && delay_phi == 3 && tid == delay_tid) spin_delay_us(delay_us);
                if (metrics_on) { comp_ns = ns_from_sec(omp_get_wtime() - c0); }

signal_done(tid);
                wait_for_neighbors(tid);
                if (metrics_on) {
                    const int64_t tphi_ns = ns_from_sec(omp_get_wtime() - phi_t0);
                    phaseAgg.end_phase(3, step, actual_nt, tphi_ns, comp_ns, mlog, rank, metrics_detail, VARIANT);
                }

            }

            // --- Troca de halo U_new (após Fase 3)
            if (metrics_on) phaseAgg.reset_if_new(4, step);
            #pragma omp master
            {
                const double cs = MPI_Wtime();
                exchange_halos(U_new.data(), N, local_n, up, down, 100 + 4*(step%1000000) + 3, COMM);
                const double ce = MPI_Wtime();
                if (metrics_on) phaseAgg.set_comm_ns(4, ns_from_sec(ce - cs));
            }
            #pragma omp barrier

            // ---------------- FASE 4 (semi-implícita): U_new nos pontos ímpares
            {
                double phi_t0 = 0.0;
                double c0 = 0.0;
                int64_t comp_ns = 0;
                if (metrics_on) { phi_t0 = omp_get_wtime(); c0 = phi_t0; }
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
                                if (delay_us > 0 && delay_phi == 4 && tid == delay_tid) spin_delay_us(delay_us);
                if (metrics_on) { comp_ns = ns_from_sec(omp_get_wtime() - c0); }

signal_done(tid);
                wait_for_neighbors(tid);
                if (metrics_on) {
                    const int64_t tphi_ns = ns_from_sec(omp_get_wtime() - phi_t0);
                    phaseAgg.end_phase(4, step, actual_nt, tphi_ns, comp_ns, mlog, rank, metrics_detail, VARIANT);
                }

                m_local++;
            }

            // --- Barreira global por passo: mantém índice temporal alinhado entre threads
            #pragma omp barrier
            #pragma omp single
            {
                m_shared += 2;
            }
            #pragma omp barrier
            if (metrics_on) {
                const int64_t tstep_ns = ns_from_sec(omp_get_wtime() - step_t0);
                stepAgg.end_step(step, actual_nt, tstep_ns, mlog, rank, metrics_detail, VARIANT);
            }
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
    double secs_max = 0.0;    MPI_Reduce(&secs, &secs_max, 1, MPI_DOUBLE, MPI_MAX, 0, COMM);
    MPI_Bcast(&secs_max, 1, MPI_DOUBLE, 0, COMM);

    if (metrics_on && mlog) {
        double Tphi_mean[4], Tcomm_mean[4], comp_max_mean[4], comp_min_mean[4], comp_mean_mean[4], delta_imb_mean[4], Cbar_hat_mean[4];
        for (int p = 0; p < 4; ++p) {
            Tphi_mean[p] = sec_from_ns(phaseAgg.sum_Tphi_ns[p].load(std::memory_order_acquire) / T);
            Tcomm_mean[p] = sec_from_ns(phaseAgg.sum_comm_ns[p].load(std::memory_order_acquire) / T);
            comp_max_mean[p] = sec_from_ns(phaseAgg.sum_compmax_ns[p].load(std::memory_order_acquire) / T);
            comp_min_mean[p] = sec_from_ns(phaseAgg.sum_compmin_ns[p].load(std::memory_order_acquire) / T);
            comp_mean_mean[p] = sec_from_ns(phaseAgg.sum_compmean_ns[p].load(std::memory_order_acquire) / T);
            delta_imb_mean[p] = sec_from_ns(phaseAgg.sum_delta_ns[p].load(std::memory_order_acquire) / T);
            Cbar_hat_mean[p] = sec_from_ns(phaseAgg.sum_cbar_ns[p].load(std::memory_order_acquire) / T);
        }
        const double Tstep_mean = sec_from_ns(stepAgg.sum_tstep_ns.load(std::memory_order_acquire) / T);

        mlog << "{\"type\":\"summary\",\"variant\":\"" << VARIANT << "\""
             << ",\"rank\":" << rank
             << ",\"ranks\":" << nprocs
             << ",\"nt\":" << actual_nt
             << ",\"elapsed_local\":" << std::setprecision(17) << secs
             << ",\"elapsed_max\":" << std::setprecision(17) << secs_max
             << ",\"Tstep_mean\":" << std::setprecision(17) << Tstep_mean;
        json_write_array4(mlog, "Tphi_mean", Tphi_mean);
        json_write_array4(mlog, "Tcomm_mean", Tcomm_mean);
        json_write_array4(mlog, "comp_max_mean", comp_max_mean);
        json_write_array4(mlog, "comp_min_mean", comp_min_mean);
        json_write_array4(mlog, "comp_mean_mean", comp_mean_mean);
        json_write_array4(mlog, "delta_imb_mean", delta_imb_mean);
        json_write_array4(mlog, "Cbar_hat_mean", Cbar_hat_mean);
        mlog << "}
";
        mlog.flush();
    }
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
