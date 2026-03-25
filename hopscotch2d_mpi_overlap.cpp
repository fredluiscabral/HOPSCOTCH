// hopscotch2d_mpi_naive_overlap_nb_1d.cpp
// Equação do calor 2D (Hopscotch) — MPI puro, decomposição 1D por linhas,
// comunicação de halo não bloqueante e cálculo interior/bordas.
// - Sem OpenMP
// - Cada rank recebe uma faixa contígua das linhas interiores [1..N-2]
// - Todas as colunas pertencem ao rank (j = 0..N-1), com fronteiras Dirichlet em j=0 e j=N-1
// - Halo norte/sul via MPI_Irecv/MPI_Isend
// - Calcula interior primeiro, espera a comunicação e depois calcula bordas
//
// Compilar:
//   mpicxx -O3 -std=c++17 -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1 \
//          hopscotch2d_mpi_overlap.cpp -o hopscotch2d_mpi_overlap

#include <mpi.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

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

static inline size_t round_up(size_t x, size_t m) {
    return ((x + m - 1) / m) * m;
}

template <typename T>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    ~AlignedBuffer() { reset(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    void allocate(size_t n, size_t alignment) {
        reset();
        if (n == 0) return;
        void* p = nullptr;
#if defined(_MSC_VER)
        p = _aligned_malloc(n * sizeof(T), alignment);
        if (!p) throw std::bad_alloc();
#else
        if (posix_memalign(&p, alignment, n * sizeof(T)) != 0) throw std::bad_alloc();
#endif
        ptr_ = static_cast<T*>(p);
        size_ = n;
    }

    void reset() {
        if (ptr_) {
#if defined(_MSC_VER)
            _aligned_free(ptr_);
#else
            free(ptr_);
#endif
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return size_; }

    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

static inline size_t idx2(int i, int j, size_t ld) {
    return static_cast<size_t>(i) * ld + static_cast<size_t>(j);
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

static inline void split_range(int total, int parts, int coord, int& start, int& count){
    const int base = total / parts;
    const int rem  = total % parts;
    count = base + (coord < rem ? 1 : 0);
    start = coord * base + std::min(coord, rem);
}

static inline int exchange_halo_rows_start(double* A, int ni, size_t ld,
                                           int nbr_north, int nbr_south,
                                           MPI_Request reqs[4], MPI_Comm comm)
{
    int nreq = 0;
    if (nbr_north != MPI_PROC_NULL) {
        MPI_Irecv(&A[idx2(0,    0, ld)], static_cast<int>(ld), MPI_DOUBLE, nbr_north, 11, comm, &reqs[nreq++]);
    }
    if (nbr_south != MPI_PROC_NULL) {
        MPI_Irecv(&A[idx2(ni+1, 0, ld)], static_cast<int>(ld), MPI_DOUBLE, nbr_south, 10, comm, &reqs[nreq++]);
    }
    if (nbr_north != MPI_PROC_NULL) {
        MPI_Isend(&A[idx2(1,    0, ld)], static_cast<int>(ld), MPI_DOUBLE, nbr_north, 10, comm, &reqs[nreq++]);
    }
    if (nbr_south != MPI_PROC_NULL) {
        MPI_Isend(&A[idx2(ni,   0, ld)], static_cast<int>(ld), MPI_DOUBLE, nbr_south, 11, comm, &reqs[nreq++]);
    }
    return nreq;
}

static inline void wait_exchange(int nreq, MPI_Request reqs[4]) {
    if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
}

// Kernels por faixa de linhas [ibeg, iend] e colunas [1..N-2]
static inline void phase1_explicit(double* U_old, const double* U_new,
                                   int ibeg, int iend, int ig0, int N, size_t ld,
                                   int m, double lam, int TILE)
{
    if (ibeg > iend) return;
    for (int ii = ibeg; ii <= iend; ii += TILE) {
        const int i_end = std::min(iend, ii + TILE - 1);
        for (int jj = 1; jj <= N - 2; jj += TILE) {
            const int j_end = std::min(N - 2, jj + TILE - 1);
            for (int i = ii; i <= i_end; ++i) {
                const int ig = ig0 + (i - 1);
                for (int j = jj; j <= j_end; ++j) {
                    if (((ig + j + m) & 1) == 0) {
                        U_old[idx2(i, j, ld)] = U_new[idx2(i, j, ld)] +
                            lam * (U_new[idx2(i+1, j,   ld)] + U_new[idx2(i-1, j,   ld)] +
                                   U_new[idx2(i,   j+1, ld)] + U_new[idx2(i,   j-1, ld)] -
                                   4.0 * U_new[idx2(i, j, ld)]);
                    } else {
                        U_old[idx2(i, j, ld)] = U_new[idx2(i, j, ld)];
                    }
                }
            }
        }
    }
}

static inline void phase2_semi(double* U_old, const double* U_new,
                               int ibeg, int iend, int ig0, int N, size_t ld,
                               int m, double lam, double denom, int TILE)
{
    if (ibeg > iend) return;
    for (int ii = ibeg; ii <= iend; ii += TILE) {
        const int i_end = std::min(iend, ii + TILE - 1);
        for (int jj = 1; jj <= N - 2; jj += TILE) {
            const int j_end = std::min(N - 2, jj + TILE - 1);
            for (int i = ii; i <= i_end; ++i) {
                const int ig = ig0 + (i - 1);
                for (int j = jj; j <= j_end; ++j) {
                    if (((ig + j + m) & 1) == 1) {
                        U_old[idx2(i, j, ld)] =
                            (U_new[idx2(i, j, ld)] +
                             lam * (U_old[idx2(i+1, j,   ld)] + U_old[idx2(i-1, j,   ld)] +
                                    U_old[idx2(i,   j+1, ld)] + U_old[idx2(i,   j-1, ld)])) / denom;
                    }
                }
            }
        }
    }
}

static inline void phase3_explicit(double* U_new, const double* U_old,
                                   int ibeg, int iend, int ig0, int N, size_t ld,
                                   int m, double lam, int TILE)
{
    if (ibeg > iend) return;
    for (int ii = ibeg; ii <= iend; ii += TILE) {
        const int i_end = std::min(iend, ii + TILE - 1);
        for (int jj = 1; jj <= N - 2; jj += TILE) {
            const int j_end = std::min(N - 2, jj + TILE - 1);
            for (int i = ii; i <= i_end; ++i) {
                const int ig = ig0 + (i - 1);
                for (int j = jj; j <= j_end; ++j) {
                    if (((ig + j + m) & 1) == 0) {
                        U_new[idx2(i, j, ld)] = U_old[idx2(i, j, ld)] +
                            lam * (U_old[idx2(i+1, j,   ld)] + U_old[idx2(i-1, j,   ld)] +
                                   U_old[idx2(i,   j+1, ld)] + U_old[idx2(i,   j-1, ld)] -
                                   4.0 * U_old[idx2(i, j, ld)]);
                    } else {
                        U_new[idx2(i, j, ld)] = U_old[idx2(i, j, ld)];
                    }
                }
            }
        }
    }
}

static inline void phase4_semi(double* U_new, const double* U_old,
                               int ibeg, int iend, int ig0, int N, size_t ld,
                               int m, double lam, double denom, int TILE)
{
    if (ibeg > iend) return;
    for (int ii = ibeg; ii <= iend; ii += TILE) {
        const int i_end = std::min(iend, ii + TILE - 1);
        for (int jj = 1; jj <= N - 2; jj += TILE) {
            const int j_end = std::min(N - 2, jj + TILE - 1);
            for (int i = ii; i <= i_end; ++i) {
                const int ig = ig0 + (i - 1);
                for (int j = jj; j <= j_end; ++j) {
                    if (((ig + j + m) & 1) == 1) {
                        U_new[idx2(i, j, ld)] =
                            (U_old[idx2(i, j, ld)] +
                             lam * (U_new[idx2(i+1, j,   ld)] + U_new[idx2(i-1, j,   ld)] +
                                    U_new[idx2(i,   j+1, ld)] + U_new[idx2(i,   j-1, ld)])) / denom;
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0, T = 0, TILE = 0;
    double alpha = 0.0;
    if (rank == 0) {
        if (!load_params_strict_rank0("param.txt", N, alpha, T, TILE)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (rank == 0) {
        std::printf("MPI processes: %d\n", size);
        std::fflush(stdout);
    }

    MPI_Bcast(&N, 1, MPI_INT,       0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE,0, MPI_COMM_WORLD);
    MPI_Bcast(&T, 1, MPI_INT,       0, MPI_COMM_WORLD);
    MPI_Bcast(&TILE, 1, MPI_INT,    0, MPI_COMM_WORLD);

    const double h   = 1.0 / (N - 1);
    const double dt  = 0.90 * (h*h) / (4.0 * alpha);
    const double lam = alpha * dt / (h*h);
    const double denom = 1.0 + 4.0 * lam;

    const int interior_rows = N - 2;
    int istart_local = 0, icount_local = 0;
    split_range(interior_rows, size, rank, istart_local, icount_local);

    const int ni = icount_local;
    const int ig0 = 1 + istart_local;

    constexpr size_t alignment = 64;
    constexpr size_t doubles_per_cacheline = 8;
    const size_t ld = round_up(static_cast<size_t>(N), doubles_per_cacheline);
    const size_t NN_local = static_cast<size_t>(ni + 2) * ld;

    AlignedBuffer<double> U_new, U_old;
    U_new.allocate(NN_local, alignment);
    U_old.allocate(NN_local, alignment);
    std::fill_n(U_new.data(), NN_local, 0.0);
    std::fill_n(U_old.data(), NN_local, 0.0);

    const int nbr_north = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    const int nbr_south = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;

    const double D = 100.0, x0 = 0.5, y0 = 0.5;
    for (int i = 1; i <= ni; ++i) {
        const int ig = ig0 + (i - 1);
        const double x = ig * h;
        for (int j = 1; j <= N - 2; ++j) {
            const double y = j * h;
            U_new[idx2(i, j, ld)] = std::exp(-D * ((x-x0)*(x-x0) + (y-y0)*(y-y0)));
        }
    }

    {
        MPI_Request reqs[4];
        const int nreq = exchange_halo_rows_start(U_new.data(), ni, ld, nbr_north, nbr_south, reqs, MPI_COMM_WORLD);
        wait_exchange(nreq, reqs);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    int m = 0;

    for (int step = 0; step < T; ++step) {
        // Fase 1: U_new -> U_old
        {
            MPI_Request reqs[4];
            const int nreq = exchange_halo_rows_start(U_new.data(), ni, ld, nbr_north, nbr_south, reqs, MPI_COMM_WORLD);
            phase1_explicit(U_old.data(), U_new.data(), 2, ni - 1, ig0, N, ld, m, lam, TILE);
            wait_exchange(nreq, reqs);
            if (ni >= 1) phase1_explicit(U_old.data(), U_new.data(), 1, 1, ig0, N, ld, m, lam, TILE);
            if (ni >= 2) phase1_explicit(U_old.data(), U_new.data(), ni, ni, ig0, N, ld, m, lam, TILE);
        }

        // Fase 2: completa U_old
        {
            MPI_Request reqs[4];
            const int nreq = exchange_halo_rows_start(U_old.data(), ni, ld, nbr_north, nbr_south, reqs, MPI_COMM_WORLD);
            phase2_semi(U_old.data(), U_new.data(), 2, ni - 1, ig0, N, ld, m, lam, denom, TILE);
            wait_exchange(nreq, reqs);
            if (ni >= 1) phase2_semi(U_old.data(), U_new.data(), 1, 1, ig0, N, ld, m, lam, denom, TILE);
            if (ni >= 2) phase2_semi(U_old.data(), U_new.data(), ni, ni, ig0, N, ld, m, lam, denom, TILE);
        }

        // Fase 3: U_old -> U_new
        {
            MPI_Request reqs[4];
            const int nreq = exchange_halo_rows_start(U_old.data(), ni, ld, nbr_north, nbr_south, reqs, MPI_COMM_WORLD);
            phase3_explicit(U_new.data(), U_old.data(), 2, ni - 1, ig0, N, ld, m, lam, TILE);
            wait_exchange(nreq, reqs);
            if (ni >= 1) phase3_explicit(U_new.data(), U_old.data(), 1, 1, ig0, N, ld, m, lam, TILE);
            if (ni >= 2) phase3_explicit(U_new.data(), U_old.data(), ni, ni, ig0, N, ld, m, lam, TILE);
        }

        // Fase 4: completa U_new
        {
            MPI_Request reqs[4];
            const int nreq = exchange_halo_rows_start(U_new.data(), ni, ld, nbr_north, nbr_south, reqs, MPI_COMM_WORLD);
            phase4_semi(U_new.data(), U_old.data(), 2, ni - 1, ig0, N, ld, m, lam, denom, TILE);
            wait_exchange(nreq, reqs);
            if (ni >= 1) phase4_semi(U_new.data(), U_old.data(), 1, 1, ig0, N, ld, m, lam, denom, TILE);
            if (ni >= 2) phase4_semi(U_new.data(), U_old.data(), ni, ni, ig0, N, ld, m, lam, denom, TILE);
        }

        ++m;
        ++m;
    }

    const double t1 = MPI_Wtime();
    const double elapsed = t1 - t0;
    double elapsed_max = 0.0;
    MPI_Reduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Tempo: " << elapsed_max << " s\n";
    }

    // Saída única (output.txt)
    std::vector<int> ig_local, jg_local;
    std::vector<double> val_local;

    auto first_on_stride = [](int global_start)->int {
        int r = (16 - (global_start % 16)) % 16;
        return 1 + r;
    };

    const int i_first = (ni > 0) ? first_on_stride(ig0) : 1;
    const int j_first = 16;

    for (int i = i_first; i <= ni; i += 16) {
        const int ig = ig0 + (i - 1);
        for (int j = j_first; j <= N - 2; j += 16) {
            ig_local.push_back(ig);
            jg_local.push_back(j);
            val_local.push_back(U_new[idx2(i, j, ld)]);
        }
    }

    const int mloc = static_cast<int>(ig_local.size());
    std::vector<int> counts, displs;
    int gtot = 0;

    if (rank == 0) counts.resize(size);
    MPI_Gather(&mloc, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs.resize(size, 0);
        for (int r = 0; r < size; ++r) {
            displs[r] = gtot;
            gtot += counts[r];
        }
    }

    std::vector<int> ig_all, jg_all;
    std::vector<double> val_all;
    if (rank == 0) {
        ig_all.resize(gtot);
        jg_all.resize(gtot);
        val_all.resize(gtot);
    }

    MPI_Gatherv(ig_local.data(),  mloc, MPI_INT,
                ig_all.data(), counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(jg_local.data(),  mloc, MPI_INT,
                jg_all.data(), counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(val_local.data(), mloc, MPI_DOUBLE,
                val_all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::unordered_map<std::uint64_t, double> mapv;
        mapv.reserve(static_cast<size_t>(gtot) * 13 / 10 + 1);
        auto key = [](int ig, int jg)->std::uint64_t {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(ig)) << 32)
                 |  static_cast<std::uint32_t>(jg);
        };
        for (int k = 0; k < gtot; ++k) {
            mapv[key(ig_all[k], jg_all[k])] = val_all[k];
        }

        std::ofstream fout("output.txt");
        if (!fout) {
            std::cerr << "Erro: não foi possível abrir output.txt para escrita\n";
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        fout.setf(std::ios::fixed);
        fout.precision(8);

        for (int i = 0; i < N; i += 16) {
            const double x = i * h;
            for (int j = 0; j < N; j += 16) {
                const double y = j * h;
                double v = 0.0;
                if (i >= 1 && i <= N - 2 && j >= 1 && j <= N - 2) {
                    auto it = mapv.find(key(i, j));
                    if (it != mapv.end()) v = it->second;
                }
                fout << x << " " << y << " " << v << "\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}