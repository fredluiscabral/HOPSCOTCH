// hopscotch2d_omp_sem_nobarrier_nofs.cpp
// Hopscotch 2D (equação do calor) — OpenMP + semáforos POSIX
// Sincronização local por fase (vizinho-vizinho) e rendezvous local por passo,
// SEM barreira global por passo. Versão com mitigação explícita de false sharing,
// pensando na microarquitetura AMD EPYC (cache line de 64 B).
//
// Estratégias implementadas para reduzir false sharing / tráfego espúrio:
//   1) U_new/U_old com alocação alinhada em 64 B.
//   2) leading dimension (ld) padded para múltiplo de 8 doubles (= 64 B).
//   3) semáforos de fase e de passo isolados em slots alinhados/padded por thread.
//   4) inicialização "first-touch" paralela para melhorar localidade/NUMA.
//
// param.txt é OBRIGATÓRIO. Mede o tempo do laço principal.
// Saída: output.txt (amostrado a cada 16 pontos).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cctype>
#include <memory>
#include <new>
#include <type_traits>

#include <omp.h>
#include <semaphore.h>

#ifndef CACHELINE_BYTES
#define CACHELINE_BYTES 64u
#endif

static_assert((CACHELINE_BYTES & (CACHELINE_BYTES - 1u)) == 0u,
              "CACHELINE_BYTES deve ser potência de 2");

constexpr std::size_t kCacheLineBytes = CACHELINE_BYTES;
constexpr std::size_t kDoublesPerCacheLine = kCacheLineBytes / sizeof(double);
static_assert(kDoublesPerCacheLine > 0, "Cache line inválida");

static inline std::size_t round_up(std::size_t x, std::size_t a) noexcept {
    return ((x + a - 1u) / a) * a;
}

constexpr std::size_t round_up_constexpr(std::size_t x, std::size_t a) noexcept {
    return ((x + a - 1u) / a) * a;
}

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

struct Range {
    int first;
    int last;
    bool empty() const noexcept { return first > last; }
};

static inline Range split_closed_interval(int first, int last, int tid, int nt) noexcept {
    if (first > last) return {1, 0};
    const int n = last - first + 1;
    const int H = n / nt;
    const int R = n % nt;
    const int begin = first + tid * H + std::min(tid, R);
    const int end   = begin + H - 1 + (tid < R ? 1 : 0);
    return {begin, end};
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
        N     = std::stoi(kv["n"]);
        alpha = std::stod(kv["alpha"]);
        T     = std::stoi(kv["t"]);
        TILE  = std::stoi(kv["tile"]);
    } catch (const std::exception& e) {
        std::cerr << "Erro: falha ao converter parâmetros de " << fname << ": " << e.what() << "\n";
        return false;
    }

    if (N < 3)             { std::cerr << "Erro: N >= 3.\n"; return false; }
    if (alpha <= 0.0)      { std::cerr << "Erro: alpha > 0.\n"; return false; }
    if (T < 0)             { std::cerr << "Erro: T >= 0.\n"; return false; }
    if (TILE < 1 || TILE > N-2) {
        std::cerr << "Erro: 1 <= tile <= N-2 (tile=" << TILE << ", N=" << N << ").\n";
        return false;
    }
    return true;
}

template <typename T>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            std::free(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~AlignedBuffer() {
        std::free(ptr_);
    }

    bool allocate(std::size_t n, std::size_t alignment) {
        std::free(ptr_);
        ptr_ = nullptr;
        size_ = 0;

        if (n == 0) return true;

        const std::size_t bytes = n * sizeof(T);
        const std::size_t padded_bytes = round_up(bytes, alignment);
        void* raw = nullptr;
        if (posix_memalign(&raw, alignment, padded_bytes) != 0) {
            return false;
        }
        ptr_ = static_cast<T*>(raw);
        size_ = n;
        return true;
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return size_; }

    T& operator[](std::size_t i) noexcept { return ptr_[i]; }
    const T& operator[](std::size_t i) const noexcept { return ptr_[i]; }

private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

constexpr std::size_t kSemSlotBytes = round_up_constexpr(sizeof(sem_t), kCacheLineBytes);

struct alignas(kCacheLineBytes) SemSlot {
    std::aligned_storage_t<kSemSlotBytes, kCacheLineBytes> storage;

    sem_t* ptr() noexcept {
        return reinterpret_cast<sem_t*>(&storage);
    }
    const sem_t* ptr() const noexcept {
        return reinterpret_cast<const sem_t*>(&storage);
    }
};

static_assert(sizeof(SemSlot) == kSemSlotBytes,
              "SemSlot deve ocupar um número inteiro de cache lines");
static_assert(alignof(SemSlot) == kCacheLineBytes,
              "SemSlot deve ser alinhado a 1 cache line");

int main() {
    omp_set_dynamic(0);

    #pragma omp parallel
    {
        #pragma omp single
        {
            std::printf("Threads: %d\n", omp_get_num_threads());
        }
    }

    int N=0, T=0, TILE=0;
    double alpha=0.0;
    if (!load_params_strict("param.txt", N, alpha, T, TILE)) {
        return 1;
    }

    const double h   = 1.0 / (N - 1);
    const double dt  = 0.90 * (h*h) / (4.0 * alpha);
    const double lam = alpha * dt / (h*h);
    const double denom = 1.0 + 4.0 * lam;

    const std::size_t ld = round_up(static_cast<std::size_t>(N), kDoublesPerCacheLine);
    const std::size_t NN = ld * static_cast<std::size_t>(N);

    AlignedBuffer<double> U_new, U_old;
    if (!U_new.allocate(NN, kCacheLineBytes) || !U_old.allocate(NN, kCacheLineBytes)) {
        std::cerr << "Erro: falha na alocação alinhada dos campos.\n";
        return 1;
    }

    auto idx = [ld](int i, int j) -> std::size_t {
        return static_cast<std::size_t>(i) * ld + static_cast<std::size_t>(j);
    };

    const double D  = 100.0;
    const double x0 = 0.5;
    const double y0 = 0.5;

    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();
        const Range rows = split_closed_interval(0, N-1, tid, nt);

        if (!rows.empty()) {
            for (int i = rows.first; i <= rows.last; ++i) {
                std::fill_n(U_new.data() + static_cast<std::size_t>(i) * ld, ld, 0.0);
                std::fill_n(U_old.data() + static_cast<std::size_t>(i) * ld, ld, 0.0);
            }

            for (int i = std::max(1, rows.first); i <= std::min(N-2, rows.last); ++i) {
                const double x = i * h;
                for (int j = 1; j <= N-2; ++j) {
                    const double y = j * h;
                    U_new[idx(i,j)] = std::exp(-D * ((x-x0)*(x-x0) + (y-y0)*(y-y0)));
                }
            }

            if (rows.first == 0) {
                for (int j = 0; j < N; ++j) {
                    U_new[idx(0, j)] = 0.0;
                    U_old[idx(0, j)] = 0.0;
                }
            }
            if (rows.last == N-1) {
                for (int j = 0; j < N; ++j) {
                    U_new[idx(N-1, j)] = 0.0;
                    U_old[idx(N-1, j)] = 0.0;
                }
            }
            for (int i = rows.first; i <= rows.last; ++i) {
                U_new[idx(i, 0)]   = 0.0;
                U_new[idx(i, N-1)] = 0.0;
                U_old[idx(i, 0)]   = 0.0;
                U_old[idx(i, N-1)] = 0.0;
            }
        }
    }

    const int max_threads = omp_get_max_threads();
    auto sem_left  = std::unique_ptr<SemSlot[]>(new (std::nothrow) SemSlot[max_threads]);
    auto sem_right = std::unique_ptr<SemSlot[]>(new (std::nothrow) SemSlot[max_threads]);
    auto step_left = std::unique_ptr<SemSlot[]>(new (std::nothrow) SemSlot[max_threads]);
    auto step_right= std::unique_ptr<SemSlot[]>(new (std::nothrow) SemSlot[max_threads]);

    if (!sem_left || !sem_right || !step_left || !step_right) {
        std::cerr << "Erro: falha ao alocar slots de semáforos.\n";
        return 1;
    }

    for (int t = 0; t < max_threads; ++t) {
        if (sem_init(sem_left[t].ptr(),   0, 0) != 0 ||
            sem_init(sem_right[t].ptr(),  0, 0) != 0 ||
            sem_init(step_left[t].ptr(),  0, 0) != 0 ||
            sem_init(step_right[t].ptr(), 0, 0) != 0) {
            std::perror("sem_init");
            for (int k = 0; k <= t; ++k) {
                sem_destroy(sem_left[k].ptr());
                sem_destroy(sem_right[k].ptr());
                sem_destroy(step_left[k].ptr());
                sem_destroy(step_right[k].ptr());
            }
            return 1;
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(shared)
    {
        const int nt  = omp_get_num_threads();
        const int tid = omp_get_thread_num();

        const int interior = (N - 2);
        const int baseSize = interior / nt;
        const int sobra    = interior % nt;
        const int iLocal   = 1 + tid * baseSize + std::min(tid, sobra);
        const int mySize   = baseSize + (tid < sobra ? 1 : 0);
        const int fLocal   = iLocal + mySize - 1;

        auto signal_done_phase = [&](int t) {
            sem_post(sem_left[t].ptr());
            sem_post(sem_right[t].ptr());
        };
        auto wait_neighbors_phase = [&](int t) {
            if (t > 0)      sem_wait(sem_right[t-1].ptr());
            if (t < nt-1)   sem_wait(sem_left[t+1].ptr());
        };
        auto announce_step = [&](int t) {
            sem_post(step_left[t].ptr());
            sem_post(step_right[t].ptr());
        };
        auto wait_step_neighbors = [&](int t) {
            if (t > 0)      sem_wait(step_right[t-1].ptr());
            if (t < nt-1)   sem_wait(step_left[t+1].ptr());
        };

        for (int s = 0; s < T; ++s) {
            if (s > 0) {
                wait_step_neighbors(tid);
            }

            int mloc = 2 * s;

            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + mloc) & 1) == 0) {
                                U_old[idx(i,j)] = U_new[idx(i,j)] +
                                    lam * ( U_new[idx(i+1,j)] + U_new[idx(i-1,j)]
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

            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + mloc) & 1) == 1) {
                                U_old[idx(i,j)] =
                                    ( U_new[idx(i,j)]
                                    + lam * ( U_old[idx(i+1,j)] + U_old[idx(i-1,j)]
                                            + U_old[idx(i,j+1)] + U_old[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            signal_done_phase(tid);
            wait_neighbors_phase(tid);
            ++mloc;

            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + mloc) & 1) == 0) {
                                U_new[idx(i,j)] = U_old[idx(i,j)] +
                                    lam * ( U_old[idx(i+1,j)] + U_old[idx(i-1,j)]
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

            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + mloc) & 1) == 1) {
                                U_new[idx(i,j)] =
                                    ( U_old[idx(i,j)]
                                    + lam * ( U_new[idx(i+1,j)] + U_new[idx(i-1,j)]
                                            + U_new[idx(i,j+1)] + U_new[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            signal_done_phase(tid);
            wait_neighbors_phase(tid);
            ++mloc;

            announce_step(tid);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Tempo : " << secs << " s\n";

    for (int t = 0; t < max_threads; ++t) {
        sem_destroy(sem_left[t].ptr());
        sem_destroy(sem_right[t].ptr());
        sem_destroy(step_left[t].ptr());
        sem_destroy(step_right[t].ptr());
    }

    std::ofstream fout("output.txt");
    if (!fout) {
        std::cerr << "Erro: não foi possível abrir output.txt para escrita\n";
        return 1;
    }
    fout.setf(std::ios::fixed);
    fout.precision(8);
    for (int i = 0; i < N; i += 16) {
        const double x = i * h;
        for (int j = 0; j < N; j += 16) {
            const double y = j * h;
            fout << x << ' ' << y << ' ' << U_new[idx(i,j)] << "\n";
        }
    }
    fout.close();

    return 0;
}
