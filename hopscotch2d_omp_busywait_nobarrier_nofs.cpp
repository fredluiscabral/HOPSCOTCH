// hopscotch2d_omp_busywait_nobarrier_nofs.cpp
// Equação do calor 2D (Hopscotch) — OpenMP com divisão em faixas e espera ocupada local,
// SEM barreira global por passo. Versão com mitigação explícita de false sharing,
// pensando na microarquitetura AMD EPYC (linha de cache de 64 B).
//
// Estratégias implementadas:
//   1) progress[] com 1 slot atômico por thread, alinhado e padded para 1 cache line.
//   2) U_new/U_old alocados com alinhamento de 64 B.
//   3) leading dimension (ld) padded para múltiplo de 8 doubles (= 64 B), evitando
//      compartilhamento espúrio entre linhas adjacentes pertencentes a threads diferentes.
//   4) inicialização "first-touch" paralela para melhorar localidade/NUMA em EPYC.
//   5) spin-wait com pause/backoff leve, reduzindo pressão de coerência.
//
// param.txt é OBRIGATÓRIO. Mede o tempo do laço principal.
// Saída: output.txt (amostrado a cada 16 pontos).

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
#include <new>

#include <omp.h>

#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
  static inline void spin_pause() noexcept { _mm_pause(); }
#else
  static inline void spin_pause() noexcept { std::this_thread::yield(); }
#endif

#ifndef CACHELINE_BYTES
#define CACHELINE_BYTES 64u
#endif

static_assert((CACHELINE_BYTES & (CACHELINE_BYTES - 1u)) == 0u,
              "CACHELINE_BYTES deve ser potência de 2");

constexpr std::size_t kCacheLineBytes = CACHELINE_BYTES;
constexpr std::size_t kDoublesPerCacheLine = kCacheLineBytes / sizeof(double);
static_assert(kDoublesPerCacheLine > 0, "Cache line inválida");

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

static inline std::size_t round_up(std::size_t x, std::size_t a) noexcept {
    return ((x + a - 1u) / a) * a;
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

// Leitura estrita de param.txt (todos obrigatórios)
bool load_params_strict(const std::string& fname,
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
        std::cerr << "Erro: 1<=TILE<=N-2.\n";
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

struct alignas(kCacheLineBytes) ProgressSlot {
    std::atomic<int> value;
    char padding[kCacheLineBytes - sizeof(std::atomic<int>)];
};
static_assert(sizeof(ProgressSlot) == kCacheLineBytes,
              "ProgressSlot deve ocupar exatamente 1 cache line");

static inline void wait_until_at_least(const std::atomic<int>& slot, int expected) noexcept {
    unsigned spins = 0;
    while (slot.load(std::memory_order_acquire) < expected) {
        spin_pause();
        spin_pause();
        spin_pause();
        spin_pause();
        if ((++spins & 0x3FFu) == 0u) {
            std::this_thread::yield();
        }
    }
}

int main() {

    #pragma omp parallel
    {
        #pragma omp single
        {
            std::printf("Threads: %d\n", omp_get_num_threads());
        }
    }

    // ---- Leitura de parâmetros ----
    int N=0, T=0, TILE=0; double alpha=0.0;
    if (!load_params_strict("param.txt", N, alpha, T, TILE)) return 1;

    // ---- Discretização ----
    const double h   = 1.0 / (N - 1);
    const double dt  = 0.90 * (h*h) / (4.0 * alpha);   // 90% do limite estável explícito 2D
    const double lam = alpha * dt / (h*h);
    const double denom = 1.0 + 4.0 * lam;

    // ---- Layout dos campos ----
    // ld padded para múltiplo de 1 linha de cache em doubles.
    const std::size_t ld = round_up(static_cast<std::size_t>(N), kDoublesPerCacheLine);
    const std::size_t NN = ld * static_cast<std::size_t>(N);

    AlignedBuffer<double> U_new, U_old;
    if (!U_new.allocate(NN, kCacheLineBytes) || !U_old.allocate(NN, kCacheLineBytes)) {
        std::cerr << "Erro: falha na alocação alinhada dos campos.\n";
        return 1;
    }

    auto idx = [ld](int i, int j)->std::size_t {
        return static_cast<std::size_t>(i) * ld + static_cast<std::size_t>(j);
    };

    // ---- Inicialização first-touch paralela ----
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

            const int ib = std::max(1, rows.first);
            const int ie = std::min(N-2, rows.last);
            for (int i = ib; i <= ie; ++i) {
                const double x = i * h;
                for (int j = 1; j <= N-2; ++j) {
                    const double y = j * h;
                    U_new[idx(i,j)] = std::exp(-D*((x-x0)*(x-x0)+(y-y0)*(y-y0)));
                }
            }

            for (int i = rows.first; i <= rows.last; ++i) {
                U_new[idx(i,0)]   = 0.0;
                U_new[idx(i,N-1)] = 0.0;
                U_old[idx(i,0)]   = 0.0;
                U_old[idx(i,N-1)] = 0.0;
            }

            if (rows.first == 0) {
                for (int j = 0; j < N; ++j) {
                    U_new[idx(0,j)] = 0.0;
                    U_old[idx(0,j)] = 0.0;
                }
            }
            if (rows.last == N-1) {
                for (int j = 0; j < N; ++j) {
                    U_new[idx(N-1,j)] = 0.0;
                    U_old[idx(N-1,j)] = 0.0;
                }
            }
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Paraleliza por faixas (linhas) com sincronização local SOMENTE (sem barreiras globais)
    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();

        const Range interior = split_closed_interval(1, N-2, tid, nt);
        const int iLocal = interior.first;
        const int fLocal = interior.last;

        // Array de progresso compartilhado (uma vez), padded para evitar false sharing.
        static std::unique_ptr<ProgressSlot[]> progress;
        #pragma omp single
        {
            progress = std::make_unique<ProgressSlot[]>(static_cast<std::size_t>(nt));
            for (int p = 0; p < nt; ++p) {
                progress[p].value.store(0, std::memory_order_relaxed);
            }
        }
        #pragma omp barrier // publica 'progress' para todas as threads

        auto wait_neighbors_atleast = [&](int expected) noexcept {
            if (tid > 0)    wait_until_at_least(progress[tid-1].value, expected);
            if (tid < nt-1) wait_until_at_least(progress[tid+1].value, expected);
        };

        int m_local = 0; // paridade

        for (int step = 0; step < T; ++step) {
            // ====== FASE 1 (explícita) ======
            wait_neighbors_atleast(4*step + 0);
            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + m_local) & 1) == 0) {
                                U_old[idx(i,j)] = U_new[idx(i,j)]
                                    + lam * ( U_new[idx(i+1,j)] + U_new[idx(i-1,j)]
                                            + U_new[idx(i,j+1)] + U_new[idx(i,j-1)]
                                            - 4.0 * U_new[idx(i,j)] );
                            } else {
                                U_old[idx(i,j)] = U_new[idx(i,j)];
                            }
                        }
                    }
                }
            }
            progress[tid].value.store(4*step + 1, std::memory_order_release);

            // ====== FASE 2 (semi-implícita) ======
            wait_neighbors_atleast(4*step + 1);
            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + m_local) & 1) == 1) {
                                U_old[idx(i,j)] = ( U_new[idx(i,j)]
                                    + lam * ( U_old[idx(i+1,j)] + U_old[idx(i-1,j)]
                                            + U_old[idx(i,j+1)] + U_old[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            progress[tid].value.store(4*step + 2, std::memory_order_release);
            ++m_local;

            // ====== FASE 3 (explícita) ======
            wait_neighbors_atleast(4*step + 2);
            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + m_local) & 1) == 0) {
                                U_new[idx(i,j)] = U_old[idx(i,j)]
                                    + lam * ( U_old[idx(i+1,j)] + U_old[idx(i-1,j)]
                                            + U_old[idx(i,j+1)] + U_old[idx(i,j-1)]
                                            - 4.0 * U_old[idx(i,j)] );
                            } else {
                                U_new[idx(i,j)] = U_old[idx(i,j)];
                            }
                        }
                    }
                }
            }
            progress[tid].value.store(4*step + 3, std::memory_order_release);

            // ====== FASE 4 (semi-implícita) ======
            wait_neighbors_atleast(4*step + 3);
            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + m_local) & 1) == 1) {
                                U_new[idx(i,j)] = ( U_old[idx(i,j)]
                                    + lam * ( U_new[idx(i+1,j)] + U_new[idx(i-1,j)]
                                            + U_new[idx(i,j+1)] + U_new[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            progress[tid].value.store(4*step + 4, std::memory_order_release);
            ++m_local;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Tempo : " << secs << " s\n";

    // ---- Saída (amostrada) ----
    std::ofstream fout("output.txt");
    if (!fout) {
        std::cerr << "Erro ao abrir output.txt\n";
        return 1;
    }
    fout.setf(std::ios::fixed);
    fout.precision(8);
    for (int i = 0; i < N; i += 16) {
        const double x = i * h;
        for (int j = 0; j < N; j += 16) {
            const double y = j * h;
            fout << x << " " << y << " " << U_new[idx(i,j)] << "\n";
        }
    }

    return 0;
}