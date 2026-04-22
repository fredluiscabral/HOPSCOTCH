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
#include <chrono>
#include <cctype>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <omp.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#  include <immintrin.h>
#endif

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

static inline std::string tolower_str(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

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
        std::string val = trim(line.substr(pos + 1));
        kv[tolower_str(key)] = val;
    }

    try {
        if (!kv.count("n"))     { std::cerr << "Erro: parâmetro obrigatório 'n' ausente em " << fname << ".\n"; return false; }
        if (!kv.count("alpha")) { std::cerr << "Erro: parâmetro obrigatório 'alpha' ausente em " << fname << ".\n"; return false; }
        if (!kv.count("t"))     { std::cerr << "Erro: parâmetro obrigatório 't' ausente em " << fname << ".\n"; return false; }
        if (!kv.count("tile"))  { std::cerr << "Erro: parâmetro obrigatório 'tile' ausente em " << fname << ".\n"; return false; }

        N     = std::stoi(kv["n"]);
        alpha = std::stod(kv["alpha"]);
        T     = std::stoi(kv["t"]);
        TILE  = std::stoi(kv["tile"]);
    } catch (const std::exception& e) {
        std::cerr << "Erro: falha ao converter parâmetros de " << fname << ": "
                  << e.what() << "\n";
        return false;
    }

    if (N < 3) {
        std::cerr << "Erro: N deve ser >= 3 (obtido N=" << N << ").\n";
        return false;
    }
    if (alpha <= 0.0) {
        std::cerr << "Erro: alpha deve ser > 0 (obtido alpha=" << alpha << ").\n";
        return false;
    }
    if (T < 0) {
        std::cerr << "Erro: T deve ser >= 0 (obtido T=" << T << ").\n";
        return false;
    }
    if (TILE < 1 || TILE > N - 2) {
        std::cerr << "Erro: tile deve satisfazer 1 <= tile <= N-2 (obtido tile="
                  << TILE << ", N=" << N << ").\n";
        return false;
    }

    return true;
}

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
static inline void spin_pause() noexcept { _mm_pause(); }
#else
static inline void spin_pause() noexcept { std::this_thread::yield(); }
#endif

static inline void wait_until_at_least(const std::atomic<int>& slot,
                                       int expected) noexcept {
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

#ifndef CACHELINE_BYTES
#define CACHELINE_BYTES 64u
#endif

static inline std::size_t round_up(std::size_t value, std::size_t mult) {
    return ((value + mult - 1u) / mult) * mult;
}

template <typename T>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    ~AlignedBuffer() { reset(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return size_; }

    void allocate(std::size_t n, std::size_t alignment = CACHELINE_BYTES) {
        reset();
        if (n == 0) return;
        void* raw = nullptr;
        if (posix_memalign(&raw, alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        ptr_ = static_cast<T*>(raw);
        size_ = n;
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            std::free(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T& operator[](std::size_t idx) noexcept { return ptr_[idx]; }
    const T& operator[](std::size_t idx) const noexcept { return ptr_[idx]; }

private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

struct alignas(CACHELINE_BYTES) PaddedAtomicInt {
    std::atomic<int> value;
    char pad[CACHELINE_BYTES - sizeof(std::atomic<int>)];
    PaddedAtomicInt() : value(0), pad{} {}
};
static_assert(sizeof(PaddedAtomicInt) == CACHELINE_BYTES,
              "PaddedAtomicInt must occupy one cache line");

struct ThreadDomain {
    int start_g = 1;
    int end_g   = 0;
    int local_n = 0;
    std::size_t ld = 0;

    AlignedBuffer<double> u_new;
    AlignedBuffer<double> u_old;

    AlignedBuffer<double> halo_unew_top[2];
    AlignedBuffer<double> halo_unew_bottom[2];
    AlignedBuffer<double> halo_uold_top[2];
    AlignedBuffer<double> halo_uold_bottom[2];

    PaddedAtomicInt unew_top_ready;
    PaddedAtomicInt unew_bottom_ready;
    PaddedAtomicInt uold_top_ready;
    PaddedAtomicInt uold_bottom_ready;
};

static inline std::size_t L(int il, int j, std::size_t ld) noexcept {
    return static_cast<std::size_t>(il) * ld + static_cast<std::size_t>(j);
}

static inline void copy_row(double* dst, const double* src, std::size_t ld) noexcept {
    std::memcpy(dst, src, ld * sizeof(double));
}

static inline void publish_unew(ThreadDomain* doms, int tid, int nt, int epoch) noexcept {
    ThreadDomain& me = doms[tid];
    const int slot = epoch & 1;
    const std::size_t ld = me.ld;
    const double* src_top    = me.u_new.data() + L(1, 0, ld);
    const double* src_bottom = me.u_new.data() + L(me.local_n, 0, ld);

    if (tid > 0) {
        ThreadDomain& up = doms[tid - 1];
        copy_row(up.halo_unew_bottom[slot].data(), src_top, ld);
        up.unew_bottom_ready.value.store(epoch, std::memory_order_release);
    }
    if (tid + 1 < nt) {
        ThreadDomain& down = doms[tid + 1];
        copy_row(down.halo_unew_top[slot].data(), src_bottom, ld);
        down.unew_top_ready.value.store(epoch, std::memory_order_release);
    }
}

static inline void publish_uold(ThreadDomain* doms, int tid, int nt, int epoch) noexcept {
    ThreadDomain& me = doms[tid];
    const int slot = epoch & 1;
    const std::size_t ld = me.ld;
    const double* src_top    = me.u_old.data() + L(1, 0, ld);
    const double* src_bottom = me.u_old.data() + L(me.local_n, 0, ld);

    if (tid > 0) {
        ThreadDomain& up = doms[tid - 1];
        copy_row(up.halo_uold_bottom[slot].data(), src_top, ld);
        up.uold_bottom_ready.value.store(epoch, std::memory_order_release);
    }
    if (tid + 1 < nt) {
        ThreadDomain& down = doms[tid + 1];
        copy_row(down.halo_uold_top[slot].data(), src_bottom, ld);
        down.uold_top_ready.value.store(epoch, std::memory_order_release);
    }
}

static inline void ensure_unew_halos(ThreadDomain& dom,
                                     int tid,
                                     int nt,
                                     int epoch) noexcept {
    const int slot = epoch & 1;
    const std::size_t ld = dom.ld;

    if (tid > 0) {
        wait_until_at_least(dom.unew_top_ready.value, epoch);
        copy_row(dom.u_new.data() + L(0, 0, ld),
                 dom.halo_unew_top[slot].data(), ld);
    }
    if (tid + 1 < nt) {
        wait_until_at_least(dom.unew_bottom_ready.value, epoch);
        copy_row(dom.u_new.data() + L(dom.local_n + 1, 0, ld),
                 dom.halo_unew_bottom[slot].data(), ld);
    }
}

static inline void ensure_uold_halos(ThreadDomain& dom,
                                     int tid,
                                     int nt,
                                     int epoch) noexcept {
    const int slot = epoch & 1;
    const std::size_t ld = dom.ld;

    if (tid > 0) {
        wait_until_at_least(dom.uold_top_ready.value, epoch);
        copy_row(dom.u_old.data() + L(0, 0, ld),
                 dom.halo_uold_top[slot].data(), ld);
    }
    if (tid + 1 < nt) {
        wait_until_at_least(dom.uold_bottom_ready.value, epoch);
        copy_row(dom.u_old.data() + L(dom.local_n + 1, 0, ld),
                 dom.halo_uold_bottom[slot].data(), ld);
    }
}

static inline void phase1_full(ThreadDomain& dom,
                               int N,
                               int TILE,
                               double lam,
                               int m0) noexcept {
    const std::size_t ld = dom.ld;
    for (int ii = 1; ii <= dom.local_n; ii += TILE) {
        const int i_end = std::min(dom.local_n, ii + TILE - 1);
        for (int jj = 1; jj <= N - 2; jj += TILE) {
            const int j_end = std::min(N - 2, jj + TILE - 1);
            for (int i = ii; i <= i_end; ++i) {
                const int ig = dom.start_g + (i - 1);
                double* row_old = dom.u_old.data() + L(i, 0, ld);
                const double* row_new    = dom.u_new.data() + L(i, 0, ld);
                const double* row_new_up = dom.u_new.data() + L(i - 1, 0, ld);
                const double* row_new_dn = dom.u_new.data() + L(i + 1, 0, ld);
                for (int j = jj; j <= j_end; ++j) {
                    if (((ig + j + m0) & 1) == 0) {
                        row_old[j] = row_new[j] +
                                     lam * (row_new_dn[j] + row_new_up[j] +
                                            row_new[j + 1] + row_new[j - 1] -
                                            4.0 * row_new[j]);
                    } else {
                        row_old[j] = row_new[j];
                    }
                }
            }
        }
    }
}

static inline void phase2_full(ThreadDomain& dom,
                               int N,
                               int TILE,
                               double lam,
                               double denom,
                               int m0) noexcept {
    const std::size_t ld = dom.ld;
    for (int ii = 1; ii <= dom.local_n; ii += TILE) {
        const int i_end = std::min(dom.local_n, ii + TILE - 1);
        for (int jj = 1; jj <= N - 2; jj += TILE) {
            const int j_end = std::min(N - 2, jj + TILE - 1);
            for (int i = ii; i <= i_end; ++i) {
                const int ig = dom.start_g + (i - 1);
                double* row_old = dom.u_old.data() + L(i, 0, ld);
                const double* row_new    = dom.u_new.data() + L(i, 0, ld);
                const double* row_old_up = dom.u_old.data() + L(i - 1, 0, ld);
                const double* row_old_dn = dom.u_old.data() + L(i + 1, 0, ld);
                for (int j = jj; j <= j_end; ++j) {
                    if (((ig + j + m0) & 1) == 1) {
                        row_old[j] = (row_new[j] +
                                     lam * (row_old_dn[j] + row_old_up[j] +
                                            row_old[j + 1] + row_old[j - 1])) / denom;
                    }
                }
            }
        }
    }
}

static inline void phase3_full(ThreadDomain& dom,
                               int N,
                               int TILE,
                               double lam,
                               int m1) noexcept {
    const std::size_t ld = dom.ld;
    for (int ii = 1; ii <= dom.local_n; ii += TILE) {
        const int i_end = std::min(dom.local_n, ii + TILE - 1);
        for (int jj = 1; jj <= N - 2; jj += TILE) {
            const int j_end = std::min(N - 2, jj + TILE - 1);
            for (int i = ii; i <= i_end; ++i) {
                const int ig = dom.start_g + (i - 1);
                double* row_new = dom.u_new.data() + L(i, 0, ld);
                const double* row_old    = dom.u_old.data() + L(i, 0, ld);
                const double* row_old_up = dom.u_old.data() + L(i - 1, 0, ld);
                const double* row_old_dn = dom.u_old.data() + L(i + 1, 0, ld);
                for (int j = jj; j <= j_end; ++j) {
                    if (((ig + j + m1) & 1) == 0) {
                        row_new[j] = row_old[j] +
                                     lam * (row_old_dn[j] + row_old_up[j] +
                                            row_old[j + 1] + row_old[j - 1] -
                                            4.0 * row_old[j]);
                    } else {
                        row_new[j] = row_old[j];
                    }
                }
            }
        }
    }
}

static inline void phase4_full(ThreadDomain& dom,
                               int N,
                               int TILE,
                               double lam,
                               double denom,
                               int m1) noexcept {
    const std::size_t ld = dom.ld;
    for (int ii = 1; ii <= dom.local_n; ii += TILE) {
        const int i_end = std::min(dom.local_n, ii + TILE - 1);
        for (int jj = 1; jj <= N - 2; jj += TILE) {
            const int j_end = std::min(N - 2, jj + TILE - 1);
            for (int i = ii; i <= i_end; ++i) {
                const int ig = dom.start_g + (i - 1);
                double* row_new = dom.u_new.data() + L(i, 0, ld);
                const double* row_old    = dom.u_old.data() + L(i, 0, ld);
                const double* row_new_up = dom.u_new.data() + L(i - 1, 0, ld);
                const double* row_new_dn = dom.u_new.data() + L(i + 1, 0, ld);
                for (int j = jj; j <= j_end; ++j) {
                    if (((ig + j + m1) & 1) == 1) {
                        row_new[j] = (row_old[j] +
                                     lam * (row_new_dn[j] + row_new_up[j] +
                                            row_new[j + 1] + row_new[j - 1])) / denom;
                    }
                }
            }
        }
    }
}

int main() {
    omp_set_dynamic(0);

    int N = 0, T = 0, TILE = 0;
    double alpha = 0.0;
    if (!load_params_strict("param.txt", N, alpha, T, TILE)) {
        return 1;
    }

    const int interior_rows = N - 2;
    if (interior_rows <= 0) {
        std::cerr << "Erro: domínio interior vazio.\n";
        return 1;
    }

    int num_threads = std::min(omp_get_max_threads(), interior_rows);
    if (num_threads < 1) num_threads = 1;

    std::cout << "Threads: " << num_threads << "\n";

    const double h     = 1.0 / (N - 1);
    const double dt    = 0.90 * (h * h) / (4.0 * alpha);
    const double lam   = alpha * dt / (h * h);
    const double denom = 1.0 + 4.0 * lam;

    const std::size_t ld =
        round_up(static_cast<std::size_t>(N),
                 CACHELINE_BYTES / sizeof(double));

    std::unique_ptr<ThreadDomain[]> doms(new ThreadDomain[num_threads]);
    std::vector<int> owner_of_row(static_cast<std::size_t>(N), -1);

    for (int tid = 0; tid < num_threads; ++tid) {
        const int H = interior_rows / num_threads;
        const int R = interior_rows % num_threads;
        const int local_n = H + (tid < R ? 1 : 0);
        const int start_g = 1 + tid * H + std::min(tid, R);
        const int end_g   = start_g + local_n - 1;

        ThreadDomain& dom = doms[tid];
        dom.start_g = start_g;
        dom.end_g   = end_g;
        dom.local_n = local_n;
        dom.ld      = ld;

        const std::size_t main_elems =
            static_cast<std::size_t>(local_n + 2) * ld;

        dom.u_new.allocate(main_elems, CACHELINE_BYTES);
        dom.u_old.allocate(main_elems, CACHELINE_BYTES);

        for (int k = 0; k < 2; ++k) {
            dom.halo_unew_top[k].allocate(ld, CACHELINE_BYTES);
            dom.halo_unew_bottom[k].allocate(ld, CACHELINE_BYTES);
            dom.halo_uold_top[k].allocate(ld, CACHELINE_BYTES);
            dom.halo_uold_bottom[k].allocate(ld, CACHELINE_BYTES);
        }

        dom.unew_top_ready.value.store(tid == 0 ? 0 : -1,
                                       std::memory_order_relaxed);
        dom.unew_bottom_ready.value.store(tid == num_threads - 1 ? 0 : -1,
                                          std::memory_order_relaxed);
        dom.uold_top_ready.value.store(tid == 0 ? 0 : -1,
                                       std::memory_order_relaxed);
        dom.uold_bottom_ready.value.store(tid == num_threads - 1 ? 0 : -1,
                                          std::memory_order_relaxed);

        for (int ig = start_g; ig <= end_g; ++ig) {
            owner_of_row[static_cast<std::size_t>(ig)] = tid;
        }
    }

    const double D  = 100.0;
    const double x0 = 0.5;
    const double y0 = 0.5;

    #pragma omp parallel num_threads(num_threads) shared(doms, N, h, D, x0, y0)
    {
        const int tid = omp_get_thread_num();
        ThreadDomain& dom = doms[tid];
        const std::size_t ld_local = dom.ld;

        for (int il = 0; il <= dom.local_n + 1; ++il) {
            std::fill_n(dom.u_new.data() + static_cast<std::size_t>(il) * ld_local,
                        ld_local, 0.0);
            std::fill_n(dom.u_old.data() + static_cast<std::size_t>(il) * ld_local,
                        ld_local, 0.0);
        }

        for (int k = 0; k < 2; ++k) {
            std::fill_n(dom.halo_unew_top[k].data(), ld_local, 0.0);
            std::fill_n(dom.halo_unew_bottom[k].data(), ld_local, 0.0);
            std::fill_n(dom.halo_uold_top[k].data(), ld_local, 0.0);
            std::fill_n(dom.halo_uold_bottom[k].data(), ld_local, 0.0);
        }

        for (int il = 1; il <= dom.local_n; ++il) {
            const int ig = dom.start_g + (il - 1);
            const double x = ig * h;

            double* row_new = dom.u_new.data() + L(il, 0, ld_local);
            double* row_old = dom.u_old.data() + L(il, 0, ld_local);

            row_new[0]   = 0.0;
            row_new[N-1] = 0.0;
            row_old[0]   = 0.0;
            row_old[N-1] = 0.0;

            for (int j = 1; j <= N - 2; ++j) {
                const double y = j * h;
                row_new[j] = std::exp(-D * ((x - x0) * (x - x0) +
                                            (y - y0) * (y - y0)));
                row_old[j] = 0.0;
            }
        }

        if (tid == 0) {
            std::fill_n(dom.u_new.data() + L(0, 0, ld_local), ld_local, 0.0);
            std::fill_n(dom.u_old.data() + L(0, 0, ld_local), ld_local, 0.0);
        }
        if (tid == num_threads - 1) {
            std::fill_n(dom.u_new.data() + L(dom.local_n + 1, 0, ld_local),
                        ld_local, 0.0);
            std::fill_n(dom.u_old.data() + L(dom.local_n + 1, 0, ld_local),
                        ld_local, 0.0);
        }
    }

    for (int tid = 0; tid < num_threads; ++tid) {
        publish_unew(doms.get(), tid, num_threads, 0);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(num_threads) shared(doms, num_threads, N, T, TILE, lam, denom)
    {
        const int tid = omp_get_thread_num();
        ThreadDomain& dom = doms[tid];

        for (int step = 0; step < T; ++step) {
            const int m0 = 2 * step;
            const int m1 = 2 * step + 1;

            const int epoch_unew_start = 4 * step;
            const int epoch_uold_p1    = 4 * step + 1;
            const int epoch_uold_p2    = 4 * step + 2;
            const int epoch_unew_p3    = 4 * step + 3;
            const int epoch_unew_end   = 4 * step + 4;

            ensure_unew_halos(dom, tid, num_threads, epoch_unew_start);
            phase1_full(dom, N, TILE, lam, m0);
            publish_uold(doms.get(), tid, num_threads, epoch_uold_p1);

            ensure_uold_halos(dom, tid, num_threads, epoch_uold_p1);
            phase2_full(dom, N, TILE, lam, denom, m0);
            publish_uold(doms.get(), tid, num_threads, epoch_uold_p2);

            ensure_uold_halos(dom, tid, num_threads, epoch_uold_p2);
            phase3_full(dom, N, TILE, lam, m1);
            publish_unew(doms.get(), tid, num_threads, epoch_unew_p3);

            ensure_unew_halos(dom, tid, num_threads, epoch_unew_p3);
            phase4_full(dom, N, TILE, lam, denom, m1);
            publish_unew(doms.get(), tid, num_threads, epoch_unew_end);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Tempo : " << secs << " s\n";

    std::ofstream fout("output.txt");
    if (!fout) {
        std::cerr << "Erro ao abrir output.txt para escrita.\n";
        return 1;
    }

    fout.setf(std::ios::fixed);
    fout.precision(8);

    for (int i = 0; i < N; i += 16) {
        const double x = i * h;
        for (int j = 0; j < N; j += 16) {
            const double y = j * h;
            double v = 0.0;

            if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
                const int tid = owner_of_row[static_cast<std::size_t>(i)];
                if (tid >= 0) {
                    const ThreadDomain& dom = doms[tid];
                    const int il = i - dom.start_g + 1;
                    v = dom.u_new[L(il, j, dom.ld)];
                }
            }

            fout << x << " " << y << " " << v << "\n";
        }
    }

    return 0;
}