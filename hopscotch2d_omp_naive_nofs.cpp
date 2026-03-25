// hopscotch2d_omp_naive_nofs.cpp
// Equação do calor 2D (Hopscotch) — versão OpenMP "naive" com mitigação de false sharing
// Lê parâmetros de param.txt (obrigatório) e mede o tempo do laço principal.
// Saída em output.txt: linhas "x y u" (amostras de 16 em 16 para reduzir o arquivo)

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
#include <memory>
#include <omp.h>

#ifndef CACHELINE_BYTES
#define CACHELINE_BYTES 64u
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
static inline std::string tolower_str(std::string s){
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    return s;
}

bool load_params_strict(const std::string& fname,
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
        if (!kv.count("n"))     { std::cerr << "Erro: parâmetro obrigatório 'n' ausente em " << fname << ".\n"; return false; }
        if (!kv.count("alpha")) { std::cerr << "Erro: parâmetro obrigatório 'alpha' ausente em " << fname << ".\n"; return false; }
        if (!kv.count("t"))     { std::cerr << "Erro: parâmetro obrigatório 't' ausente em " << fname << ".\n"; return false; }
        if (!kv.count("tile"))  { std::cerr << "Erro: parâmetro obrigatório 'tile' ausente em " << fname << ".\n"; return false; }

        N     = std::stoi(kv["n"]);
        alpha = std::stod(kv["alpha"]);
        T     = std::stoi(kv["t"]);
        TILE  = std::stoi(kv["tile"]);
    } catch (const std::exception& e) {
        std::cerr << "Erro: falha ao converter parâmetros de " << fname << ": " << e.what() << "\n";
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
    if (TILE < 1 || TILE > N-2) {
        std::cerr << "Erro: tile deve satisfazer 1 <= tile <= N-2 (obtido tile="
                  << TILE << ", N=" << N << ").\n";
        return false;
    }
    return true;
}

static inline std::size_t round_up(std::size_t n, std::size_t mult) {
    return ((n + mult - 1u) / mult) * mult;
}

template <typename T>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    ~AlignedBuffer() { reset(); }

    void allocate(std::size_t count, std::size_t alignment) {
        reset();
        if (count == 0) return;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, count * sizeof(T)) != 0 || ptr == nullptr) {
            throw std::bad_alloc();
        }
        data_ = static_cast<T*>(ptr);
        size_ = count;
    }

    void reset() {
        if (data_ != nullptr) {
            std::free(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

int main() {
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::printf("Threads: %d\n", omp_get_num_threads());
        }
    }

    int N = 0, T = 0, TILE = 0;
    double alpha = 0.0;
    if (!load_params_strict("param.txt", N, alpha, T, TILE)) {
        std::cerr << "Erro: não foi possível ler param.txt\n";
        return 1;
    }

    const double h   = 1.0 / (N - 1);
    const double dt  = 0.90 * (h*h) / (4.0 * alpha);
    const double lam = alpha * dt / (h*h);
    const double denom = 1.0 + 4.0 * lam;

    constexpr std::size_t kDoublesPerCacheLine = CACHELINE_BYTES / sizeof(double);
    const std::size_t ld = round_up(static_cast<std::size_t>(N), kDoublesPerCacheLine);
    const std::size_t NN = ld * static_cast<std::size_t>(N);

    AlignedBuffer<double> U_new, U_old;
    try {
        U_new.allocate(NN, CACHELINE_BYTES);
        U_old.allocate(NN, CACHELINE_BYTES);
    } catch (const std::bad_alloc&) {
        std::cerr << "Erro: falha de alocação para U_new/U_old.\n";
        return 1;
    }

    auto idx = [ld](int i, int j) -> std::size_t {
        return static_cast<std::size_t>(i) * ld + static_cast<std::size_t>(j);
    };

    const double D  = 100.0;
    const double x0 = 0.5;
    const double y0 = 0.5;

    // First-touch paralelo + zeragem inicial de todas as linhas padded
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double* row_new = U_new.data() + static_cast<std::size_t>(i) * ld;
        double* row_old = U_old.data() + static_cast<std::size_t>(i) * ld;
        std::fill_n(row_new, ld, 0.0);
        std::fill_n(row_old, ld, 0.0);
    }

    // Condição inicial apenas no interior
    #pragma omp parallel for schedule(static)
    for (int i = 1; i <= N-2; ++i) {
        const double x = i * h;
        for (int j = 1; j <= N-2; ++j) {
            const double y = j * h;
            U_new[idx(i,j)] = std::exp(-D*((x - x0)*(x - x0) + (y - y0)*(y - y0)));
        }
    }

    // Contorno Dirichlet 0 — define explicitamente
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        U_new[idx(i,0)]   = 0.0;
        U_new[idx(i,N-1)] = 0.0;
        U_new[idx(0,i)]   = 0.0;
        U_new[idx(N-1,i)] = 0.0;

        U_old[idx(i,0)]   = 0.0;
        U_old[idx(i,N-1)] = 0.0;
        U_old[idx(0,i)]   = 0.0;
        U_old[idx(N-1,i)] = 0.0;
    }

    int m = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(none) shared(N,T,TILE,lam,denom,U_new,U_old,m,idx)
    {
        for (int step = 0; step < T; ++step) {
            #pragma omp for collapse(2) schedule(static)
            for (int ii = 1; ii <= N-2; ii += TILE) {
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int i_end = std::min(N-2, ii + TILE - 1);
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + m) & 1) == 0) {
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

            #pragma omp for collapse(2) schedule(static)
            for (int ii = 1; ii <= N-2; ii += TILE) {
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int i_end = std::min(N-2, ii + TILE - 1);
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + m) & 1) == 1) {
                                U_old[idx(i,j)] =
                                    ( U_new[idx(i,j)]
                                    + lam*( U_old[idx(i+1,j)] + U_old[idx(i-1,j)]
                                          + U_old[idx(i,j+1)] + U_old[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }

            #pragma omp single
            { m++; }

            #pragma omp for collapse(2) schedule(static)
            for (int ii = 1; ii <= N-2; ii += TILE) {
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int i_end = std::min(N-2, ii + TILE - 1);
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + m) & 1) == 0) {
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

            #pragma omp for collapse(2) schedule(static)
            for (int ii = 1; ii <= N-2; ii += TILE) {
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int i_end = std::min(N-2, ii + TILE - 1);
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if (((i + j + m) & 1) == 1) {
                                U_new[idx(i,j)] =
                                    ( U_old[idx(i,j)]
                                    + lam*( U_new[idx(i+1,j)] + U_new[idx(i-1,j)]
                                          + U_new[idx(i,j+1)] + U_new[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }

            #pragma omp single
            { m++; }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Tempo : " << secs << " s\n";

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
