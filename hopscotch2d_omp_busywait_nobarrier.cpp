// hopscotch2d_omp_busywait_nobarrier.cpp
// Equação do calor 2D (Hopscotch) — OpenMP com divisão em faixas e espera ocupada local,
// SEM barreira global por passo (toda a coordenação é local vizinho-a-vizinho).
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

#include <omp.h>

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
        std::cerr << "Erro: 1<=TILE<=N-2.\n"; return false;
    }
    return true;
}

int main() {
    // ---- Leitura de parâmetros ----
    int N=0, T=0, TILE=0; double alpha=0.0;
    if (!load_params_strict("param.txt", N, alpha, T, TILE)) return 1;

    // ---- Discretização ----
    const double h   = 1.0 / (N - 1);
    const double dt  = 0.90 * (h*h) / (4.0 * alpha);   // 90% do limite estável explícito 2D
    const double lam = alpha * dt / (h*h);
    const double denom = 1.0 + 4.0 * lam;

    // ---- Alocação ----
    const size_t NN = static_cast<size_t>(N) * static_cast<size_t>(N);
    std::vector<double> U_new(NN, 0.0), U_old(NN, 0.0);
    auto idx = [N](int i, int j)->size_t { return static_cast<size_t>(i)*N + j; };

    // ---- Condição inicial (apenas interior) + contorno 0 uma vez ----
    const double D=100.0, x0=0.5, y0=0.5;
    for (int i = 1; i <= N-2; ++i) {
        const double x = i*h;
        for (int j = 1; j <= N-2; ++j) {
            const double y = j*h;
            U_new[idx(i,j)] = std::exp(-D*((x-x0)*(x-x0)+(y-y0)*(y-y0)));
        }
    }
    for (int i=0;i<N;++i){
        U_new[idx(i,0)]=U_new[idx(i,N-1)]=0.0;
        U_new[idx(0,i)]=U_new[idx(N-1,i)]=0.0;
        U_old[idx(i,0)]=U_old[idx(i,N-1)]=0.0;
        U_old[idx(0,i)]=U_old[idx(N-1,i)]=0.0;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Paraleliza por faixas (linhas) com sincronização local SOMENTE (sem barreiras globais)
    #pragma omp parallel default(shared)
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();

        // Divisão de [1..N-2] em nt faixas contíguas
        const int H = (N-2)/nt;
        const int R = (N-2)%nt;
        int iLocal = 1 + tid*H + std::min(tid, R);
        int fLocal = iLocal + H - 1 + (tid < R ? 1 : 0);

        // Array de progresso compartilhado (uma vez)
        static std::unique_ptr<std::atomic<int>[]> progress;
        #pragma omp single
        {
            progress.reset(new std::atomic<int>[nt]);
            for (int p=0; p<nt; ++p) progress[p].store(0, std::memory_order_relaxed);
        }
        #pragma omp barrier // publica 'progress' para todas as threads

        auto wait_neighbors_atleast = [&](int expected){
            if (tid>0)     { while (progress[tid-1].load(std::memory_order_acquire) < expected) spin_pause(); }
            if (tid<nt-1)  { while (progress[tid+1].load(std::memory_order_acquire) < expected) spin_pause(); }
        };

        int m_local = 0; // paridade

        for (int step=0; step<T; ++step) {
            // *** A ausência de barreira global é compensada por "marcos" (progress) por fase ***

            // ====== FASE 1 (explícita) ======
            // Garante que vizinhos terminaram o passo anterior (valor alvo: 4*step)
            wait_neighbors_atleast(4*step + 0);
            for (int ii=iLocal; ii<=fLocal; ii+=TILE){
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj=1; jj<=N-2; jj+=TILE){
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i=ii; i<=i_end; ++i){
                        for (int j=jj; j<=j_end; ++j){
                            if (((i+j+m_local)&1)==0){
                                U_old[idx(i,j)] = U_new[idx(i,j)] +
                                  lam*( U_new[idx(i+1,j)] + U_new[idx(i-1,j)]
                                      + U_new[idx(i,j+1)] + U_new[idx(i,j-1)]
                                      - 4.0*U_new[idx(i,j)] );
                            } else {
                                U_old[idx(i,j)] = U_new[idx(i,j)];
                            }
                        }
                    }
                }
            }
            progress[tid].store(4*step + 1, std::memory_order_release);

            // ====== FASE 2 (semi-implícita) ======
            wait_neighbors_atleast(4*step + 1);
            for (int ii=iLocal; ii<=fLocal; ii+=TILE){
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj=1; jj<=N-2; jj+=TILE){
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i=ii; i<=i_end; ++i){
                        for (int j=jj; j<=j_end; ++j){
                            if (((i+j+m_local)&1)==1){
                                U_old[idx(i,j)] = ( U_new[idx(i,j)]
                                  + lam*( U_old[idx(i+1,j)] + U_old[idx(i-1,j)]
                                        + U_old[idx(i,j+1)] + U_old[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            progress[tid].store(4*step + 2, std::memory_order_release);
            m_local++; // alterna paridade após Fase 2

            // ====== FASE 3 (explícita) ======
            wait_neighbors_atleast(4*step + 2);
            for (int ii=iLocal; ii<=fLocal; ii+=TILE){
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj=1; jj<=N-2; jj+=TILE){
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i=ii; i<=i_end; ++i){
                        for (int j=jj; j<=j_end; ++j){
                            if (((i+j+m_local)&1)==0){
                                U_new[idx(i,j)] = U_old[idx(i,j)] +
                                  lam*( U_old[idx(i+1,j)] + U_old[idx(i-1,j)]
                                      + U_old[idx(i,j+1)] + U_old[idx(i,j-1)]
                                      - 4.0*U_old[idx(i,j)] );
                            } else {
                                U_new[idx(i,j)] = U_old[idx(i,j)];
                            }
                        }
                    }
                }
            }
            progress[tid].store(4*step + 3, std::memory_order_release);

            // ====== FASE 4 (semi-implícita) ======
            wait_neighbors_atleast(4*step + 3);
            for (int ii=iLocal; ii<=fLocal; ii+=TILE){
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj=1; jj<=N-2; jj+=TILE){
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i=ii; i<=i_end; ++i){
                        for (int j=jj; j<=j_end; ++j){
                            if (((i+j+m_local)&1)==1){
                                U_new[idx(i,j)] = ( U_old[idx(i,j)]
                                  + lam*( U_new[idx(i+1,j)] + U_new[idx(i-1,j)]
                                        + U_new[idx(i,j+1)] + U_new[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            progress[tid].store(4*step + 4, std::memory_order_release);
            m_local++; // completa o passo

            // SEM barreira global aqui — o início do próximo passo fará:
            // wait_neighbors_atleast(4*(step+1) + 0) == wait de (4*step+4),
            // garantindo que vizinhos concluíram o passo atual.
        } // step
    } // parallel

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Tempo : " << secs << " s\n";

    // ---- Saída (amostrada) ----
    std::ofstream fout("output.txt");
    if (!fout) { std::cerr << "Erro ao abrir output.txt\n"; return 1; }
    fout.setf(std::ios::fixed); fout.precision(8);
    for (int i=0;i<N;i+=16){
        const double x=i*h;
        for (int j=0;j<N;j+=16){
            const double y=j*h;
            fout << x << " " << y << " " << U_new[idx(i,j)] << "\n";
        }
    }
    return 0;
}
