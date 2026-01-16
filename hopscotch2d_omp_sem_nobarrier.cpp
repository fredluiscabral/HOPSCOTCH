// hopscotch2d_omp_sema_nobarrier.cpp
// Hopscotch 2D (equação do calor) — OpenMP + semáforos POSIX
// Sincronização local por fase (vizinho-vizinho) e "rendezvous" local por passo (sem barreira global).
// Lê parâmetros obrigatórios de param.txt e mede o tempo do laço principal.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cctype>
#include <omp.h>
#include <semaphore.h>

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

int main() {
    // ---- Parâmetros ----
    int N=0, T=0, TILE=0;
    double alpha=0.0;
    if (!load_params_strict("param.txt", N, alpha, T, TILE)) {
        return 1; // aborta se não ler param.txt
    }

    // ---- Discretização ----
    const double h   = 1.0 / (N - 1);
    // Explícito 2D: dt <= h^2 / (4*alpha). Usamos 90% do limite.
    const double dt  = 0.90 * (h*h) / (4.0 * alpha);
    const double lam = alpha * dt / (h*h);
    const double denom = 1.0 + 4.0 * lam;

    // ---- Campos ----
    const size_t NN = static_cast<size_t>(N) * static_cast<size_t>(N);
    std::vector<double> U_new(NN, 0.0), U_old(NN, 0.0);
    auto idx = [N](int i, int j) -> size_t { return static_cast<size_t>(i)*N + j; };

    // ---- Condição inicial (interior) ----
    const double D  = 100.0, x0 = 0.5, y0 = 0.5;
    for (int i = 1; i <= N-2; ++i) {
        const double x = i * h;
        for (int j = 1; j <= N-2; ++j) {
            const double y = j * h;
            U_new[idx(i,j)] = std::exp(-D*((x-x0)*(x-x0) + (y-y0)*(y-y0)));
        }
    }
    // Contorno Dirichlet 0 (uma única vez)
    for (int i = 0; i < N; ++i) {
        U_new[idx(i,0)]   = 0.0; U_new[idx(i,N-1)] = 0.0;
        U_new[idx(0,i)]   = 0.0; U_new[idx(N-1,i)] = 0.0;
        U_old[idx(i,0)]   = 0.0; U_old[idx(i,N-1)] = 0.0;
        U_old[idx(0,i)]   = 0.0; U_old[idx(N-1,i)] = 0.0;
    }

    // ---- Semáforos (fase e passo) ----
    // Criamos vetores compartilhados; dimensionamos e inicializamos dentro da região paralela.
    std::vector<sem_t> sem_left, sem_right;       // sinalização por fase
    std::vector<sem_t> step_left, step_right;     // rendezvous por passo (sem barreira global)

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(none) shared(N,T,TILE,lam,denom,U_new,U_old,idx,sem_left,sem_right,step_left,step_right)
    {
        const int nt  = omp_get_num_threads();
        const int tid = omp_get_thread_num();

        // Inicialização dos semáforos (uma vez)
        #pragma omp single
        {
            sem_left.resize(nt);  sem_right.resize(nt);
            step_left.resize(nt); step_right.resize(nt);
            for (int t = 0; t < nt; ++t) {
                sem_init(&sem_left[t],  0, 0);
                sem_init(&sem_right[t], 0, 0);
                // início do bloco do passo (rótulo removido; era apenas decorativo)
                sem_init(&step_left[t],  0, 0);
                sem_init(&step_right[t], 0, 0);
            }
        }
        #pragma omp barrier

        // Particionamento em faixas contíguas (interior: linhas 1..N-2)
        const int interior = (N-2);
        const int baseSize = interior / nt;
        const int sobra    = interior % nt;
        int iLocal = 1 + tid * baseSize + std::min(tid, sobra);
        int mySize = baseSize + (tid < sobra ? 1 : 0);
        int fLocal = iLocal + mySize - 1;

        // Funções auxiliares (lambdas) para sincronização local por fase:
        auto signal_done_phase = [&](int t) {
            sem_post(&sem_left[t]);
            sem_post(&sem_right[t]);
        };
        auto wait_neighbors_phase = [&](int t) {
            if (t > 0)      sem_wait(&sem_right[t-1]); // vizinho da esquerda
            if (t < nt-1)   sem_wait(&sem_left[t+1]);  // vizinho da direita
        };

        // Rendezvous local por passo: anunciar avanço e checar vizinhos
        auto announce_step = [&](int t) {
            sem_post(&step_left[t]);
            sem_post(&step_right[t]);
        };
        auto wait_step_neighbors = [&](int t) {
            if (t > 0)      sem_wait(&step_right[t-1]); // aguarda esquerda avançar de passo
            if (t < nt-1)   sem_wait(&step_left[t+1]);  // aguarda direita avançar de passo
        };

        // Laço de passos SEM BARREIRA GLOBAL
        for (int s = 0; s < T; ++s) {
            // Antes de iniciar o passo s, garanta que os vizinhos também estão no passo s
            // (Para s==0 não é necessário; para s>0, aguarda anúncio do passo anterior)
            if (s > 0) {
                wait_step_neighbors(tid);
            }

            // Paridade "m" consistente sem variável global: m = 2*s no início do passo
            int mloc = 2*s;

            // ---- Fase 1 (explícita): escreve U_old a partir de U_new nos pontos (i+j+mloc) par
            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if ( ((i + j + mloc) & 1) == 0 ) {
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

            // ---- Fase 2 (semi-implícita): escreve U_old nos pontos (i+j+mloc) ímpar
            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if ( ((i + j + mloc) & 1) == 1 ) {
                                U_old[idx(i,j)] =
                                    ( U_new[idx(i,j)]
                                    + lam*( U_old[idx(i+1,j)] + U_old[idx(i-1,j)]
                                          + U_old[idx(i,j+1)] + U_old[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            signal_done_phase(tid);
            wait_neighbors_phase(tid);
            ++mloc; // alterna paridade dentro do passo

            // ---- Fase 3 (explícita): escreve U_new a partir de U_old nos pontos (i+j+mloc) par
            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if ( ((i + j + mloc) & 1) == 0 ) {
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

            // ---- Fase 4 (semi-implícita): escreve U_new nos pontos (i+j+mloc) ímpar
            for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                const int i_end = std::min(fLocal, ii + TILE - 1);
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if ( ((i + j + mloc) & 1) == 1 ) {
                                U_new[idx(i,j)] =
                                    ( U_old[idx(i,j)]
                                    + lam*( U_new[idx(i+1,j)] + U_new[idx(i-1,j)]
                                          + U_new[idx(i,j+1)] + U_new[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }
            signal_done_phase(tid);
            wait_neighbors_phase(tid);
            ++mloc; // completa o passo

            // Anunciar aos vizinhos que cheguei ao próximo passo (s+1)
            announce_step(tid);
            // (não há barreira global; o "aguardar vizinhos" acontece no início do próximo s)
        }

        // Destrói semáforos (uma vez)
        #pragma omp single
        {
            for (int t = 0; t < nt; ++t) {
                sem_destroy(&sem_left[t]);
                sem_destroy(&sem_right[t]);
                sem_destroy(&step_left[t]);
                sem_destroy(&step_right[t]);
            }
        }
    } // fim região paralela

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Threads: " << omp_get_num_threads(); << "Tempo : " << secs << " s\n";

    // ---- Saída (amostrada) ----
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
            fout << x << " " << y << " " << U_new[idx(i,j)] << "\n";
        }
    }
    fout.close();

    //std::cout << "Concluído. Resultado em output.txt\n";
    return 0;
}
