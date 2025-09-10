// hopscotch2d_omp_semaphores.cpp
// Equação do calor 2D (Hopscotch) — OpenMP + semáforos POSIX (sincronização local)
// Lê parâmetros de param.txt (obrigatório) e mede o tempo do laço principal.
// Compilar: g++ -std=c++17 -O3 -fopenmp -pthread hopscotch2d_omp_semaphores.cpp -o hopscotch2d_omp_semaphores -lm
// Observação: semáforos POSIX não têm barreiras implícitas; os handshakes são estritamente locais.

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
#include <omp.h>

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

    if (N < 3)          { std::cerr << "Erro: N >= 3.\n"; return false; }
    if (alpha <= 0.0)   { std::cerr << "Erro: alpha > 0.\n"; return false; }
    if (T < 0)          { std::cerr << "Erro: T >= 0.\n"; return false; }
    if (TILE < 1 || TILE > N-2) {
        std::cerr << "Erro: tile deve satisfazer 1 <= tile <= N-2.\n";
        return false;
    }
    return true;
}

int main() {
    // ---- Leitura de parâmetros ----
    int N = 0, T = 0, TILE = 0;
    double alpha = 0.0;
    if (!load_params_strict("param.txt", N, alpha, T, TILE)) {
        return 1; // aborta se não ler param.txt
    }

    // ---- Discretização ----
    const double h   = 1.0 / (N - 1);
    // Estável para o explícito 2D: dt <= h^2 / (4*alpha). Usamos 90% do limite.
    const double dt  = 0.90 * (h*h) / (4.0 * alpha);
    const double lam = alpha * dt / (h*h);
    const double denom = 1.0 + 4.0 * lam;

    // ---- Alocação ----
    const size_t NN = static_cast<size_t>(N) * static_cast<size_t>(N);
    std::vector<double> U_new(NN, 0.0), U_old(NN, 0.0);
    auto idx = [N](int i, int j) -> size_t { return static_cast<size_t>(i)*N + j; };

    // ---- Condição inicial: pulso gaussiano no centro (apenas interior) ----
    const double D  = 100.0;
    const double x0 = 0.5;
    const double y0 = 0.5;
    for (int i = 1; i <= N-2; ++i) {
        const double x = i * h;
        for (int j = 1; j <= N-2; ++j) {
            const double y = j * h;
            U_new[idx(i,j)] = std::exp(-D*((x - x0)*(x - x0) + (y - y0)*(y - y0)));
        }
    }
    // Contorno Dirichlet 0 — define uma única vez (bordas já estão em 0)
    for (int i = 0; i < N; ++i) {
        U_new[idx(i,0)]   = 0.0; U_new[idx(i,N-1)] = 0.0;
        U_new[idx(0,i)]   = 0.0; U_new[idx(N-1,i)] = 0.0;
        U_old[idx(i,0)]   = 0.0; U_old[idx(i,N-1)] = 0.0;
        U_old[idx(0,i)]   = 0.0; U_old[idx(N-1,i)] = 0.0;
    }

    // ---- Semáforos e paralelismo ----
    double t0 = omp_get_wtime();

    // Variáveis compartilhadas configuradas dentro da região paralela
    int NT = 0;                      // nº de threads reais
    std::vector<sem_t> sem_left, sem_right;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        // Descobre nº de threads e inicializa semáforos (apenas uma thread)
        #pragma omp single
        {
            NT = omp_get_num_threads();
            sem_left.resize(NT);
            sem_right.resize(NT);
            for (int t = 0; t < NT; ++t) {
                sem_init(&sem_left[t],  0, 0);
                sem_init(&sem_right[t], 0, 0);
            }
        }
        #pragma omp barrier  // garante semáforos prontos

        // Particionamento 1D por faixas contíguas (somente região interna 1..N-2)
        const int H = (N-2);
        const int base = H / NT;
        const int sobra = H % NT;

        int iLocal = 1 + tid*base + std::min(tid, sobra);
        int fLocal = iLocal + base - 1;
        if (tid < sobra) fLocal += 1;

        // Lambdas de sincronização local
        auto signal_done = [&](int t){
            sem_post(&sem_left[t]);
            sem_post(&sem_right[t]);
        };
        auto wait_for_neighbors = [&](int t){
            if (t > 0)     sem_wait(&sem_right[t-1]);  // espera vizinho da esquerda
            if (t < NT-1)  sem_wait(&sem_left[t+1]);   // espera vizinho da direita
        };

        // Índice de paridade global (compartilhado), mas cada thread usa uma cópia local por passo
        static int m_shared = 0;

        // Laço no tempo (cada passo tem 4 fases; handshakes locais por fase)
        for (int step = 0; step < T; ++step) {

            int m_local = m_shared; // cópia local coerente no início do passo (há barreira no fim do passo anterior)

            // ---------------- FASE 1 (explícita): escreve U_old a partir de U_new nos pontos (i+j+m_local) pares
            {
                signal_done(tid); // “token” de conclusão lógica anterior (para ordenamento consistente)
                // computação
                for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                    const int i_end = std::min(fLocal, ii + TILE - 1);
                    for (int jj = 1; jj <= N-2; jj += TILE) {
                        const int j_end = std::min(N-2, jj + TILE - 1);
                        for (int i = ii; i <= i_end; ++i) {
                            for (int j = jj; j <= j_end; ++j) {
                                if ( ((i + j + m_local) & 1) == 0 ) {
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
                // handshake local com vizinhos (assegura fronteiras consistentes para Fase 2)
                signal_done(tid);
                wait_for_neighbors(tid);
            }

            // ---------------- FASE 2 (semi-implícita): escreve U_old nos pontos (i+j+m_local) ímpares
            {
                signal_done(tid);
                for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                    const int i_end = std::min(fLocal, ii + TILE - 1);
                    for (int jj = 1; jj <= N-2; jj += TILE) {
                        const int j_end = std::min(N-2, jj + TILE - 1);
                        for (int i = ii; i <= i_end; ++i) {
                            for (int j = jj; j <= j_end; ++j) {
                                if ( ((i + j + m_local) & 1) == 1 ) {
                                    U_old[idx(i,j)] =
                                        ( U_new[idx(i,j)]
                                        + lam*( U_old[idx(i+1,j)] + U_old[idx(i-1,j)]
                                              + U_old[idx(i,j+1)] + U_old[idx(i,j-1)] ) ) / denom;
                                }
                            }
                        }
                    }
                }
                signal_done(tid);
                wait_for_neighbors(tid);
                m_local++; // alterna paridade local para Fases 3/4
            }

            // ---------------- FASE 3 (explícita): escreve U_new a partir de U_old nos pontos (i+j+m_local) pares
            {
                signal_done(tid);
                for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                    const int i_end = std::min(fLocal, ii + TILE - 1);
                    for (int jj = 1; jj <= N-2; jj += TILE) {
                        const int j_end = std::min(N-2, jj + TILE - 1);
                        for (int i = ii; i <= i_end; ++i) {
                            for (int j = jj; j <= j_end; ++j) {
                                if ( ((i + j + m_local) & 1) == 0 ) {
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
                signal_done(tid);
                wait_for_neighbors(tid);
            }

            // ---------------- FASE 4 (semi-implícita): escreve U_new nos pontos (i+j+m_local) ímpares
            {
                signal_done(tid);
                for (int ii = iLocal; ii <= fLocal; ii += TILE) {
                    const int i_end = std::min(fLocal, ii + TILE - 1);
                    for (int jj = 1; jj <= N-2; jj += TILE) {
                        const int j_end = std::min(N-2, jj + TILE - 1);
                        for (int i = ii; i <= i_end; ++i) {
                            for (int j = jj; j <= j_end; ++j) {
                                if ( ((i + j + m_local) & 1) == 1 ) {
                                    U_new[idx(i,j)] =
                                        ( U_old[idx(i,j)]
                                        + lam*( U_new[idx(i+1,j)] + U_new[idx(i-1,j)]
                                              + U_new[idx(i,j+1)] + U_new[idx(i,j-1)] ) ) / denom;
                                }
                            }
                        }
                    }
                }
                signal_done(tid);
                wait_for_neighbors(tid);
                m_local++; // completa o passo
            }

            // --- Barreira global por passo (mantém índice temporal alinhado entre threads)
            #pragma omp barrier
            #pragma omp single
            {
                m_shared += 2; // avança paridade global para o próximo passo
            }
            #pragma omp barrier
        } // fim do laço no tempo

        // Destrói semáforos (apenas uma thread)
        #pragma omp single
        {
            for (int t = 0; t < NT; ++t) {
                sem_destroy(&sem_left[t]);
                sem_destroy(&sem_right[t]);
            }
        }
    } // fim região paralela

    double t1 = omp_get_wtime();
    std::cout << "Tempo : " << (t1 - t0) << " s\n";

    // ---- Saída amostrada ----
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

    std::cout << "Concluído. Resultado escrito em output.txt\n";
    return 0;
}
