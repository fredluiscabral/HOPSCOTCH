// hopscotch2d_omp_naive.cpp
// Equação do calor 2D (Hopscotch) — versão OpenMP "naive"
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
#include <vector>
#include <chrono>
#include <cctype>
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

    // Todos os parâmetros são obrigatórios
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

    // Validações rígidas (sem ajustes silenciosos)
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

int main() {
    // ---- Leitura de parâmetros (estrita) ----
    int N = 0, T = 0, TILE = 0;
    double alpha = 0.0;
    if (!load_params_strict("param.txt", N, alpha, T, TILE)) {
        std::cerr << "Erro: não foi possível ler param.txt\n";
        return 1; // aborta como desejado
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
    // Contorno Dirichlet 0 — define uma única vez
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

    // ---- Integração temporal (Hopscotch: 4 “laços” por passo) ----
    int m = 0; // paridade do tabuleiro

    auto t0 = std::chrono::high_resolution_clock::now();

    // Uma única região paralela; cada fase usa omp for (com barreira implícita)
    #pragma omp parallel default(none) shared(N,T,TILE,lam,denom,U_new,U_old,m,idx)
    {
        for (int step = 0; step < T; ++step) {
            // Loop 1 (explícito) — escreve U_old a partir de U_new nos pontos (i+j+m) par
            #pragma omp for collapse(2) schedule(static)
            for (int ii = 1; ii <= N-2; ii += TILE) {
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int i_end = std::min(N-2, ii + TILE - 1);
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if ( ((i + j + m) & 1) == 0 ) {
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

            // Loop 2 (semi-implícito) — escreve U_old nos pontos (i+j+m) ímpar
            #pragma omp for collapse(2) schedule(static)
            for (int ii = 1; ii <= N-2; ii += TILE) {
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int i_end = std::min(N-2, ii + TILE - 1);
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if ( ((i + j + m) & 1) == 1 ) {
                                U_old[idx(i,j)] =
                                    ( U_new[idx(i,j)]
                                    + lam*( U_old[idx(i+1,j)] + U_old[idx(i-1,j)]
                                          + U_old[idx(i,j+1)] + U_old[idx(i,j-1)] ) ) / denom;
                            }
                        }
                    }
                }
            }

            // Atualiza paridade (uma única thread, barreira implícita ao final do single)
            #pragma omp single
            { m++; }

            // Loop 3 (explícito) — escreve U_new a partir de U_old nos pontos (i+j+m) par
            #pragma omp for collapse(2) schedule(static)
            for (int ii = 1; ii <= N-2; ii += TILE) {
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int i_end = std::min(N-2, ii + TILE - 1);
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if ( ((i + j + m) & 1) == 0 ) {
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

            // Loop 4 (semi-implícito) — escreve U_new nos pontos (i+j+m) ímpar
            #pragma omp for collapse(2) schedule(static)
            for (int ii = 1; ii <= N-2; ii += TILE) {
                for (int jj = 1; jj <= N-2; jj += TILE) {
                    const int i_end = std::min(N-2, ii + TILE - 1);
                    const int j_end = std::min(N-2, jj + TILE - 1);
                    for (int i = ii; i <= i_end; ++i) {
                        for (int j = jj; j <= j_end; ++j) {
                            if ( ((i + j + m) & 1) == 1 ) {
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
            { m++; } // completa o passo
        }
    } // fim região paralela

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - std::chrono::high_resolution_clock::now() + (t1 - t1)).count(); // dummy to avoid warnings
    secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Tempo : " << secs << " s\n";

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

    std::cout << "Concluído. Resultado escrito em output.txt\n";
    return 0;
}
