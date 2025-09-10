#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Configurações
# ----------------------------
# Binários OpenMP que serão testados (adicione/remova conforme necessário)
BINS=(
  "./hopscotch2d_omp_naive"
  "./hopscotch2d_omp_sem_nobarrier"
  "./hopscotch2d_omp_busywait_barrier"
  "./hopscotch2d_omp_busywait_nobarrier"
)

# Faixa de threads
TMIN=1
TMAX=4

# Arquivo de parâmetros obrigatório
PARAM_FILE="param.txt"

# Afinidade (opcional, ajustável)
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export LC_ALL=C

# Diretorio de resultados (timestamp)
STAMP=$(date +"%Y%m%d_%H%M%S")
OUTDIR="results_${STAMP}"
mkdir -p "$OUTDIR"

RAW_CSV="${OUTDIR}/raw_times.csv"
MEAN_CSV="${OUTDIR}/mean_times.csv"

echo "exe,threads,run_idx,time_sec" > "$RAW_CSV"
echo "exe,threads,avg_time_sec"     > "$MEAN_CSV"

# ----------------------------
# Checagens básicas
# ----------------------------
if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "ERRO: ${PARAM_FILE} não encontrado. O programa abortará se o arquivo não existir."
  exit 1
fi

# ----------------------------
# Função para extrair tempo do stdout
# Busca a linha que contém 'Tempo' e pega o último número (em segundos)
# ----------------------------
extract_time() {
  # stdin: stdout do programa
  # stdout: número (segundos)
  local t
  t=$(grep -i "Tempo" | tail -n1 | grep -Eo '[0-9]+([.][0-9]+)?' | tail -n1 || true)
  if [[ -z "${t:-}" ]]; then
    # fallback: tenta pegar último número do stdout todo
    t=$(grep -Eo '[0-9]+([.][0-9]+)?' | tail -n1 || true)
  fi
  echo "${t:-NaN}"
}

# ----------------------------
# Loop principal
# ----------------------------
for exe in "${BINS[@]}"; do
  if [[ ! -x "$exe" ]]; then
    echo "Aviso: binário '$exe' não encontrado ou sem permissão de execução — pulando."
    continue
  fi

  echo ">>> Executando ${exe}"

  for th in $(seq "${TMIN}" "${TMAX}"); do
    export OMP_NUM_THREADS="${th}"
    echo "  - Threads: ${th}"

    times=()
    for run in {1..6}; do
      # Executa e captura stdout
      out="$("$exe" 2>&1 || true)"
      # Extrai tempo
      t=$(printf "%s\n" "$out" | extract_time)

      if [[ "$t" == "NaN" ]]; then
        echo "    [run ${run}] Falha ao extrair tempo. Saída:"
        echo "$out"
        exit 1
      fi

      times+=("$t")
      echo "${exe},${th},${run},${t}" >> "$RAW_CSV"

      # Mostra o tempo da run
      echo "    [run ${run}] tempo = ${t} s"
    done

    # Descarta a primeira execução e calcula média das 5 restantes
    avg=$(printf "%s\n" "${times[@]:1}" | awk '{s+=$1} END {printf "%.6f", s/NR}')
    echo "${exe},${th},${avg}" >> "$MEAN_CSV"
    echo "    -> média (descartando 1ª) = ${avg} s"
  done
done

echo
echo "OK! Resultados:"
echo " - Detalhado: ${RAW_CSV}"
echo " - Médias:    ${MEAN_CSV}"
