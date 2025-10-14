#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH -p cpu_amd
#SBATCH -J Hopscotch
#SBATCH --time=00:30:00
#SBATCH --account=superpd
#SBATCH --output=slurm-%j.out

# Não usar -e para não abortar o loop se uma rodada falhar
set -uo pipefail

EXEC="${1:?informe o executável como 1º argumento}"
shift || true
EXEC_ARGS=("$@")

cd "$SLURM_SUBMIT_DIR"

NT_MAX="${SLURM_CPUS_PER_TASK:-1}"
if (( NT_MAX < 1 )); then
  echo "ERRO: SLURM_CPUS_PER_TASK inválido: ${SLURM_CPUS_PER_TASK:-?}" >&2
  exit 1
fi
if [[ ! -x "$EXEC" ]]; then
  echo "ERRO: executável não encontrado/sem permissão: $EXEC" >&2
  exit 127
fi

# Parâmetros de repetição
RODADAS=4       # total por nt
DESCARTAR=1     # descarta as primeiras N (aquecimento); aqui 1 → média das 3 restantes

echo "Executável: $EXEC"
echo "CPUs por tarefa alocadas: $NT_MAX"
[[ ${#EXEC_ARGS[@]} -gt 0 ]] && echo "Args: ${EXEC_ARGS[*]}"

RESULTS="results_${SLURM_JOB_ID}.csv"
RESULTS_RAW="results_${SLURM_JOB_ID}_raw.csv"
echo "program,threads,elapsed_seconds,exit_code" > "$RESULTS"
echo "program,threads,run_index,elapsed_seconds,exit_code,is_warmup" > "$RESULTS_RAW"

# Afinidade básica
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Função para média (usa somente linhas numéricas)
avg_file() {
  awk '
    $1 ~ /^[0-9]+([.][0-9]+)?$/ { s+=$1; n++ }
    END { if(n>0) printf "%.3f\n", s/n; else print "NaN" }
  ' "$1"
}

for nt in $(seq 1 "$NT_MAX"); do
  echo "=== nt=${nt} ==="
  export OMP_NUM_THREADS="$nt"

  # Onde acumular tempos e códigos desta configuração de nt
  tmp_list="times_nt${nt}.lst"
  : > "$tmp_list"

  rc_final=0

  for r in $(seq 1 "$RODADAS"); do
    time_file="time_nt${nt}_r${r}.txt"
    err_log="stderr_nt${nt}_r${r}.log"

    # 'time' embutido do bash: mede o 'real' em segundos com casas decimais
    TIMEFORMAT=%3R
    { time srun -n 1 -c "$nt" --cpu-bind=cores \
        "$EXEC" "${EXEC_ARGS[@]}" \
        >/dev/null 2>"$err_log"; } 2> "$time_file"
    rc=$?

    # Captura tempo
    if [[ -s "$time_file" ]]; then
      elapsed=$(<"$time_file")
    else
      elapsed="NaN"
    fi

    is_warmup=$([[ $r -le $DESCARTAR ]] && echo 1 || echo 0)
    echo "$EXEC,$nt,$r,$elapsed,$rc,$is_warmup" >> "$RESULTS_RAW"

    # Guarda apenas as rodadas após o aquecimento
    if (( r > DESCARTAR )); then
      echo "$elapsed $rc" >> "$tmp_list"
      if (( rc_final == 0 && rc != 0 )); then
        rc_final=$rc
      fi
    fi

    # Opcional: alerta se falhar
    if (( rc != 0 )); then
      echo "Aviso: nt=$nt rodada=$r saiu com rc=$rc"; tail -n 3 "$err_log" || true
    fi
  done

  # Média das rodadas úteis
  avg_elapsed="$(avg_file "$tmp_list")"

  # Se alguma das 3 úteis falhou, rc_final já será ≠ 0
  echo "$EXEC,$nt,$avg_elapsed,$rc_final" | tee -a "$RESULTS" >/dev/null
done

echo "Concluído."
echo "CSV final:   $RESULTS"
echo "CSV detalhado (inclui aquecimento): $RESULTS_RAW"

