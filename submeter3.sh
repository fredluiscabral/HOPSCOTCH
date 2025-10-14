#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH -p cpu_amd_dev
#SBATCH -J Hopscotch
#SBATCH --time=00:30:00
#SBATCH --account=superpd
#SBATCH --output=slurm-%j.out

set -uo pipefail   # (sem -e para não abortar o loop se algum nt falhar)

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

echo "Executável: $EXEC"
echo "CPUs por tarefa alocadas: $NT_MAX"
[[ ${#EXEC_ARGS[@]} -gt 0 ]] && echo "Args: ${EXEC_ARGS[*]}"

RESULTS="results_${SLURM_JOB_ID}.csv"
echo "program,threads,elapsed_seconds,exit_code" > "$RESULTS"

# Afinidade básica
export OMP_PROC_BIND=close
export OMP_PLACES=cores

for nt in $(seq 1 "$NT_MAX"); do
  echo "=== nt=${nt} ==="
  export OMP_NUM_THREADS="$nt"

  # Onde salvar a medição e eventual stderr do programa
  time_file="time_nt${nt}.txt"
  err_log="stderr_nt${nt}.log"

  # 'time' embutido do bash: mede só o 'real' em segundos (com casas decimais)
  TIMEFORMAT=%3R
  { time srun -n 1 -c "$nt" --cpu-bind=cores \
      "$EXEC" "${EXEC_ARGS[@]}" \
      >/dev/null 2>"$err_log"; } 2> "$time_file"
  rc=$?

  # Captura do tempo (se não houver, marca NaN)
  if [[ -s "$time_file" ]]; then
    elapsed=$(cat "$time_file")
  else
    elapsed="NaN"
  fi

  echo "$EXEC,$nt,$elapsed,$rc" | tee -a "$RESULTS" >/dev/null

  # Opcional: mostre último erro se falhar
  if (( rc != 0 )); then
    echo "Aviso: nt=$nt saiu com rc=$rc"; tail -n 3 "$err_log" || true
  fi
done

echo "Concluído. CSV: $RESULTS"

