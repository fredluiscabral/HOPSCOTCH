#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH -p cpu_amd
#SBATCH -J HopscotchSweep
#SBATCH --time=00:30:00
#SBATCH --account=superpd
#SBATCH --output=slurm-%j.out

set -uo pipefail   # sem -e: não aborta o loop em falhas

EXEC="${1:?informe o executável como 1º argumento}"
shift || true
EXEC_ARGS=("$@")

cd "$SLURM_SUBMIT_DIR"

echo "Nó(s): ${SLURM_JOB_NODELIST:-?}"
command -v nodeset &>/dev/null && nodeset -e "${SLURM_JOB_NODELIST:-}" || true
echo "Executável: $EXEC"
echo "Diretório:  $PWD"

# ---------- Pré-flight no executável ----------
if [[ ! -e "$EXEC" ]]; then
  echo "ERRO: arquivo '$EXEC' não existe no diretório de submissão." >&2
  exit 127
fi
if [[ ! -x "$EXEC" ]]; then
  echo "ERRO: arquivo '$EXEC' não é executável (chmod +x)." >&2
  ls -l "$EXEC"
  exit 127
fi

echo ">> ls -l:"
ls -l "$EXEC" || true
echo ">> file:"
file "$EXEC" || true

if command -v ldd &>/dev/null; then
  echo ">> ldd:"
  ldd "$EXEC" || true
  # Se houver 'not found', avisa e aborta (isso causa rc=127 no run)
  MISSING=$(ldd "$EXEC" 2>/dev/null | awk '/not found/ {print $1}')
  if [[ -n "$MISSING" ]]; then
    echo "ERRO: bibliotecas faltando (ldd 'not found'):"
    echo "$MISSING" | sed 's/^/  - /'
    echo
    echo "Dicas:"
    echo "  * Carregue módulos que forneçam essas libs. Exemplos comuns:"
    echo "      module spider intel-oneapi  | module spider intel"
    echo "      module spider gcc           | module spider libgomp"
    echo "      module spider mkl           | module spider openmp"
    echo "  * OU linke estaticamente a runtime OpenMP (ex.: -qopenmp-link=static no icx/icc,"
    echo "    ou -static-libgcc -static-libstdc++ para g++ se suportado pelo cluster)."
    exit 127
  fi
fi

# ---------- Varredura ----------
NT_MAX="${SLURM_CPUS_PER_TASK:-1}"
if (( NT_MAX < 1 )); then
  echo "ERRO: SLURM_CPUS_PER_TASK inválido: ${SLURM_CPUS_PER_TASK:-?}" >&2
  exit 1
fi
echo "CPUs por tarefa alocadas: $NT_MAX"

LOGDIR="logs_${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"
RESULTS="results_${SLURM_JOB_ID}.csv"
echo "program,threads,elapsed_seconds,exit_code" > "$RESULTS"

export OMP_PROC_BIND=close
export OMP_PLACES=cores

for nt in $(seq 1 "$NT_MAX"); do
  echo "=== nt=${nt} ==="
  export OMP_NUM_THREADS="$nt"

  out_log="${LOGDIR}/stdout_nt${nt}.log"
  err_log="${LOGDIR}/stderr_nt${nt}.log"
  time_file="${LOGDIR}/time_nt${nt}.txt"

  /usr/bin/time -f "%e" -o "$time_file" \
    srun -n 1 -c "$nt" --cpu-bind=cores \
      "$EXEC" "${EXEC_ARGS[@]}" >"$out_log" 2>"$err_log"
  rc=$?

  elapsed="NaN"
  [[ -s "$time_file" ]] && elapsed="$(cat "$time_file")"
  echo "$EXEC,$nt,$elapsed,$rc" | tee -a "$RESULTS" >/dev/null

  if (( rc != 0 )); then
    echo "Aviso: nt=$nt saiu com rc=$rc. Últimas linhas do stderr:"
    tail -n 5 "$err_log" || true
  fi
done

echo "Concluído. CSV: $RESULTS  |  Logs: $LOGDIR/"

