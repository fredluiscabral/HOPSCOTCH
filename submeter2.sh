#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192          # padrão (pode sobrescrever no sbatch)
#SBATCH -p cpu_amd
#SBATCH -J Hopscotch
#SBATCH --time=00:30:00
#SBATCH --account=superpd

set -euo pipefail

# -------- Parâmetros --------
# Uso:
#   sbatch --cpus-per-task=32 submeter.sh /caminho/para/exe [NT] [args...]
# Ex.: usa NT=32 do SLURM
#   sbatch --cpus-per-task=32 submeter.sh ./hopscotch2d_omp_naive
# Ex.: força NT=64 (atenção à alocação!)
#   sbatch --cpus-per-task=64 submeter.sh ./hopscotch2d_omp_naive 64

EXEC="${1:?informe o executável como 1º argumento}"
shift || true

# Se você passar um 2º argumento numérico, ele é o número de threads (NT).
NT="${SLURM_CPUS_PER_TASK:-1}"
if [[ $# -ge 1 && "${1:-}" =~ ^[0-9]+$ ]]; then
  NT="$1"
  shift
fi
EXEC_ARGS=("$@")   # argumentos do programa (opcional)

# -------- Checagens --------
if [[ ! -x "$EXEC" ]]; then
  echo "Erro: executável não encontrado/sem permissão: $EXEC" >&2
  exit 1
fi
if (( NT < 1 )); then
  echo "Erro: número de threads inválido: $NT" >&2
  exit 1
fi
# Evita oversubscription sem querer
if (( NT > ${SLURM_CPUS_PER_TASK:-0} )); then
  echo "Aviso: NT=$NT > SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-0}; usando NT=${SLURM_CPUS_PER_TASK:-0}." >&2
  NT="${SLURM_CPUS_PER_TASK:-0}"
fi

# -------- Informações --------
echo "Nó(s): $SLURM_JOB_NODELIST"
command -v nodeset &>/dev/null && nodeset -e "$SLURM_JOB_NODELIST" || true
echo "CPUs alocadas pelo SLURM: ${SLURM_CPUS_PER_TASK:-?}"
echo "Executável: $EXEC"
echo "Threads (NT): $NT"
[[ ${#EXEC_ARGS[@]} -gt 0 ]] && echo "Args do executável: ${EXEC_ARGS[*]}"

cd "$SLURM_SUBMIT_DIR"

# -------- OpenMP/Afinidade --------
export OMP_NUM_THREADS="$NT"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# -------- Execução --------
# -n 1: uma tarefa; -c NT: NT CPUs para essa tarefa
srun -n 1 -c "$NT" --cpu-bind=cores "$EXEC" "${EXEC_ARGS[@]}"

