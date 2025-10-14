#!/usr/bin/env bash
set -euo pipefail

# Uso: ./submit_all.sh <num_threads>
if [[ $# -lt 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Uso: $0 <num_threads>"
  exit 1
fi

THREADS="$1"
CPUS_PER_TASK="$THREADS"   # ajuste aqui se quiser diferente de THREADS

# Bins a submeter (na mesma ordem da sua linha original)
APPS=(
  ./hopscotch2d_omp_naive
  ./hopscotch2d_omp_busywait_nobarrier
  ./hopscotch2d_omp_busywait_barrier
  ./hopscotch2d_omp_sem_nobarrier
  ./hopscotch2d_omp_semaphores
)

for app in "${APPS[@]}"; do
  if [[ ! -x "$app" ]]; then
    echo "Aviso: $app não existe ou não é executável — pulando."
    continue
  fi
  echo "Submetendo: $app  | cpus-per-task=$CPUS_PER_TASK | OMP_NUM_THREADS=$THREADS"
  sbatch --cpus-per-task="$CPUS_PER_TASK" \
         --export=ALL,OMP_NUM_THREADS="$THREADS" \
         submeter5.sh "$app"
done

