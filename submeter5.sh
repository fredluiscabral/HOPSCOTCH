#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH -p cpu_amd
#SBATCH -J Hopscotch
#SBATCH --time=24:00:00
#SBATCH --account=superpd
#SBATCH --output=slurm-%j.out


set -uo pipefail   # sem -e para não abortar em caso de falha numa rodada

EXEC="${1:?informe o executável como 1º argumento}"
shift || true
EXEC_ARGS=("$@")

cd "$SLURM_SUBMIT_DIR"
NT_MAX="${SLURM_CPUS_PER_TASK:-1}"

# Parâmetros simples
RODADAS=4
DESCARTAR=1
TOKEN="${TOKEN:-TIME_S=}"   # prefixo que seu programa imprime (ex.: "TIME_S=")

RESULTS="results_${SLURM_JOB_ID}.csv"
echo "program,threads,elapsed_seconds,exit_code" > "$RESULTS"

export OMP_PROC_BIND=close
export OMP_PLACES=cores

# média simples
avg_file() { awk '{s+=$1;n++} END{if(n) printf "%.6f\n", s/n; else print "NaN"}' "$1"; }

for nt in $(seq 1 "$NT_MAX"); do
  export OMP_NUM_THREADS="$nt"
  tmp="times_nt${nt}.lst"; : > "$tmp"
  rc_final=0

  for r in $(seq 1 "$RODADAS"); do
    # roda e captura stdout na variável (bem simples)
    out="$(srun -n 1 -c "$nt" --cpu-bind=cores "$EXEC" "${EXEC_ARGS[@]}")"
    rc=$?

    # pega o número imediatamente após TOKEN (ex.: TIME_S=0.123)
    elapsed="$(awk -v t="$TOKEN" '
      index($0,t){ sub(".*"t,""); if (match($0,/[0-9]+(\.[0-9]+)?/)) { print substr($0,RSTART,RLENGTH); exit } }
    ' <<< "$out")"

    [[ -n "$elapsed" ]] || elapsed="NaN"

    # acumula só após aquecimento
    if (( r > DESCARTAR )); then
      echo "$elapsed" >> "$tmp"
      (( rc_final == 0 && rc != 0 )) && rc_final=$rc
    fi
  done

  avg="$(avg_file "$tmp")"
  echo "$EXEC,$nt,$avg,$rc_final" >> "$RESULTS"
done

echo "Concluído. CSV: $RESULTS"

