import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIGURAÇÕES
# =========================

VERSIONS = {
    "NAIVE": "Naive",
    "BUSYWAIT": "Busy-wait",
    "SEMAPHORE": "Semáforos"
}

TILES = [1, 2, 4, 8, 16, 32]
N_RUNS = 6          # total de execuções
DISCARD = 1         # descartar a primeira

LOG_PATTERN = "{version}_TILE{tile}.log"

OUTPUT_DIR = Path("results_exp2")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# FUNÇÕES AUXILIARES
# =========================

def parse_log(filename):
    """
    Retorna dict: threads -> lista de tempos
    """
    data = {}
    with open(filename, "r") as f:
        lines = f.readlines()

    current_threads = None
    for line in lines:
        if "Threads:" in line:
            current_threads = int(re.findall(r"\d+", line)[0])
            data.setdefault(current_threads, [])
        elif "Tempo" in line:
            time = float(re.findall(r"[\d.]+", line)[0])
            data[current_threads].append(time)

    return data


def compute_stats(raw):
    """
    raw: dict threads -> list(times)
    Retorna dict threads -> mean_time
    """
    stats = {}
    for t, times in raw.items():
        if len(times) <= DISCARD:
            raise ValueError(f"Poucas execuções para {t} threads")
        trimmed = times[DISCARD:]
        stats[t] = np.mean(trimmed)
    return stats


def compute_speedup_efficiency(mean_times):
    """
    mean_times: dict threads -> mean_time
    """
    t1 = mean_times[1]
    speedup = {}
    efficiency = {}
    for t, tm in mean_times.items():
        speedup[t] = t1 / tm
        efficiency[t] = speedup[t] / t
    return speedup, efficiency


# =========================
# PROCESSAMENTO PRINCIPAL
# =========================

all_results = []

for tile in TILES:
    for key, label in VERSIONS.items():
        logfile = LOG_PATTERN.format(version=key, tile=tile)
        logfile = Path(logfile)

        if not logfile.exists():
            print(f"[AVISO] Arquivo não encontrado: {logfile}")
            continue

        raw = parse_log(logfile)
        mean_times = compute_stats(raw)
        speedup, efficiency = compute_speedup_efficiency(mean_times)

        for t in mean_times:
            all_results.append({
                "version": label,
                "tile": tile,
                "threads": t,
                "time": mean_times[t],
                "speedup": speedup[t],
                "efficiency": efficiency[t]
            })

# =========================
# DATAFRAME FINAL
# =========================

df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_DIR / "exp2_summary.csv", index=False)

print("Arquivo exp2_summary.csv gerado")

# =========================
# GRÁFICOS POR TILE
# =========================

for tile in TILES:
    df_tile = df[df["tile"] == tile]

    if df_tile.empty:
        continue

    # --- Speedup ---
    plt.figure()
    for version in df_tile["version"].unique():
        d = df_tile[df_tile["version"] == version]
        plt.plot(d["threads"], d["speedup"], marker="o", label=version)

    plt.xlabel("Número de threads")
    plt.ylabel("Speedup")
    plt.title(f"Speedup — TILE = {tile}")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / f"speedup_TILE{tile}.pdf", bbox_inches="tight")
    plt.close()

    # --- Eficiência ---
    plt.figure()
    for version in df_tile["version"].unique():
        d = df_tile[df_tile["version"] == version]
        plt.plot(d["threads"], d["efficiency"], marker="o", label=version)

    plt.xlabel("Número de threads")
    plt.ylabel("Eficiência")
    plt.title(f"Eficiência — TILE = {tile}")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / f"efficiency_TILE{tile}.pdf", bbox_inches="tight")
    plt.close()

print("Gráficos por TILE gerados com sucesso")
