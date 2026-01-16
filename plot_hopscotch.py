import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hopscotch_metrics.csv")

# ---------------- Tempo ----------------
plt.figure()
for version in df["version"].unique():
    sub = df[df["version"] == version]
    plt.plot(sub["threads"], sub["time"], marker="o", label=version)

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Número de threads")
plt.ylabel("Tempo (s)")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("tempo_vs_threads.pdf")

# ---------------- Speedup ----------------
plt.figure()
for version in df["version"].unique():
    sub = df[df["version"] == version]
    plt.plot(sub["threads"], sub["speedup"], marker="o", label=version)

threads_unique = sorted(df["threads"].unique())
plt.plot(threads_unique, threads_unique, "--", label="Speedup ideal")

plt.xscale("log", base=2)
plt.xlabel("Número de threads")
plt.ylabel("Speedup")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("speedup.pdf")

# ---------------- Eficiência ----------------
plt.figure()
for version in df["version"].unique():
    sub = df[df["version"] == version]
    plt.plot(sub["threads"], sub["efficiency"], marker="o", label=version)

plt.xscale("log", base=2)
plt.xlabel("Número de threads")
plt.ylabel("Eficiência")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("efficiency.pdf")

print("Gráficos gerados com sucesso.")
