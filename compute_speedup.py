import pandas as pd

df = pd.read_csv("hopscotch_times.csv")

results = []

for version in df["version"].unique():
    sub = df[df["version"] == version].sort_values("threads")
    t1 = sub[sub["threads"] == 1]["time"].values[0]

    for _, row in sub.iterrows():
        speedup = t1 / row["time"]
        efficiency = speedup / row["threads"]

        results.append({
            "version": version,
            "threads": row["threads"],
            "time": row["time"],
            "speedup": speedup,
            "efficiency": efficiency
        })

out = pd.DataFrame(results)
out.to_csv("hopscotch_metrics.csv", index=False)

print("Arquivo hopscotch_metrics.csv gerado.")
