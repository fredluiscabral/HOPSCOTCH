import re
import csv

files = {
    "NAIVE": "NAIVE.log",
    "BUSYWAIT": "BUSYWAIT.log",
    "SEMAPHORE": "SEMAPHORE.log"
}

pattern_threads = re.compile(r"Threads:\s*(\d+)")
pattern_time = re.compile(r"Tempo\s*:\s*([0-9.]+)")

results = []

for version, filename in files.items():
    with open(filename, "r") as f:
        current_threads = None

        for line in f:
            m_t = pattern_threads.search(line)
            if m_t:
                current_threads = int(m_t.group(1))
                continue

            m_time = pattern_time.search(line)
            if m_time and current_threads is not None:
                time = float(m_time.group(1))
                results.append({
                    "version": version,
                    "threads": current_threads,
                    "time": time
                })
                current_threads = None  # reseta para evitar erro

# grava CSV
with open("hopscotch_times.csv", "w", newline="") as csvfile:
    fieldnames = ["version", "threads", "time"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("Arquivo hopscotch_times.csv gerado com sucesso.")
