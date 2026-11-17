"""Monitor de recursos: CPU (por nucleo, psutil) y GPU (nvidia-smi) con marcas de fase."""
import csv, subprocess, threading, time
import psutil

GPU_Q = "utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu"


class Monitor:
    def __init__(self, path, interval=0.5):
        self.path, self.interval = path, interval
        self.label, self.rows, self._stop = "inicio", [], threading.Event()
        self.ncpu = psutil.cpu_count()

    def mark(self, label):
        self.label = label

    def _gpu(self):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_Q}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().split(",")
            return [float(v) for v in out]
        except Exception:
            return [float("nan")] * 5

    def _loop(self):
        psutil.cpu_percent(percpu=True)
        t0 = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.interval)
            cores = psutil.cpu_percent(percpu=True)
            self.rows.append([round(time.perf_counter() - t0, 2), self.label,
                              round(sum(cores) / len(cores), 1), max(cores),
                              sum(c > 50 for c in cores)] + self._gpu() + cores)

    def __enter__(self):
        self._th = threading.Thread(target=self._loop, daemon=True); self._th.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self._th.join()
        head = ["t_s", "fase", "cpu_prom_pct", "cpu_max_nucleo_pct", "nucleos_gt50", "gpu_util_pct",
                "gpu_mem_util_pct", "gpu_mem_MiB", "gpu_power_W", "gpu_temp_C"] + [f"cpu{i}" for i in range(self.ncpu)]
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(head); w.writerows(self.rows)
