"""Grafica el CSV del monitor: CPU promedio / nucleos activos y GPU (utilizacion y memoria) por fase."""
import sys
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(csv_path, png_path, title, tmax=None):
    d = pd.read_csv(csv_path)
    if tmax: d = d[d.t_s <= tmax]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(d.t_s, d.cpu_prom_pct, label="CPU promedio (%)"); ax[0].plot(d.t_s, d.cpu_max_nucleo_pct, alpha=.5, label="nucleo mas cargado (%)")
    ax[0].set_ylabel("CPU %"); ax[0].legend(loc="upper right")
    ax[1].plot(d.t_s, d.nucleos_gt50, color="tab:green"); ax[1].set_ylabel("nucleos > 50 %")
    ax[2].plot(d.t_s, d.gpu_util_pct, color="tab:red", label="GPU util (%)")
    a2 = ax[2].twinx(); a2.plot(d.t_s, d.gpu_mem_MiB, color="tab:purple", ls="--", label="memoria GPU (MiB)")
    ax[2].set_ylabel("GPU %"); a2.set_ylabel("MiB"); ax[2].set_xlabel("tiempo (s)")
    # bandas por fase
    fases = d.fase.ne(d.fase.shift()).cumsum()
    cols = plt.cm.Pastel1.colors
    for k, (_, g) in enumerate(d.groupby(fases)):
        for a in ax:
            a.axvspan(g.t_s.min(), g.t_s.max(), color=cols[k % len(cols)], alpha=.35)
        if g.t_s.max() - g.t_s.min() > 0.045 * (d.t_s.max() - d.t_s.min()):
            ax[0].text(g.t_s.min(), 101, g.fase.iloc[0], fontsize=7, va="bottom")
    ax[0].set_ylim(0, 105); ax[2].set_ylim(0, 105)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(png_path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    plot(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "Uso de recursos")
