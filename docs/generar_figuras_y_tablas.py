"""Genera tablas (docs/tablas_comparacion.md) y figuras (docs/figuras/*.png) de las secciones 5 y 6."""
import sys; from pathlib import Path
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common.plot_monitor import plot
F = R / "docs" / "figuras"; F.mkdir(exist_ok=True)
S = pd.read_csv(R / "secuencial/resultados/secuencial_tiempos.csv")
G = pd.read_csv(R / "gpu/resultados/gpu_tiempos.csv")
P = pd.read_csv(R / "gpu/resultados/torch_cpu_tiempos.csv")
sc = pd.read_csv(R / "gpu/resultados/prep_escalamiento_resumen.csv", index_col=0)
out = []
def ph(d, par=False):
    return {"F1": d.F1_listado_split, "F2": d.F2a_preproc_train + d.F2b_preproc_val,
            "F3a": d.F3_train_CNN1_total, "F3b": d.F3_train_CNN2_total}
s, g, p = ph(S), ph(G), ph(P)
m = lambda x: x.mean(); sd = lambda x: x.std()
rows = [("F1 Listado + split (CPU, igual)", "F1"), ("F2 Preprocesamiento train+val", "F2"),
        ("F3 Entrenamiento CNN 1 (10 ép.)", "F3a"), ("F3 Entrenamiento CNN 2 (10 ép.)", "F3b")]
out.append("| Fase | Secuencial (TF, CPU) s | Paralelo (CPU multiproc. / GPU) s | Speedup | PyTorch en CPU s |\n|---|---|---|---|---|")
for name, k in rows:
    out.append(f"| {name} | {m(s[k]):.2f} ± {sd(s[k]):.2f} | {m(g[k]):.2f} ± {sd(g[k]):.2f} | **{m(s[k])/m(g[k]):.2f}×** | {m(p[k]):.2f} ± {sd(p[k]):.2f} |")
ex = G.T1_transfer_H2D + G.N_normalizacion
out.append(f"| T1+N Transferencia H2D + normalización GPU | — | {m(ex):.3f} ± {sd(ex):.3f} | — | {m(P.T1_transfer_H2D+P.N_normalizacion):.3f} |")
inf_s = S.F5_inferencia_val_CNN1 + S.F5_inferencia_val_CNN2
inf_g = G.F5_inferencia_val_CNN1 + G.F5_inferencia_val_CNN2
inf_k = G.F5_inferencia_kernels_propios_CNN1 + G.F5_inferencia_kernels_propios_CNN2
inf_p = P.F5_inferencia_val_CNN1 + P.F5_inferencia_val_CNN2
out.append(f"| F5 Inferencia val. (CNN 1+2) | {m(inf_s):.3f} | PyTorch GPU {m(inf_g):.3f} / kernels propios {m(inf_k):.3f} | {m(inf_s)/m(inf_g):.2f}× / {m(inf_s)/m(inf_k):.2f}× | {m(inf_p):.3f} |")
tot_s = S.F1_listado_split + S.F2a_preproc_train + S.F2b_preproc_val + S.F3_train_CNN1_total + S.F3_train_CNN2_total + inf_s
tot_g = G.F1_listado_split + G.F2a_preproc_train + G.F2b_preproc_val + ex + G.F3_train_CNN1_total + G.F3_train_CNN2_total + inf_g
tot_p = P.F1_listado_split + P.F2a_preproc_train + P.F2b_preproc_val + P.T1_transfer_H2D + P.N_normalizacion + P.F3_train_CNN1_total + P.F3_train_CNN2_total + inf_p
out.append(f"| **Total pipeline CNN (sin Random Forest)** | **{m(tot_s):.2f} ± {sd(tot_s):.2f}** | **{m(tot_g):.2f} ± {sd(tot_g):.2f}** | **{m(tot_s)/m(tot_g):.2f}×** | {m(tot_p):.2f} ± {sd(tot_p):.2f} |")
prep_s = S.F1_listado_split + S.F2a_preproc_train + S.F2b_preproc_val
prep_g = G.F1_listado_split + G.F2a_preproc_train + G.F2b_preproc_val
train_s = S.F3_train_CNN1_total + S.F3_train_CNN2_total
train_g = G.F3_train_CNN1_total + G.F3_train_CNN2_total
out.append("")
out.append("| Agrupación | Secuencial s | Paralelo s | Speedup |\n|---|---|---|---|")
out.append(f"| Preparación de datos (F1+F2) | {m(prep_s):.2f} | {m(prep_g):.2f} | {m(prep_s)/m(prep_g):.2f}× |")
out.append(f"| Entrenamiento (CNN 1+2) | {m(train_s):.2f} | {m(train_g):.2f} | {m(train_s)/m(train_g):.2f}× |")
out.append(f"| Total | {m(tot_s):.2f} | {m(tot_g):.2f} | {m(tot_s)/m(tot_g):.2f}× |")
out.append("")
out.append("| Por época | Secuencial (TF CPU) | PyTorch CPU | PyTorch GPU | Speedup GPU vs TF CPU | Speedup GPU vs PyTorch CPU |\n|---|---|---|---|---|---|")
for n in ("CNN1", "CNN2"):
    a, b, c = S[f"F3_train_{n}_por_epoca"].mean(), P[f"F3_train_{n}_por_epoca"].mean(), G[f"F3_train_{n}_por_epoca"].mean()
    out.append(f"| {n} | {a:.2f} s | {b:.2f} s | {c:.2f} s | {a/c:.2f}× | {b/c:.2f}× |")
out.append("")
out.append("| Accuracy val. (10 ép.) | Secuencial TF | PyTorch GPU | PyTorch CPU |\n|---|---|---|---|")
for n in ("CNN1", "CNN2"):
    out.append(f"| {n} | {S['acc_val_'+n].mean():.4f} | {G['acc_val_'+n].mean():.4f} ± {G['acc_val_'+n].std():.4f} | {P['acc_val_'+n].mean():.4f} |")
# por corrida
out.append("")
out.append("| Corrida | Total secuencial s | Total paralelo s | Speedup |\n|---|---|---|---|")
for i in range(len(S)):
    out.append(f"| {i+1} | {tot_s[i]:.2f} | {tot_g[i]:.2f} | {tot_s[i]/tot_g[i]:.2f}× |")
out.append("")
out.append("| Procesos | Tiempo (s) | Speedup | Eficiencia |\n|---|---|---|---|")
for w, r in sc.iterrows():
    out.append(f"| {w} | {r['mean']:.2f} ± {r['std']:.2f} | {r.speedup:.2f}× | {r.eficiencia:.0%} |")
# monitor por fase
out.append("")
for nm, f in (("secuencial", "secuencial/resultados/monitor_secuencial.csv"), ("paralelo", "gpu/resultados/monitor_gpu.csv")):
    d = pd.read_csv(R / f); d = d[d.fase != "fin"]
    g_ = d.groupby("fase", sort=False).agg(muestras=("t_s", "size"), cpu_prom=("cpu_prom_pct", "mean"), nucleos_gt50=("nucleos_gt50", "mean"),
                                           cpu_max=("cpu_max_nucleo_pct", "max"), gpu_util=("gpu_util_pct", "mean"), gpu_util_max=("gpu_util_pct", "max"),
                                           gpu_mem=("gpu_mem_MiB", "max"), gpu_pw=("gpu_power_W", "mean"))
    out.append(f"**Monitor ({nm})**\n\n| Fase | Muestras | CPU prom % | Núcleos >50 % | GPU util prom % | GPU util máx % | Mem GPU máx MiB | Potencia GPU W |\n|---|---|---|---|---|---|---|---|")
    for k, r in g_.iterrows():
        out.append(f"| {k} | {r.muestras:.0f} | {r.cpu_prom:.0f} | {r.nucleos_gt50:.1f} | {r.gpu_util:.0f} | {r.gpu_util_max:.0f} | {r.gpu_mem:.0f} | {r.gpu_pw:.0f} |")
    out.append("")
    plot(R / f, F / f"monitor_{nm}.png", f"Uso de recursos - {nm}")
    if nm == "paralelo": plot(R / f, F / "monitor_paralelo_1corrida.png", "Uso de recursos - paralelo (primera corrida)", tmax=27)
(R / "docs" / "tablas_comparacion.md").write_text("\n".join(out), encoding="utf-8")
# figuras de comparacion
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
labs = ["Preparación\n(F1+F2)", "Entrenamiento\nCNN 1+2", "Inferencia", "Total"]
vs = [m(prep_s), m(train_s), m(inf_s), m(tot_s)]; vg = [m(prep_g), m(train_g), m(inf_g), m(tot_g)]
x = np.arange(4); ax[0].bar(x - .2, vs, .4, label="Secuencial (TF, CPU)"); ax[0].bar(x + .2, vg, .4, label="Paralelo (CPU multiproc. + GPU)")
ax[0].set_xticks(x, labs); ax[0].set_ylabel("segundos"); ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("Tiempo por fase (escala log)")
sp = [a / b for a, b in zip(vs, vg)]; bars = ax[1].bar(labs, sp, color="tab:green"); ax[1].axhline(1, color="k", lw=.8)
for b_, v in zip(bars, sp): ax[1].text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}×", ha="center", va="bottom")
ax[1].set_title("Speedup (secuencial / paralelo)"); fig.tight_layout(); fig.savefig(F / "comparacion_fases.png", dpi=130); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(sc.index, sc.speedup, "o-", label="medido"); ax[0].plot(sc.index, sc.index, "k--", label="ideal"); ax[0].set_xlabel("procesos"); ax[0].set_ylabel("speedup"); ax[0].legend(); ax[0].set_title("Preprocesamiento: speedup")
ax[1].plot(sc.index, sc.eficiencia * 100, "o-", color="tab:orange"); ax[1].set_xlabel("procesos"); ax[1].set_ylabel("eficiencia %"); ax[1].set_title("Eficiencia")
fig.tight_layout(); fig.savefig(F / "escalamiento_preprocesamiento.png", dpi=130); plt.close(fig)
print("\n".join(out))
