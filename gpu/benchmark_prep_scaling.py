"""Escalamiento del preprocesamiento: 1 proceso (secuencial, sin pool) vs N procesos. Incluye arranque del pool."""
import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import pandas as pd, kagglehub
    from common import prep
    path = Path(kagglehub.dataset_download("grassknoted/asl-alphabet"))
    _, tr, va, _ = prep.list_split_subsample(path / "asl_alphabet_train" / "asl_alphabet_train")
    paths = list(tr.filepath) + list(va.filepath)
    prep.preprocess_sequential(paths[:2000])  # calentar cache de disco
    rows = []
    for w in [1, 2, 4, 8, 12, 16]:
        for rep in range(5):
            t0 = time.perf_counter()
            X = prep.preprocess_sequential(paths) if w == 1 else prep.preprocess_parallel(paths, w)
            rows.append({"procesos": w, "rep": rep + 1, "segundos": time.perf_counter() - t0, "n": len(X)})
        print(w, [round(r["segundos"], 2) for r in rows[-5:]], flush=True)
    d = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "resultados"; out.mkdir(exist_ok=True)
    d.to_csv(out / "prep_escalamiento_tiempos.csv", index=False)
    s = d.groupby("procesos").segundos.agg(["mean", "std", "min"]); s["speedup"] = s.loc[1, "mean"] / s["mean"]
    s["eficiencia"] = s.speedup / s.index
    s.to_csv(out / "prep_escalamiento_resumen.csv"); print(s.round(3))
