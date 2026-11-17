"""Benchmark de la solucion PARALELA: preprocesamiento multiproceso (CPU) + entrenamiento PyTorch (GPU/CUDA)
+ inferencia con kernels propios (Numba CUDA). Mismos datos, semilla, arquitecturas, lote y epocas que
secuencial/benchmark_secuencial.py.

  uv run python gpu/benchmark_gpu.py --device cuda --runs 5 --epochs 10
  uv run python gpu/benchmark_gpu.py --device cpu  --runs 3 --epochs 10     # PyTorch en CPU (referencia)
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    import numpy as np, pandas as pd, torch, kagglehub
    from concurrent.futures import ProcessPoolExecutor
    from common import prep
    from common.monitor import Monitor
    from gpu.models_torch import build_cnn, train, evaluate, sync

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--monitor", default="")
    a = ap.parse_args()
    dev = a.device
    OUT = Path(__file__).resolve().parents[1] / "gpu" / "resultados"; OUT.mkdir(exist_ok=True)
    tag = "gpu" if dev == "cuda" else "torch_cpu"
    if dev == "cuda":
        from numba import cuda as nbcuda
        from gpu import kernels_cuda as K
        from gpu.forward_cuda import predict
        torch.backends.cudnn.benchmark = True

    path = Path(kagglehub.dataset_download("grassknoted/asl-alphabet"))
    train_dir = path / "asl_alphabet_train" / "asl_alphabet_train"

    def run_once(epochs, mark=lambda s: None):
        t = {}
        mark("F1 listado+split"); t0 = time.perf_counter()
        classes, tr, va, _ = prep.list_split_subsample(train_dir)
        t["F1_listado_split"] = time.perf_counter() - t0
        cidx = {c: i for i, c in enumerate(classes)}

        mark("F2 preproc CPU paralelo"); t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:      # un pool por corrida (incluye su arranque)
            Xtr8 = prep.preprocess_parallel(tr.filepath, a.workers, executor=ex)
            t["F2a_preproc_train"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            Xva8 = prep.preprocess_parallel(va.filepath, a.workers, executor=ex)
            t["F2b_preproc_val"] = time.perf_counter() - t0
        ytr_np, yva_np = tr.label.map(cidx).to_numpy(), va.label.map(cidx).to_numpy()
        t["n_train"], t["n_val"] = len(tr), len(va)

        mark("T1 transferencia H2D"); sync(dev); t0 = time.perf_counter()
        d8tr, d8va = torch.from_numpy(Xtr8).to(dev), torch.from_numpy(Xva8).to(dev)
        ytr, yva = torch.from_numpy(ytr_np).to(dev), torch.from_numpy(yva_np).to(dev)
        sync(dev); t["T1_transfer_H2D"] = time.perf_counter() - t0

        mark("N normalizacion"); sync(dev); t0 = time.perf_counter()
        if dev == "cuda":   # kernel propio: uint8 NHWC -> float32 NCHW /255
            Xtr = torch.empty((len(d8tr), 3, 64, 64), device=dev); K.normalize(d8tr, Xtr)
            Xva = torch.empty((len(d8va), 3, 64, 64), device=dev); K.normalize(d8va, Xva)
        else:
            Xtr, Xva = d8tr.permute(0, 3, 1, 2).float().div(255).contiguous(), d8va.permute(0, 3, 1, 2).float().div(255).contiguous()
        sync(dev); t["N_normalizacion"] = time.perf_counter() - t0

        for name in ("CNN1", "CNN2"):
            mark(f"F3 entrenamiento {name}")
            torch.manual_seed(prep.SEED)
            m = build_cnn(name).to(dev)
            sync(dev); t0 = time.perf_counter()
            times, acc = train(m, Xtr, ytr, Xva, yva, epochs, dev)
            sync(dev); t[f"F3_train_{name}_total"] = time.perf_counter() - t0
            t[f"F3_train_{name}_por_epoca"] = t[f"F3_train_{name}_total"] / epochs
            t[f"acc_val_{name}"] = acc
            mark(f"F5 inferencia {name}")
            sync(dev); t0 = time.perf_counter(); evaluate(m, Xva, yva); sync(dev)
            t[f"F5_inferencia_val_{name}"] = time.perf_counter() - t0
            if dev == "cuda":   # misma inferencia con kernels propios
                sync(dev); t0 = time.perf_counter()
                pred = predict(m, Xva); sync(dev)
                t[f"F5_inferencia_kernels_propios_{name}"] = time.perf_counter() - t0
                t[f"acc_val_kernels_propios_{name}"] = (pred == yva).float().mean().item()
        return t

    env = {"device": dev, "cpu_count": os.cpu_count(), "workers": a.workers, "torch": torch.__version__,
           "epochs": a.epochs, "runs": a.runs,
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
           "cuda": torch.version.cuda if dev == "cuda" else None}
    print(env, flush=True)
    print("Calentamiento (descartado)...", flush=True)
    run_once(1)
    res = []
    mon = Monitor(a.monitor) if a.monitor else None
    if mon: mon.__enter__()
    for i in range(a.runs):
        r = run_once(a.epochs, mon.mark if mon else (lambda s: None)); r["run"] = i + 1; res.append(r)
        print(f"run {i+1}:", {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    if mon: mon.mark("fin"); mon.__exit__()
    df = pd.DataFrame(res)
    df.to_csv(OUT / f"{tag}_tiempos.csv", index=False)
    json.dump(env, open(OUT / f"{tag}_entorno.json", "w"), indent=2)
    summ = df.drop(columns="run").agg(["mean", "std", "min", "max"]).T
    summ.to_csv(OUT / f"{tag}_resumen.csv"); print(summ.round(3).to_string())
