"""Medicion de tiempos de la solucion SECUENCIAL (Lab 3 ASL) por fases.

Replica fielmente el pipeline del notebook notebooks/01_eda_preprocessing.ipynb:
  F1 listado de archivos + split estratificado + submuestreo
  F2 carga + preprocesamiento de imagenes (PIL, bucle secuencial) -> arreglos NumPy
  F3 entrenamiento CNN 1 / CNN 2 (Keras/TensorFlow en CPU, batch 32)
  F4 Random Forest (300 arboles)
  F5 inferencia (evaluacion en validacion)
Para que las corridas sean comparables, las CNN se entrenan un numero FIJO de
epocas (sin EarlyStopping). Uso: uv run python secuencial/benchmark_secuencial.py
"""
import os, sys, json, time, argparse, platform
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
from PIL import Image
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

SEED, IMG_SIZE, N_PER_CLASS = 42, 64, 600
OUT = Path(__file__).parent / "resultados"


def build_cnn_1(n):
    m = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(2, 2),
        layers.Flatten(), layers.Dense(128, activation="relu"),
        layers.Dense(n, activation="softmax")])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def build_cnn_2(n):
    m = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(2, 2),
        layers.Flatten(), layers.Dense(256, activation="relu"), layers.Dropout(0.5),
        layers.Dense(n, activation="softmax")])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def load_and_preprocess(fp):  # identica a la del notebook
    with Image.open(fp) as src:
        img = src.convert("RGB").resize((IMG_SIZE, IMG_SIZE), resample=Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def run_once(train_dir, epochs, mark=lambda s: None):
    t = {}
    mark('F1 listado+split')
    t0 = time.perf_counter()
    classes = sorted(os.listdir(train_dir))
    rows = [(str(train_dir / c / f), c) for c in classes for f in os.listdir(train_dir / c)]
    df = pd.DataFrame(rows, columns=["filepath", "label"])
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=SEED)
    sub = lambda d, n: pd.concat([g.sample(min(n, len(g)), random_state=SEED)
                                  for _, g in d.groupby("label")]).reset_index(drop=True)
    tr, va = sub(tr, int(N_PER_CLASS * .70)), sub(va, int(N_PER_CLASS * .15))
    t["F1_listado_split"] = time.perf_counter() - t0

    mark('F2 preproc secuencial')
    t0 = time.perf_counter()
    Xtr = np.stack([load_and_preprocess(p) for p in tr.filepath])
    t["F2a_preproc_train"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    Xva = np.stack([load_and_preprocess(p) for p in va.filepath])
    t["F2b_preproc_val"] = time.perf_counter() - t0
    enc = LabelEncoder().fit(tr.label)
    ytr_i, yva_i = enc.transform(tr.label), enc.transform(va.label)
    ytr, yva = to_categorical(ytr_i, len(classes)), to_categorical(yva_i, len(classes))
    t["n_train"], t["n_val"] = len(tr), len(va)

    for name, build in (("CNN1", build_cnn_1), ("CNN2", build_cnn_2)):
        mark(f'F3 entrenamiento {name}')
        tf.keras.utils.set_random_seed(SEED)
        m = build(len(classes))
        t0 = time.perf_counter()
        h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=32, verbose=0)
        t[f"F3_train_{name}_total"] = time.perf_counter() - t0
        t[f"F3_train_{name}_por_epoca"] = t[f"F3_train_{name}_total"] / epochs
        t0 = time.perf_counter()
        _, acc = m.evaluate(Xva, yva, verbose=0)
        t[f"F5_inferencia_val_{name}"] = time.perf_counter() - t0
        t[f"acc_val_{name}"] = float(acc)

    mark('F4 Random Forest')
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(Xtr.reshape(len(Xtr), -1), ytr_i)
    t["F4_random_forest_fit"] = time.perf_counter() - t0
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--monitor", default="")
    ap.add_argument("--tag", default="secuencial")
    a = ap.parse_args()
    path = Path(kagglehub.dataset_download("grassknoted/asl-alphabet"))
    train_dir = path / "asl_alphabet_train" / "asl_alphabet_train"
    env = {"cpu_count": os.cpu_count(), "platform": platform.platform(),
           "python": sys.version.split()[0], "tensorflow": tf.__version__,
           "tf_gpus": [g.name for g in tf.config.list_physical_devices("GPU")],
           "epochs": a.epochs, "runs": a.runs}
    print(env, flush=True)
    print("Calentamiento (descartado: llena cache de disco y compila grafos)...", flush=True)
    run_once(train_dir, 1)
    res = []
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.monitor import Monitor
    mon = Monitor(a.monitor) if a.monitor else None
    if mon: mon.__enter__()
    for i in range(a.runs):
        r = run_once(train_dir, a.epochs, mon.mark if mon else (lambda s: None)); r["run"] = i + 1; res.append(r)
        print(f"run {i+1}:", {k: round(v, 2) for k, v in r.items() if isinstance(v, float)}, flush=True)
    if mon: mon.mark("fin"); mon.__exit__()
    if a.tag != "secuencial":  # corrida auxiliar (p. ej. solo monitoreo): no sobrescribe las mediciones oficiales
        sys.exit(0)
    df = pd.DataFrame(res)
    df.to_csv(OUT / "secuencial_tiempos.csv", index=False)
    json.dump(env, open(OUT / "secuencial_entorno.json", "w"), indent=2)
    summ = df.drop(columns="run").agg(["mean", "std", "min", "max"]).T
    summ.to_csv(OUT / "secuencial_resumen.csv")
    print(summ.round(3).to_string())
