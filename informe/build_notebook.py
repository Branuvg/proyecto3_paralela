"""Genera Proyecto3_CUDA.ipynb (entregable unico: codigo secuencial original + version optimizada con CUDA).
El codigo de kernels, forward, modelos y preprocesamiento se toma de los modulos del repo (gpu/, common/)
para que el notebook y los benchmarks ejecuten exactamente el mismo codigo."""
import re
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
rd = lambda p: (ROOT / p).read_text(encoding="utf8")
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Proyecto 3: Mejorando algoritmos con CUDA
**CC3069 Computación Paralela, UVG.** Clasificador del alfabeto ASL (Laboratorio 3 de Data Science).

Este notebook contiene en un solo archivo:

1. **Parte A:** la solución **secuencial original** (TensorFlow/Keras en CPU), con medición de tiempo por fase.
2. **Parte B:** la versión **optimizada**, con la preparación de datos en **CPU multiproceso** (B1) separada de la matemática de la red en **GPU con CUDA** (B2: kernels Numba CUDA propios + PyTorch/cuDNN).
3. **Parte C:** verificación de corrección de los kernels, comparación de tiempos y speedup.

Las mediciones completas (5 corridas, resultados, figuras y análisis) están en `informe/` y `docs/`. Este notebook ejecuta por defecto **1 corrida medida** (`N_RUNS`) con **10 épocas fijas** (`EPOCHS`) para que sea rápido de reproducir.

Requisitos: `uv sync` (ver `README.md`), GPU NVIDIA con CUDA 12.8 y el dataset de Kaggle `grassknoted/asl-alphabet` (se descarga con `kagglehub`).
""")

md("## Configuración e importaciones")
code('''
import os, sys, time, warnings, types
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())          # el notebook se ejecuta desde la raiz del repo

import numpy as np, pandas as pd
import kagglehub
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import torch, torch.nn as nn
from numba import cuda, float32

SEED, IMG_SIZE, N_PER_CLASS = 42, 64, 600     # mismos valores que el laboratorio original
EPOCHS = 10                                   # epocas fijas (sin EarlyStopping) para comparar el mismo trabajo
N_RUNS = 1                                    # corridas medidas (el informe usa 5)
INCLUDE_RF = True                             # Random Forest solo en la parte secuencial
WORKERS = os.cpu_count()

path = Path(kagglehub.dataset_download("grassknoted/asl-alphabet"))
TRAIN_DIR = path / "asl_alphabet_train" / "asl_alphabet_train"
print("CPU logicos:", WORKERS, "| TF GPUs:", tf.config.list_physical_devices("GPU"),
      "| torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
''')

# ------------------------------------------------------------------ Parte A
md("""
# Parte A. Solución secuencial original (TensorFlow en CPU)

Réplica fiel del pipeline del notebook del laboratorio (`notebooks/01_eda_preprocessing.ipynb`):

| Fase | Contenido |
|---|---|
| F1 | listado de archivos, split estratificado 70/15/15 y submuestreo (600 imágenes por clase) |
| F2 | carga y preprocesamiento de cada imagen (PIL, un solo proceso): RGB, resize 64×64, `/255` |
| F3 | entrenamiento de CNN 1 y CNN 2 (Keras, lote 32, Adam) |
| F5 | inferencia (evaluación en validación) |
| F4 | Random Forest (300 árboles, `n_jobs=-1`) |

TensorFlow en Windows nativo **no usa GPU**, por lo que esta línea base corre en CPU (usa los 16 hilos mediante oneDNN).
""")
code('''
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


def load_and_preprocess(fp):   # identica a la del laboratorio
    with Image.open(fp) as src:
        img = src.convert("RGB").resize((IMG_SIZE, IMG_SIZE), resample=Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def run_sequential(epochs, rf=True):
    t = {}
    t0 = time.perf_counter()                                           # F1
    classes = sorted(os.listdir(TRAIN_DIR))
    rows = [(str(TRAIN_DIR / c / f), c) for c in classes for f in os.listdir(TRAIN_DIR / c)]
    df = pd.DataFrame(rows, columns=["filepath", "label"])
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=SEED)
    sub = lambda d, n: pd.concat([g.sample(min(n, len(g)), random_state=SEED)
                                  for _, g in d.groupby("label")]).reset_index(drop=True)
    tr, va = sub(tr, int(N_PER_CLASS * .70)), sub(va, int(N_PER_CLASS * .15))
    t["F1_listado_split"] = time.perf_counter() - t0

    t0 = time.perf_counter()                                           # F2 (un solo proceso)
    Xtr = np.stack([load_and_preprocess(p) for p in tr.filepath])
    Xva = np.stack([load_and_preprocess(p) for p in va.filepath])
    t["F2_preproc"] = time.perf_counter() - t0
    enc = LabelEncoder().fit(tr.label)
    ytr_i, yva_i = enc.transform(tr.label), enc.transform(va.label)
    ytr, yva = to_categorical(ytr_i, len(classes)), to_categorical(yva_i, len(classes))

    for name, build in (("CNN1", build_cnn_1), ("CNN2", build_cnn_2)):  # F3 + F5
        tf.keras.utils.set_random_seed(SEED)
        m = build(len(classes))
        t0 = time.perf_counter()
        m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=32, verbose=0)
        t[f"F3_train_{name}"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, acc = m.evaluate(Xva, yva, verbose=0)
        t[f"F5_inferencia_{name}"] = time.perf_counter() - t0
        t[f"acc_{name}"] = float(acc)

    if rf:                                                              # F4
        t0 = time.perf_counter()
        RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr.reshape(len(Xtr), -1), ytr_i)
        t["F4_random_forest"] = time.perf_counter() - t0
    return t
''')
code('''
print("Calentamiento secuencial (1 epoca, descartado)...")
run_sequential(1, rf=False)
seq_runs = []
for i in range(N_RUNS):
    r = run_sequential(EPOCHS, rf=INCLUDE_RF); seq_runs.append(r)
    print(f"secuencial, corrida {i+1}:", {k: round(v, 2) for k, v in r.items()})
seq = pd.DataFrame(seq_runs).mean()
''')

# ------------------------------------------------------------------ Parte B1
md("""
# Parte B. Versión optimizada

## B1. Preparación de datos en CPU multinúcleo

Lo que se hace en CPU (E/S, decodificación JPEG): el preprocesamiento por imagen es **independiente**, así que se reparte en bloques de 128 imágenes entre un pool de procesos (`ProcessPoolExecutor`, uno por hilo lógico). Se usan procesos y no hilos por el GIL de Python. Los procesos devuelven `uint8`; la división `/255` se hará en la GPU, lo que reduce la transferencia CPU→GPU 4 veces.

La función de los procesos hijo debe poder importarse (en Windows se usa *spawn*), por eso se escribe en un módulo ligero (solo PIL y NumPy) con `%%writefile`.
""")
cells.append(nbf.v4.new_code_cell("%%writefile prep_worker_nb.py\n" + rd("common/prep_worker.py").strip("\n")))
code('''
from concurrent.futures import ProcessPoolExecutor
from prep_worker_nb import load_chunk


def list_split_subsample():
    """F1 (identica a la secuencial)."""
    classes = sorted(os.listdir(TRAIN_DIR))
    rows = [(str(TRAIN_DIR / c / f), c) for c in classes for f in os.listdir(TRAIN_DIR / c)]
    df = pd.DataFrame(rows, columns=["filepath", "label"])
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=SEED)
    sub = lambda d, n: pd.concat([g.sample(min(n, len(g)), random_state=SEED)
                                  for _, g in d.groupby("label")]).reset_index(drop=True)
    return classes, sub(tr, int(N_PER_CLASS * .70)), sub(va, int(N_PER_CLASS * .15))


def preprocess_parallel(paths, executor, chunk=128):
    paths = list(paths)
    chunks = [paths[i:i + chunk] for i in range(0, len(paths), chunk)]
    return np.concatenate(list(executor.map(load_chunk, chunks)))
''')

# ------------------------------------------------------------------ Parte B2
md("""
## B2. Matemática de la red en GPU con CUDA

### Kernels CUDA propios (Numba CUDA)

Mapeo de trabajo a hilos, bloques y grids (diseño completo en `docs/04_diseno_algoritmo_paralelo.md`):

| Kernel | Hilo = | Bloque | Grid | Memoria compartida |
|---|---|---|---|---|
| `k_normalize` | 1 valor de salida (*grid-stride*) | 256 (1D) | `min(⌈N·3·64·64/256⌉, 4096)` | No |
| `k_conv3x3_relu` | 1 píxel de salida | 16×16 | `(⌈OW/16⌉, ⌈OH/16⌉, N·Cout)` | Sí: pesos del filtro |
| `k_maxpool2` | 1 píxel de salida | 16×16 | `(⌈OW/16⌉, ⌈OH/16⌉, N·C)` | No |
| `k_dense` | 1 elemento de la matriz resultado | 16×16 | `(⌈Nout/16⌉, ⌈M/16⌉)` | Sí: GEMM por teselas |
| `k_argmax` | 1 muestra | 128 (1D) | `⌈N/128⌉` | No |

Los tensores son tensores de PyTorch en GPU; Numba los lee directamente por `__cuda_array_interface__`, sin copias.
""")
kern = re.sub(r'^""".*?"""\n', "", rd("gpu/kernels_cuda.py"), flags=re.S)
kern = kern.replace("import math\nfrom numba import cuda, float32, int32\n", "")
code(kern)

md("""
### Propagación hacia adelante con los kernels propios
Usa **solo** los kernels anteriores (sin cuDNN). El entrenamiento (retropropagación + Adam) lo hace PyTorch, por eso esta función se usa para la **inferencia** y para validar la corrección.
""")
fw = re.sub(r'^""".*?"""\n', "", rd("gpu/forward_cuda.py"), flags=re.S)
fw = fw.replace("import torch\nimport torch.nn as nn\nfrom gpu import kernels_cuda as K\n", "")
code("K = types.SimpleNamespace(normalize=normalize, conv3x3_relu=conv3x3_relu, maxpool2=maxpool2, dense=dense, argmax=argmax)\n\n" + fw)

md("""
### Modelos y entrenamiento en PyTorch (cuDNN/cuBLAS)
Mismas arquitecturas que Keras (CNN 1: 1,628,893 parámetros; CNN 2: 1,280,605), inicialización Glorot, Adam (lr 1e-3, ε 1e-7), lote 32, barajado por época y validación al final de cada época. Los datos y los pesos permanecen en la GPU.
""")
mt = re.sub(r'^""".*?"""\n', "", rd("gpu/models_torch.py"), flags=re.S)
mt = mt.replace("import time\nimport torch\nimport torch.nn as nn\n", "")
code(mt)

md("""
### Pipeline paralelo completo
Fases: F1 (CPU) → F2 preprocesamiento (CPU multiproceso) → **T1 transferencia CPU→GPU (incluida en la medición)** → N normalización (kernel propio) → F3 entrenamiento (GPU) → F5 inferencia. Se sincroniza la GPU (`torch.cuda.synchronize()`) antes y después de cada fase, porque los lanzamientos de kernels son asíncronos.
""")
code('''
torch.backends.cudnn.benchmark = True
dev = "cuda"


def run_parallel(epochs):
    t = {}
    t0 = time.perf_counter()
    classes, tr, va = list_split_subsample()                                   # F1 (CPU)
    t["F1_listado_split"] = time.perf_counter() - t0
    cidx = {c: i for i, c in enumerate(classes)}

    t0 = time.perf_counter()                                                   # F2 (CPU multiproceso)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:                        # el tiempo incluye el arranque del pool
        Xtr8 = preprocess_parallel(tr.filepath, ex)
        Xva8 = preprocess_parallel(va.filepath, ex)
    t["F2_preproc"] = time.perf_counter() - t0
    ytr_np, yva_np = tr.label.map(cidx).to_numpy(), va.label.map(cidx).to_numpy()

    sync(dev); t0 = time.perf_counter()                                         # T1: transferencia CPU -> GPU
    d8tr, d8va = torch.from_numpy(Xtr8).to(dev), torch.from_numpy(Xva8).to(dev)
    ytr, yva = torch.from_numpy(ytr_np).to(dev), torch.from_numpy(yva_np).to(dev)
    sync(dev); t["T1_transfer_H2D"] = time.perf_counter() - t0

    sync(dev); t0 = time.perf_counter()                                         # N: normalizacion en GPU (kernel propio)
    Xtr = torch.empty((len(d8tr), 3, 64, 64), device=dev); normalize(d8tr, Xtr)
    Xva = torch.empty((len(d8va), 3, 64, 64), device=dev); normalize(d8va, Xva)
    sync(dev); t["N_normalizacion"] = time.perf_counter() - t0

    for name in ("CNN1", "CNN2"):
        torch.manual_seed(SEED)
        m = build_cnn(name).to(dev)
        sync(dev); t0 = time.perf_counter()
        train(m, Xtr, ytr, Xva, yva, epochs, dev)                               # F3 (GPU)
        sync(dev); t[f"F3_train_{name}"] = time.perf_counter() - t0
        t[f"acc_{name}"] = evaluate(m, Xva, yva)
        sync(dev); t0 = time.perf_counter(); evaluate(m, Xva, yva); sync(dev)   # F5 con PyTorch/cuDNN
        t[f"F5_inferencia_{name}"] = time.perf_counter() - t0
        sync(dev); t0 = time.perf_counter(); pred = predict(m, Xva); sync(dev)  # F5 con kernels propios
        t[f"F5_inferencia_kernels_{name}"] = time.perf_counter() - t0
        t[f"acc_kernels_{name}"] = (pred == yva).float().mean().item()
    return t


print("Calentamiento paralelo (1 epoca, descartado: inicializa CUDA, cuDNN y compila los kernels)...")
run_parallel(1)
par_runs = []
for i in range(N_RUNS):
    r = run_parallel(EPOCHS); par_runs.append(r)
    print(f"paralelo, corrida {i+1}:", {k: round(v, 3) for k, v in r.items()})
par = pd.DataFrame(par_runs).mean()
''')

# ------------------------------------------------------------------ Parte C
md("""
# Parte C. Verificación y resultados

## C1. Corrección de los kernels propios
Se compara la salida de los kernels propios con la de PyTorch (misma red, mismos pesos).
""")
code('''
torch.manual_seed(0)
xu8 = torch.randint(0, 256, (64, 64, 64, 3), dtype=torch.uint8, device=dev)
xf = torch.empty((64, 3, 64, 64), device=dev); normalize(xu8, xf)
print("normalizacion, diferencia maxima vs x/255:", (xf - xu8.permute(0, 3, 1, 2).float() / 255).abs().max().item())
for name in ("CNN1", "CNN2"):
    m = build_cnn(name).to(dev).eval()
    x = torch.rand(64, 3, 64, 64, device=dev)
    ref, mine = m(x), forward(m, x)
    same = (ref.argmax(1) == mine.argmax(1)).float().mean().item()
    print(f"{name}: diferencia maxima de logits = {(ref - mine).abs().max().item():.2e}, clase predicha identica en {same*100:.0f} %")
    assert same == 1.0
''')

md("## C2. Comparación de tiempos y speedup (segundos)")
code('''
rows = [
    ("F1 Listado + split + submuestreo", seq["F1_listado_split"], par["F1_listado_split"]),
    ("F2 Preprocesamiento de imágenes", seq["F2_preproc"], par["F2_preproc"]),
    ("T1+N Transferencia CPU->GPU + normalización", 0.0, par["T1_transfer_H2D"] + par["N_normalizacion"]),
    ("F3 Entrenamiento CNN 1", seq["F3_train_CNN1"], par["F3_train_CNN1"]),
    ("F3 Entrenamiento CNN 2", seq["F3_train_CNN2"], par["F3_train_CNN2"]),
    ("F5 Inferencia CNN 1+2 (PyTorch/cuDNN)", seq["F5_inferencia_CNN1"] + seq["F5_inferencia_CNN2"],
     par["F5_inferencia_CNN1"] + par["F5_inferencia_CNN2"]),
    ("F5 Inferencia CNN 1+2 (kernels propios)", seq["F5_inferencia_CNN1"] + seq["F5_inferencia_CNN2"],
     par["F5_inferencia_kernels_CNN1"] + par["F5_inferencia_kernels_CNN2"]),
]
cmp = pd.DataFrame(rows, columns=["Fase", "Secuencial (TF, CPU)", "Paralelo (CPU + GPU)"])
cmp["Speedup"] = np.where(cmp["Secuencial (TF, CPU)"] > 0, cmp["Secuencial (TF, CPU)"] / cmp["Paralelo (CPU + GPU)"], np.nan)

seq_total = seq["F1_listado_split"] + seq["F2_preproc"] + seq["F3_train_CNN1"] + seq["F3_train_CNN2"] + seq["F5_inferencia_CNN1"] + seq["F5_inferencia_CNN2"]
par_total = (par["F1_listado_split"] + par["F2_preproc"] + par["T1_transfer_H2D"] + par["N_normalizacion"]
             + par["F3_train_CNN1"] + par["F3_train_CNN2"] + par["F5_inferencia_CNN1"] + par["F5_inferencia_CNN2"])
cmp.loc[len(cmp)] = ["TOTAL pipeline de CNN (sin Random Forest, inferencia PyTorch)", seq_total, par_total, seq_total / par_total]
display(cmp.round(3))
if "F4_random_forest" in seq:
    print(f"Random Forest (solo CPU, sin cambios): {seq['F4_random_forest']:.2f} s. "
          f"Speedup total incluyéndolo: {(seq_total + seq['F4_random_forest']) / (par_total + seq['F4_random_forest']):.2f}x")
print(f"Accuracy en validación ({EPOCHS} épocas): "
      f"CNN1 TF {seq['acc_CNN1']:.4f} / GPU {par['acc_CNN1']:.4f} | CNN2 TF {seq['acc_CNN2']:.4f} / GPU {par['acc_CNN2']:.4f}; "
      f"kernels propios: CNN1 {par['acc_kernels_CNN1']:.4f}, CNN2 {par['acc_kernels_CNN2']:.4f}")
''')

md("""
## C3. Lectura de resultados

- El entrenamiento de las CNN domina el costo y es lo que la GPU acelera (≈ 3.4–3.6× frente a TensorFlow en CPU); el preprocesamiento mejora ≈ 7× con el pool de procesos.
- La transferencia CPU→GPU más la normalización cuestan milésimas de segundo porque se envía `uint8` una sola vez y los datos permanecen en la GPU.
- Los kernels propios producen las mismas predicciones que PyTorch pero son más lentos que cuDNN (convolución directa, sin optimizaciones avanzadas); su valor es de diseño y validación.
- La ganancia total queda limitada por la fracción que no se acelera (Ley de Amdahl), por el tamaño pequeño del problema (64×64, lote 32) y por la línea base fuerte (TensorFlow ya usa 15 de 16 núcleos).
- Una sola corrida es ruidosa (en el informe: 5 corridas, media ± desviación). Las cifras provienen de una sola máquina (Windows 11, 16 hilos, RTX 5060 Laptop) y no deben extrapolarse.

Análisis completo, figuras y evidencia de uso de recursos (Administrador de Tareas, `nvidia-smi`, `psutil`): `informe/Proyecto3_Informe` y `docs/`.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, ROOT / "Proyecto3_CUDA.ipynb")
print("OK", len(cells), "celdas")
