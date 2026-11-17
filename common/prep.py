"""Preparacion de datos (CPU). Modulo ligero: lo importan los procesos hijo (spawn en Windows),
por eso NO importa torch ni tensorflow."""
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

SEED, IMG_SIZE, N_PER_CLASS = 42, 64, 600


def list_split_subsample(train_dir):
    """F1: identico al notebook. Devuelve (classes, train_df, val_df, test_df)."""
    classes = sorted(os.listdir(train_dir))
    rows = [(str(train_dir / c / f), c) for c in classes for f in os.listdir(train_dir / c)]
    df = pd.DataFrame(rows, columns=["filepath", "label"])
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=SEED)
    sub = lambda d, n: pd.concat([g.sample(min(n, len(g)), random_state=SEED)
                                  for _, g in d.groupby("label")]).reset_index(drop=True)
    return classes, sub(tr, int(N_PER_CLASS * .70)), sub(va, int(N_PER_CLASS * .15)), sub(te, int(N_PER_CLASS * .15))


from common.prep_worker import load_u8, load_chunk  # noqa: E402  (modulo ligero, tambien usado por los hijos)


def preprocess_sequential(paths):
    return load_chunk(list(paths))


def preprocess_parallel(paths, workers, chunk=128, executor=None):
    """Reparte las rutas en bloques entre procesos. Si no se pasa `executor`, crea (y cierra) un pool propio,
    de modo que el tiempo incluya el arranque de los procesos."""
    from concurrent.futures import ProcessPoolExecutor
    paths = list(paths)
    chunks = [paths[i:i + chunk] for i in range(0, len(paths), chunk)]
    if executor is None:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return np.concatenate(list(ex.map(load_chunk, chunks)))
    return np.concatenate(list(executor.map(load_chunk, chunks)))
