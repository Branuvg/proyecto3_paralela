"""Funcion de los procesos hijo. Solo importa PIL y NumPy (arranque rapido del pool en Windows/spawn)."""
import numpy as np
from PIL import Image

IMG_SIZE = 64


def load_u8(fp):
    """Abrir + decodificar JPEG + RGB + resize 64x64 bilineal -> uint8 (64,64,3). La division /255 se hace luego en GPU."""
    with Image.open(fp) as src:
        img = src.convert("RGB").resize((IMG_SIZE, IMG_SIZE), resample=Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def load_chunk(paths):
    return np.stack([load_u8(p) for p in paths])
