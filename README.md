# proyecto3_paralela

Proyecto 3 de Computaci√≥n Paralela: aceleraci√≥n con CUDA del clasificador ASL del Laboratorio 3 de Data Science.
Enunciado: [Proyecto_3_CUDA.md](Proyecto_3_CUDA.md).

## Estructura

- `notebooks/01_eda_preprocessing.ipynb`: soluci√≥n secuencial base (Lab 3, repo `Laboratorio3-ds`). Mejor modelo: CNN 2, accuracy en test 0.9866.
- `lab3_original/`: informe y notas originales del Lab 3 (referencia).
- `common/`, `secuencial/`, `gpu/`: preprocesamiento paralelo, benchmark secuencial (TensorFlow CPU), benchmark CUDA (PyTorch + kernels Numba CUDA) y sus resultados.
- `docs/`: documentaciÛn por secciÛn (01ñ06), tablas y figuras.
- `pyproject.toml` / `uv.lock`: entorno con `uv` (Python 3.12).

## Entorno

```bash
uv sync
uv run jupyter lab
```
