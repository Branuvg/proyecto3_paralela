# 2. Medición del tiempo de ejecución secuencial

## 2.1 Procedimiento

- **Script:** `secuencial/benchmark_secuencial.py`. Replica el pipeline del notebook (mismas funciones de preprocesamiento, mismas arquitecturas CNN 1 y CNN 2, mismo split y submuestra con `SEED=42`) y mide cada fase con `time.perf_counter()`.
- **Comando:** `uv run python secuencial/benchmark_secuencial.py --runs 5 --epochs 10`
- **Datos:** 12,180 imágenes de entrenamiento y 2,610 de validación (64×64×3, las mismas del notebook).
- **Repeticiones:** 1 corrida de calentamiento descartada (llena la caché de disco de Windows y compila los grafos de TensorFlow) + **5 corridas medidas**, todas en el mismo proceso y condiciones.
- **Épocas fijas:** cada CNN entrena **10 épocas sin EarlyStopping**, para que todas las corridas hagan exactamente el mismo trabajo. El notebook original usa EarlyStopping (CNN 1 paró en 17 épocas, CNN 2 usó 30). Por eso se reporta también el tiempo **por época**, que permite extrapolar.
- **Random Forest:** 300 árboles, `n_jobs=-1`, sobre los píxeles aplanados (la configuración final del notebook).
- **Salidas crudas:** `secuencial/resultados/secuencial_tiempos.csv` (cada corrida), `secuencial_resumen.csv` (media, desviación, mín., máx.), `secuencial_log.txt`, `secuencial_entorno.json`.

### Entorno de medición

| Elemento | Valor |
|---|---|
| CPU | 16 hilos lógicos (`os.cpu_count()` = 16) |
| GPU (no usada en esta fase) | NVIDIA GeForce RTX 5060 Laptop, 8 GB |
| SO | Windows 11 (10.0.26300) |
| Python / TensorFlow | 3.12.8 / 2.21.0 |
| Dispositivos GPU visibles para TensorFlow | ninguno (TensorFlow en Windows nativo no usa GPU) |

## 2.2 Resultados por corrida (segundos)

| Corrida | F1 listado + split | F2 preproc. train | F2 preproc. val | CNN 1 (10 ép.) | CNN 2 (10 ép.) | Inferencia val (CNN 1+2) | Random Forest (300) |
|---|---|---|---|---|---|---|---|
| 1 | 0.92 | 9.13 | 1.98 | 45.27 | 52.37 | 0.81 | 12.46 |
| 2 | 0.61 | 4.87 | 1.05 | 37.63 | 44.11 | 0.72 | 12.85 |
| 3 | 0.62 | 4.67 | 1.00 | 37.26 | 44.18 | 0.73 | 12.75 |
| 4 | 0.64 | 4.74 | 1.00 | 37.03 | 44.05 | 0.72 | 12.88 |
| 5 | 0.61 | 4.95 | 1.08 | 37.47 | 44.73 | 0.74 | 12.91 |

La corrida 1 es un valor atípico (≈20 % más lenta en casi todas las fases). Probablemente fue ruido del sistema justo después del calentamiento y no se descartó para no sesgar el resultado. La mediana es más representativa que la media, y la tabla siguiente muestra ambas.

## 2.3 Tiempo promedio por fase

| Fase | Media (s) | Desv. est. (s) | Mín. (s) | Máx. (s) | % del total |
|---|---|---|---|---|---|
| F1 Listado + split + submuestreo | 0.68 | 0.14 | 0.61 | 0.92 | 0.6 % |
| F2 Preprocesamiento de imágenes (train + val) | 6.89 | 2.35 | 5.67 | 11.11 | 6.5 % |
| **Preparación de datos (F1 + F2)** | **7.57** | 2.49 | 6.29 | 12.03 | **7.2 %** |
| F3 Entrenamiento CNN 1 (10 épocas) | 38.93 | 3.55 | 37.03 | 45.27 | 36.8 % |
| F3 Entrenamiento CNN 2 (10 épocas) | 45.89 | 3.63 | 44.05 | 52.37 | 43.3 % |
| **Entrenamiento de CNN (F3)** | **84.82** | 7.18 | 81.08 | 97.64 | **80.1 %** |
| F5 Inferencia en validación (CNN 1 + CNN 2) | 0.74 | 0.04 | 0.72 | 0.81 | 0.7 % |
| F4 Random Forest (300 árboles) | 12.77 | 0.18 | 12.46 | 12.91 | 12.1 % |
| **Total del pipeline** | **105.90** | 9.54 | 101.05 | 122.94 | 100 % |

### Métricas derivadas

| Métrica | Valor |
|---|---|
| Preprocesamiento por imagen (caché de disco caliente, corridas 2–5) | ≈ 0.39 ms/imagen (≈ 2,500 imágenes/s) |
| Tiempo por época, CNN 1 | 3.89 s (media; 3.70–4.53 s) |
| Tiempo por época, CNN 2 | 4.59 s (media; 4.40–5.24 s) |
| Tiempo por paso de entrenamiento (381 pasos/época) | ≈ 10.2 ms (CNN 1) y ≈ 12.0 ms (CNN 2) |
| Accuracy en validación tras 10 épocas | CNN 1: 0.9226; CNN 2: 0.9667 (idénticas en las 5 corridas, la semilla es fija) |

### Extrapolación a la configuración original del notebook (con EarlyStopping)

| Concepto | Cálculo | Tiempo estimado |
|---|---|---|
| CNN 1, 17 épocas | 17 × 3.89 s | ≈ 66 s |
| CNN 2, 30 épocas | 30 × 4.59 s | ≈ 138 s |
| Preparación de datos | medido | ≈ 8 s |
| **Pipeline con CNN 1 + CNN 2 (sin RF)** | | **≈ 213 s** |

En esa configuración el entrenamiento llega a ≈ 96 % del tiempo y la preparación de datos queda en ≈ 4 %. Con más épocas, la fase de entrenamiento domina todavía más.

## 2.4 Análisis de los tiempos y factores que influyen

1. **El entrenamiento domina el costo.** CNN 1 + CNN 2 consumen ≈ 80 % del tiempo medido (con solo 10 épocas), y la proporción crece con el número de épocas. Es la fase candidata a trasladarse a la GPU.
2. **La preparación de datos es barata aquí (≈ 7 %).** Son ≈ 0.4 ms por imagen con la caché caliente. Esto es un resultado que conviene tener presente: la ganancia de paralelizar el preprocesamiento en CPU será modesta en términos absolutos (unos 7 s), aunque proporcionalmente puede ser grande.
3. **Efecto del disco y la caché.** El preprocesamiento de la corrida 1 tardó ≈ 11 s frente a ≈ 5.7 s en las demás. Como el calentamiento ya había leído el mismo conjunto de archivos, no se puede asegurar que la causa sea la caché; pudo ser ruido del sistema. Aun así, una lectura en frío o un disco más lento cambiaría esta fase, que depende de E/S.
4. **TensorFlow en CPU ya usa varios núcleos.** Aunque el código es "secuencial" a nivel de Python (un bucle de pasos), las operaciones internas (convoluciones, multiplicaciones de matrices) se reparten en los 16 hilos lógicos mediante oneDNN. Esto significa que la línea base **no** es de un solo núcleo, y es un competidor fuerte para la GPU. Su utilización de CPU se registró con `psutil` y con el Administrador de Tareas (sección 5.6): 15 de 16 núcleos por encima de 50 % durante el entrenamiento.
5. **Tamaño pequeño del problema.** Las imágenes son de 64×64 y los modelos tienen ≈ 1.3–1.6 M parámetros con lote 32. Cada paso de entrenamiento dura ≈ 10–12 ms. En la GPU el costo de lanzar *kernels* y de transferir datos puede ser comparable al cómputo útil, lo cual limitará el speedup (se discute en la sección 3).
6. **Otros factores:** el modo de energía del portátil y la temperatura (variación de ≈ 20 % en la corrida 1), procesos en segundo plano, la semilla (fija, por lo que la exactitud es idéntica entre corridas), y el orden aleatorio de los lotes.
7. **Random Forest (12.8 s)** ya está paralelizado en CPU con `n_jobs=-1` y varía poco (desv. 0.18 s). No participará en la comparación CUDA.

## 2.5 Limitaciones de esta medición

- No se incluyó el entrenamiento de las redes densas ni de los modelos con aumento de datos, ni la evaluación en test, para mantener el benchmark enfocado en el pipeline principal. Se pueden añadir.
- Los tiempos de este equipo (portátil con 16 hilos lógicos) difieren de los del notebook original (p. ej. ≈ 12 s por época de la CNN 1 en la máquina del compañero, contra ≈ 3.9 s aquí). Todas las comparaciones CPU vs GPU de este proyecto deben hacerse en **esta misma máquina**.
- La utilización de CPU no se midió en la corrida cronometrada, para no afectar los tiempos con el muestreo; se registró en una corrida aparte (sección 5.6).
