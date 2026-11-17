# 3. Identificación de oportunidades de paralelización

Basada en el análisis de la sección 1 y en los tiempos medidos de la sección 2. La asignación sigue la arquitectura que pide el proyecto: **CPU multinúcleo para la preparación de imágenes, GPU/CUDA para la matemática de la red neuronal**.

## 3.1 Evaluación operación por operación

Criterios: ¿hay independencia entre los elementos?, ¿qué tipo de cómputo es (E/S o aritmética)?, ¿qué fracción del tiempo representa?

| No. | Operación | % del tiempo medido | Independencia de los datos | ¿Candidata a CUDA? | Dónde conviene ejecutarla |
|-----|-----------|---|---|---|---|
| 2, 4, 5 | Listado, split y submuestreo | 0.6 % | Operaciones sobre listas/tablas pequeñas | No. Es trivial y se ejecuta una vez | CPU (secuencial) |
| 6 | **Preprocesamiento por imagen** (abrir, decodificar JPEG, resize, normalizar) | 6.5 % | **Total.** Cada imagen es independiente | Parcial: el *resize* y la normalización sí, pero **abrir el archivo y decodificar el JPEG** son E/S y lógica irregular (Huffman) que no se adaptan a la GPU | **CPU con varios procesos o hilos**, mientras la GPU trabaja |
| 7 | Codificación de etiquetas | < 0.1 % | Trivial | No | CPU |
| 10 | **Entrenamiento de las CNN** (convoluciones, capas densas, retropropagación, Adam) | **80 %** (96 % con las épocas del notebook) | Dentro de un paso: alta (entre imágenes del lote, filtros y píxeles de salida). Entre pasos: **dependiente** (los pesos se actualizan) | **Sí. Es el objetivo principal** | **GPU** |
| 11 | Evaluación / inferencia | 0.7 % | Alta entre imágenes | **Sí** (mismas operaciones que la propagación hacia adelante) | GPU |
| 12 | Redes densas | n/m (no medido) | Alta (GEMM) | Sí, mismo mecanismo | GPU |
| 13 | Random Forest | 12 % | Árboles independientes | **No con CUDA en este proyecto.** scikit-learn no tiene versión GPU y ya usa `n_jobs=-1` | CPU |
| 14 | Aumento de datos (rotar, zoom, trasladar, contraste) | n/m | Total: cada imagen | **Sí**, si se hace por lote en la GPU (evita transferir imágenes aumentadas) | GPU |

## 3.2 Operaciones seleccionadas para paralelizar y justificación

### A. Entrenamiento y evaluación de la CNN en la GPU (la principal)

**Qué se paraleliza**
- **Convolución 3×3** (`Conv2D`): cada valor de la salida (imagen, filtro, fila, columna) es una suma de 27 a 576 productos que **no depende de ningún otro valor de salida**. En la CNN 1, la primera capa produce 62 × 62 × 32 = 123,008 valores por imagen, y en un lote de 32 imágenes son ≈ 3.9 millones de resultados independientes.
- **Capas densas** (multiplicación de matrices): lote (32) × entradas (12,544 o 4,608) por pesos (128 o 256); cada elemento de la matriz resultante es independiente.
- **MaxPooling, ReLU y softmax**: operaciones elemento a elemento o por ventana, trivialmente paralelas.
- **Retropropagación**: los gradientes de las capas son otras convoluciones y multiplicaciones de matrices del mismo tipo (≈ 2 veces el costo de la propagación hacia adelante).

**Por qué se beneficia de la GPU**
- Es aritmética de punto flotante densa, con **alta intensidad aritmética** (muchas operaciones por dato leído) y control de flujo regular, que es el caso para el que están diseñados los miles de núcleos CUDA.
- El volumen de cómputo es grande: ≈ 2.3×10¹¹ MACs por época solo en las convoluciones de la CNN 1 (sección 1.3).
- Los pesos y los datos pueden **permanecer en la memoria de la GPU** durante todo el entrenamiento, así que la transferencia CPU→GPU se paga una vez y no en cada paso.

**Límite de la dependencia entre pasos:** no se pueden ejecutar en paralelo los pasos sucesivos de un mismo entrenamiento. El paralelismo se aprovecha *dentro* de cada paso.

### B. Preprocesamiento de imágenes en CPU multinúcleo

**Qué se paraleliza:** el bucle `for p in filepath: load_and_preprocess(p)` reparte las rutas entre N procesos (por ejemplo `multiprocessing.Pool` o `ProcessPoolExecutor`, porque la decodificación con PIL es trabajo de CPU y el GIL limita a los hilos en las partes que no liberan el GIL).

**Por qué en CPU y no en GPU**
- La lectura del archivo y la **decodificación JPEG** son E/S y lógica de ramificación irregular (decodificación Huffman, difícil de mapear a la GPU).
- Los 16 hilos lógicos del CPU pueden hacerlo en paralelo mientras la GPU entrena, lo que permite **solapar** preparación y entrenamiento.
- Enviar los JPG a la GPU no ahorra nada si luego hay que decodificarlos en CPU de todos modos.

**Beneficio esperado:** limitado en términos absolutos (la fase vale ≈ 7 s de ≈ 106 s). Es un candidato válido porque es el paralelismo de datos más puro del programa y porque el proyecto exige separar esta fase del cómputo en GPU. Con datasets más grandes (las 87,000 imágenes) el beneficio crecería proporcionalmente.

### C. Aumento de datos en GPU (opcional)

La transformación por imagen es independiente y de tipo aritmético, por lo que puede hacerse en la GPU sobre el lote ya transferido. Se incluiría solo si se reentrenan los modelos aumentados.

## 3.3 Operaciones que se mantienen en CPU

| Operación | Motivo |
|---|---|
| Listado, split, submuestreo, etiquetas | Costo despreciable (≈ 0.7 s); no justifican el costo de transferir. |
| Lectura de archivos y decodificación JPEG | E/S y lógica irregular; ver B. |
| Random Forest | Sin versión CUDA en scikit-learn y ya paralelo en CPU. |
| Orquestación del bucle de entrenamiento y de métricas | Control de flujo secuencial. |

## 3.4 Qué tan grande puede ser la mejora: ley de Amdahl

Con los porcentajes medidos (sección 2), para el pipeline de CNN (sin Random Forest, 10 épocas): la preparación ≈ 8.1 % (7.57 s de 93.1 s) y el entrenamiento + inferencia ≈ 91.9 %.

| Escenario | Cálculo | Speedup máximo del pipeline |
|---|---|---|
| GPU acelera el entrenamiento ×2 | 1 / (0.081 + 0.919/2) | ≈ 1.85× |
| GPU acelera el entrenamiento ×5 | 1 / (0.081 + 0.919/5) | ≈ 3.8× |
| GPU acelera el entrenamiento ×10 | 1 / (0.081 + 0.919/10) | ≈ 5.8× |
| Entrenamiento infinitamente rápido | 1 / 0.081 | ≈ 12.3× |

El techo lo fija la fracción que sigue en CPU, y por eso **también interesa acelerar la preparación** (B). Si además el preprocesamiento se paraleliza ×8, la fracción restante baja a ≈ 1 % y el techo sube considerablemente.

## 3.5 Riesgos y factores que pueden limitar la ganancia (a verificar en las secciones 5 y 6)

1. **Línea base fuerte:** TensorFlow en CPU ya usa los 16 hilos lógicos y un paso dura solo ≈ 10–12 ms.
2. **Problema pequeño:** 64×64 con lote 32 y ≈ 1.3 M parámetros puede no saturar la GPU. Tamaños de lote mayores favorecerían a la GPU, pero cambian la dinámica del entrenamiento (hay que anotar el cambio si se hace).
3. **Costo fijo:** lanzamiento de *kernels*, sincronización y la transferencia inicial de ≈ 12,180 × 12,288 × 4 B ≈ 0.6 GB (train) a la GPU.
4. **Entorno:** TensorFlow no usa GPU en Windows nativo, y la RTX 5060 requiere CUDA 12.8 o superior. Hay que decidir la librería (PyTorch con CUDA, Numba/CuPy para *kernels* propios, o TensorFlow en WSL2) antes de la sección 4. Para cumplir con el diseño de hilos/bloques/grids que pide el paso 4, conviene incluir al menos algunos *kernels* escritos a mano (por ejemplo, convolución, ReLU, MaxPool o el preprocesamiento por lote) además del entrenamiento con un *framework*.
5. **Comparación justa:** misma semilla, mismos datos y mismas épocas; medir y reportar por separado la transferencia CPU↔GPU.

## 3.6 Resumen de la propuesta

| Fase | Hoy (secuencial) | Propuesta |
|---|---|---|
| Preparación de imágenes | Un solo proceso en CPU | **CPU multinúcleo** (varios procesos) |
| Entrenamiento de CNN 1 y CNN 2 | TensorFlow en CPU | **GPU con CUDA** (datos y pesos residentes en GPU) |
| Inferencia/evaluación | CPU | **GPU** |
| Random Forest | CPU con `n_jobs=-1` | Se queda igual (fuera de alcance de CUDA) |

**Decisión pendiente para la sección 4:** la librería CUDA a usar y el alcance de los *kernels* propios.
