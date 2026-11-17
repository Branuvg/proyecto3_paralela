# 4. Diseño del algoritmo paralelo

Se parte de las operaciones seleccionadas en la sección 3. El diseño separa estrictamente dos fases, como exige el enunciado:

- **Preparación de datos → CPU multinúcleo** (E/S y decodificación JPEG).
- **Matemática de la red neuronal → GPU con CUDA.**

## 4.1 Arquitectura general

```
                    CPU (16 hilos lógicos)                              GPU (RTX 5060, CUDA 12.8)
 ┌──────────────────────────────────────────────┐        ┌─────────────────────────────────────────────┐
 │ F1 listado + split + submuestreo (secuencial)│        │                                             │
 │            │                                 │        │                                             │
 │            ▼ rutas                           │        │                                             │
 │ F2 pool de N procesos ──► abrir + decodificar│        │                                             │
 │    JPEG + RGB + resize 64x64 (por bloques)   │        │                                             │
 │            │ uint8 (N,64,64,3)               │  H2D   │  uint8 en memoria global                    │
 │            └─────────────────────────────────┼───────►│       │ kernel `normalize`                  │
 │                                              │ (1 vez)│       ▼ float32 (N,3,64,64) /255            │
 │ orquestación del bucle de entrenamiento      │        │  entrenamiento CNN (PyTorch/cuDNN):         │
 │ (Python: lanza ≈ 381 pasos por época)        │◄──────►│  conv, ReLU, pool, GEMM, Adam               │
 │                                              │ lanza  │  inferencia: kernels propios (Numba CUDA)   │
 └──────────────────────────────────────────────┘        └─────────────────────────────────────────────┘
```

**Decisiones de diseño y justificación**

| Decisión | Motivo |
|---|---|
| El CPU entrega `uint8` (no `float32`) y la normalización `/255` se hace en la GPU. | La transferencia CPU→GPU baja 4 veces (≈ 150 MB en lugar de ≈ 600 MB para el train) y se usa el paralelismo de la GPU para una operación elemento a elemento. |
| Una sola transferencia H2D del dataset completo; los pesos y los datos permanecen residentes en la GPU. | Evita pagar la transferencia en cada paso de entrenamiento. El dataset (≈ 0.6 GB en `float32`) cabe de sobra en los 8 GB de la GPU. |
| El entrenamiento usa PyTorch (cuDNN/cuBLAS) para las CNN. | Retropropagación, Adam y las convoluciones de entrenamiento ya están optimizadas; escribirlas a mano no es práctico. |
| Los *kernels* propios (Numba CUDA) implementan normalización y toda la **propagación hacia adelante** (conv+ReLU, max-pool, capa densa, argmax). | Cumple el requisito de diseñar la distribución en hilos/bloques/grids, y se **valida** contra PyTorch. |
| Se conservan los mismos hiperparámetros del secuencial: lote 32, Adam (lr 1e-3, ε 1e-7), inicialización Glorot, 10 épocas, `SEED=42`. | Comparación justa: mismo trabajo, mismos datos. |
| El Random Forest no se migra. | scikit-learn no tiene versión CUDA y ya usa todos los núcleos del CPU. |

## 4.2 Paralelismo en CPU: preprocesamiento por bloques (`common/prep.py`)

- Las 14,790 rutas (12,180 train + 2,610 val) se dividen en **bloques de 128 imágenes**.
- Un `ProcessPoolExecutor` con **N procesos** (N = 16 por defecto) reparte los bloques de forma dinámica; cada proceso abre, decodifica, convierte a RGB y redimensiona sus imágenes, y devuelve un arreglo `uint8`.
- Se usan **procesos** y no hilos porque la decodificación con PIL es trabajo de CPU y el GIL de Python limita a los hilos.
- El módulo que importan los procesos hijo (`common/prep_worker.py`) solo carga PIL y NumPy, para que el arranque del pool en Windows (*spawn*) sea rápido.
- El tiempo medido **incluye el arranque del pool** (un pool por ejecución del pipeline).

## 4.3 Paralelismo en GPU: mapeo de trabajo a hilos, bloques y grids

Los kernels están en `gpu/kernels_cuda.py` (Numba CUDA). Los tensores son tensores de PyTorch en GPU, que Numba lee directamente por `__cuda_array_interface__` (sin copias). Notación: `N` = lote, `Cin/Cout` = canales, `OH×OW` = tamaño de la salida.

| Kernel | Hilo = | Bloque | Grid | Memoria compartida |
|---|---|---|---|---|
| `k_normalize` (uint8 NHWC → float32 NCHW, `/255`) | 1 valor de salida (patrón *grid-stride*) | 256 hilos (1D) | `min(⌈N·3·64·64/256⌉, 4096)` bloques (1D) | No |
| `k_conv3x3_relu` (convolución 3×3 + sesgo + ReLU) | 1 píxel de salida `(oy, ox)` de un filtro y una imagen | 16×16 hilos | `(⌈OW/16⌉, ⌈OH/16⌉, N·Cout)` | **Sí:** los `Cin·9` pesos del filtro se cargan cooperativamente |
| `k_maxpool2` (2×2, paso 2) | 1 píxel de salida | 16×16 | `(⌈OW/16⌉, ⌈OH/16⌉, N·C)` | No |
| `k_dense` (`y = x·Wᵀ + b`, ReLU opcional) | 1 elemento de la matriz resultado | 16×16 | `(⌈Nout/16⌉, ⌈M/16⌉)` | **Sí:** GEMM por teselas 16×16 de `x` y `W` |
| `k_argmax` (clase predicha) | 1 muestra (29 clases) | 128 (1D) | `⌈N/128⌉` | No |

### Ejemplo: la primera convolución de la CNN 1 con un lote de 256 imágenes

- Salida: `256 × 32 × 62 × 62` = 31.5 millones de valores independientes.
- Grid: `(⌈62/16⌉, ⌈62/16⌉, 256·32)` = `(4, 4, 8192)` → 131,072 bloques de 256 hilos = **33.5 millones de hilos**. Cada hilo calcula 27 productos-acumulados (3 canales × 3 × 3).
- Los pesos del filtro (27 valores) se leen una vez por bloque a memoria compartida y se reutilizan por los 256 hilos.
- Los hilos del borde (`ox ≥ OW` o `oy ≥ OH`, porque 62 no es múltiplo de 16) se desactivan con una guarda.

### Justificación de las decisiones de mapeo

- **Bloques de 16×16 = 256 hilos:** múltiplo del tamaño del *warp* (32), buena ocupación, y las teselas del GEMM son cuadradas.
- **El eje `z` del grid codifica `(imagen, canal)`:** hace que cada bloque trabaje en un único filtro, de modo que los pesos puedan guardarse en memoria compartida. Límite: `N·Cout ≤ 65,535` (con lote 256 y 128 filtros son 32,768, dentro del límite).
- **Accesos coalescidos:** hilos consecutivos en `x` (`threadIdx.x`) leen y escriben posiciones consecutivas del último eje (`ox`) de los tensores NCHW.
- **GEMM por teselas:** reduce las lecturas a memoria global por un factor de 16 al reutilizar cada tesela de `x` y `W` 16 veces desde memoria compartida, con `__syncthreads` entre la carga y el uso.
- **Normalización con *grid-stride*:** el número de elementos (≈ 150 millones para el train) excede lo razonable para un bloque por cada uno; un grid fijo recorre el arreglo en saltos.

## 4.4 Entrenamiento (PyTorch)

Por época se barajan los índices (en GPU), se toman mini-lotes de 32 y, para cada uno: propagación hacia adelante, entropía cruzada, retropropagación y paso de Adam. Al final de cada época se evalúa en validación (igual que `model.fit(validation_data=…)` de Keras). Los 381 pasos por época **siguen siendo secuenciales** (dependencia de pesos); el paralelismo ocurre dentro de cada paso.

## 4.5 Estrategia de validación y medición

1. **Corrección de los kernels:** la salida de los kernels propios se compara con la de PyTorch (diferencia máxima de logits y coincidencia de la clase predicha). Ver sección 5.
2. **Equivalencia de los resultados:** accuracy de validación de CNN 1 y CNN 2 vs. el secuencial (mismos datos y semilla; los valores no serán idénticos porque los frameworks difieren en detalles numéricos).
3. **Medición:** mismas fases, mismas épocas, 1 corrida de calentamiento descartada + 5 corridas medidas, con `torch.cuda.synchronize()` antes y después de cada fase y **las transferencias CPU↔GPU incluidas** en el tiempo total.
4. **Referencia adicional:** el mismo código PyTorch ejecutado en CPU (`--device cpu`), para separar el efecto del *framework* del efecto de la GPU.
5. **Evidencia de recursos:** monitor de CPU por núcleo (`psutil`) y GPU (`nvidia-smi`) con marcas por fase (`common/monitor.py`).

## 4.6 Limitaciones conocidas del diseño

- Los kernels propios cubren solo la **propagación hacia adelante**. La retropropagación y Adam se delegan a PyTorch.
- La convolución propia es directa (sin transformada de Winograd/FFT ni *tiling* de la entrada en memoria compartida), por lo que se espera que sea más lenta que cuDNN. Su valor aquí es de diseño, validación y aprendizaje.
- Los dos tipos de cómputo (CPU y GPU) se ejecutan en fases consecutivas y no solapadas, lo que mantiene la medición simple. Solapar la preparación del siguiente bloque de datos con el entrenamiento es una mejora futura.
