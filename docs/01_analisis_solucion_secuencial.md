# 1. Análisis de la solución secuencial existente

**Fuente:** `notebooks/01_eda_preprocessing.ipynb` (Laboratorio 3 de Data Science, repo `Laboratorio3-ds`).
Es la solución base. Se eligió sobre la otra versión del equipo (`DS-Lab1`) por mayor accuracy (98.7 % vs 95.5 % en test), entorno reproducible con `uv` y porque ya trae informe.

## 1.1 Resumen de la solución

Clasificador de las 29 clases del alfabeto ASL (A–Z, `space`, `del`, `nothing`) a partir de fotos de manos del dataset *ASL Alphabet* (Kaggle): ~87,000 imágenes JPG de 200×200 RGB, 3,000 por clase.

Parámetros fijos del pipeline: `SEED=42`, `N_PER_CLASS=600`, `IMG_SIZE=64`, batch 32, optimizador Adam, EarlyStopping sobre `val_loss` (paciencia 5).

Todo el cómputo corre en CPU. Los modelos usan Keras/TensorFlow, y el propio notebook registra que TensorFlow en Windows nativo **no usa GPU** (`failed call to cuInit`). Esta es la línea base secuencial real.

## 1.2 Programa descompuesto en operaciones

| No. | Operación identificada | Descripción de su funcionamiento |
|-----|------------------------|----------------------------------|
| 1 | Descarga del dataset | `kagglehub.dataset_download` baja y cachea ~1 GB en disco. Se hace una sola vez. |
| 2 | Listado de archivos | `os.listdir` recorre las 29 carpetas y arma un `DataFrame` (`filepath`, `label`) con ~87,000 filas. |
| 3 | EDA | Muestras visuales, conteo por clase, formato/resolución, brillo medio por clase (`mean_brightness`), grupos de letras confundibles. Es análisis exploratorio, no es parte del pipeline de predicción. |
| 4 | Split estratificado 70/15/15 | `train_test_split` dos veces (stratify por clase, `random_state=42`) → 60,900 / 13,050 / 13,050 rutas. |
| 5 | Submuestreo estratificado | `stratified_subsample`: 420 / 90 / 90 imágenes por clase → 12,180 train, 2,610 val, 2,610 test. |
| 6 | Carga y preprocesamiento de imágenes | `load_and_preprocess` por imagen: abrir JPG → decodificar → `convert("RGB")` → `resize` 64×64 bilineal → `float32` → dividir entre 255. Un bucle `for` secuencial (comprensión de lista + `np.stack`). |
| 7 | Codificación de etiquetas | `LabelEncoder` (ajustado solo con train) + `to_categorical` (one-hot de 29 dimensiones). |
| 8 | Construcción de la CNN 1 | `Conv2D(32)→MaxPool→Conv2D(64)→MaxPool→Flatten→Dense(128)→Dense(29, softmax)`. 1,628,893 parámetros. |
| 9 | Construcción de la CNN 2 | Tres bloques `Conv2D(32/64/128)+MaxPool`, `Dense(256)`, `Dropout(0.5)`, `Dense(29)`. 1,280,605 parámetros. |
| 10 | Entrenamiento de las CNN | `model.fit`: por época recorre el train en mini-lotes de 32 (381 pasos). Cada paso hace propagación hacia adelante, pérdida (entropía cruzada categórica), retropropagación y actualización Adam. |
| 11 | Evaluación | `model.evaluate` en validación/test (propagación hacia adelante sin gradiente) + matriz de confusión. |
| 12 | Redes densas (3 configuraciones) | `Flatten` (12,288 entradas) + capas `Dense`. Se entrenan con la misma mecánica que las CNN. Las configuraciones sin dropout no aprendieron (accuracy ≈ 3.4 %). |
| 13 | Random Forest | `RandomForestClassifier` sobre los píxeles aplanados (12,288 variables), con 100, 300 y 500 árboles, `n_jobs=-1` (ya usa todos los núcleos del CPU). |
| 14 | Aumento de datos | Capa de Keras con rotación ±4 %, zoom ±8 %, traslación ±8 % y contraste ±10 % (sin flips), insertada al inicio de la CNN. Se aplica en cada paso de entrenamiento. Se reentrenan CNN 1 y CNN 2. |
| 15 | Prueba con fotos propias | Las 15 fotos del equipo pasan por el mismo `load_and_preprocess` (con opción de preservar la relación de aspecto) y por el mejor modelo. |

### Flujo de datos

```
disco (87,000 JPG) ──► listado/split/submuestreo (rutas)
        │
        ▼
 [bucle secuencial PIL: abrir+decodificar+resize+normalizar]  ──► X: (N,64,64,3) float32
        │                                                          y: (N,29) one-hot
        ▼
 model.fit (mini-lotes de 32, CPU/TensorFlow) ──► pesos entrenados
        │
        ▼
 model.evaluate / predict ──► accuracy, matriz de confusión
```

Hay dos fases con perfiles de cómputo distintos: **preparación de datos** (operación 6, dominada por E/S de disco y decodificación JPEG) y **cómputo del modelo** (operaciones 10–14, dominado por aritmética de punto flotante).

## 1.3 Operaciones repetitivas

Las cantidades se derivan del notebook y de su configuración. Los factores de cómputo son MACs (multiplicaciones-acumulaciones) por imagen en una propagación hacia adelante, calculados a partir de las arquitecturas. El entrenamiento cuesta aproximadamente 3 veces la propagación hacia adelante (adelante + atrás + gradientes).

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
|----------------------|-------------------|--------------------------|
| Abrir + decodificar + `convert` + `resize` + normalizar una imagen (`load_and_preprocess`) | Una imagen JPG 200×200×3 (120,000 valores) → 64×64×3 (12,288 valores) | 12,180 (train) + 2,610 (val) + 2,610 (test) = **17,400** por ejecución del pipeline (+15 fotos propias) |
| Recorrido de carpetas con `os.listdir` | 29 carpetas con 3,000 nombres | 29 listados; ~87,000 filas del `DataFrame` |
| Paso de entrenamiento de una CNN (mini-lote) | Lote de 32 imágenes 64×64×3 | **381 pasos por época**; CNN 1 paró en 17 épocas y CNN 2 usó las 30 → 6,477 y 11,430 pasos. A esto se suman 2 modelos aumentados y las redes densas. |
| Convolución 3×3 (capa `Conv2D`) | Mapa de características completo por cada filtro (p. ej. 62×62×32 en la primera capa de la CNN 1) | Por imagen y por paso: CNN 1 ≈ 18.8 M MACs en convoluciones; CNN 2 ≈ 29.4 M MACs. Por época (12,180 imágenes): ≈ 2.3×10¹¹ y 3.6×10¹¹ MACs solo hacia adelante. |
| Capa densa (multiplicación matriz-vector/matriz) | Vector aplanado de 12,544 (CNN 1) o 4,608 (CNN 2) valores | 1 por imagen y por paso (1.6 M MACs en CNN 1; 1.2 M en CNN 2) |
| Evaluación (propagación hacia adelante) | 2,610 imágenes de validación/test | 1 por modelo y por evaluación (varias veces en el notebook) |
| Entrenamiento de un árbol de Random Forest | 12,180 muestras × 12,288 variables | **100 + 300 + 500 + 300 (modelo final) = 1,200 árboles** en la búsqueda del notebook |
| Aumento de datos | Cada imagen de cada lote | 12,180 × épocas transformaciones por modelo aumentado (aleatorias, en cada paso) |

## 1.4 Hallazgos clave del análisis

1. **Hay dos tipos de carga muy distintos.** La preparación de imágenes (E/S + decodificación JPEG) y el entrenamiento (álgebra lineal densa sobre tensores). El proyecto pide justamente separar ambas.
2. **El bucle de preprocesamiento es embarazosamente paralelo:** cada imagen es independiente de las demás (no comparte estado ni depende de resultados previos).
3. **El entrenamiento es la parte más pesada** y su núcleo son convoluciones y multiplicaciones de matrices sobre lotes, el patrón típico donde la GPU rinde mejor. Los pasos sucesivos de un mismo entrenamiento sí dependen entre sí (los pesos se actualizan paso a paso), así que el paralelismo está *dentro* de cada paso (entre imágenes del lote, filtros y píxeles), no *entre* pasos.
4. **El Random Forest ya usa todos los núcleos del CPU** (`n_jobs=-1`) y scikit-learn no tiene versión CUDA. Queda fuera del alcance de CUDA, pero sirve como referencia.
5. **TensorFlow en Windows nativo no usa GPU**, así que la línea base es CPU y cualquier migración a GPU requiere otra librería o entorno (decisión de la sección 3).
6. El split se calcula sobre las ~87,000 rutas y luego se submuestrea. La cantidad de datos realmente usados es 17,400 imágenes (20 % del dataset).
