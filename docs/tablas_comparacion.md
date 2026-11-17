| Fase | Secuencial (TF, CPU) s | Paralelo (CPU multiproc. / GPU) s | Speedup | PyTorch en CPU s |
|---|---|---|---|---|
| F1 Listado + split (CPU, igual) | 0.68 ± 0.14 | 0.62 ± 0.03 | **1.09×** | 0.73 ± 0.08 |
| F2 Preprocesamiento train+val | 6.89 ± 2.36 | 0.99 ± 0.01 | **6.95×** | 1.02 ± 0.04 |
| F3 Entrenamiento CNN 1 (10 ép.) | 38.93 ± 3.55 | 11.37 ± 0.08 | **3.42×** | 69.91 ± 13.46 |
| F3 Entrenamiento CNN 2 (10 ép.) | 45.89 ± 3.63 | 12.76 ± 0.07 | **3.60×** | 92.38 ± 11.65 |
| T1+N Transferencia H2D + normalización GPU | — | 0.019 ± 0.003 | — | 0.060 |
| F5 Inferencia val. (CNN 1+2) | 0.740 | PyTorch GPU 0.195 / kernels propios 0.362 | 3.79× / 2.04× | 1.151 |
| **Total pipeline CNN (sin Random Forest)** | **93.13 ± 9.71** | **25.96 ± 0.12** | **3.59×** | 165.26 ± 25.24 |

| Agrupación | Secuencial s | Paralelo s | Speedup |
|---|---|---|---|
| Preparación de datos (F1+F2) | 7.57 | 1.62 | 4.69× |
| Entrenamiento (CNN 1+2) | 84.82 | 24.13 | 3.51× |
| Total | 93.13 | 25.96 | 3.59× |

| Por época | Secuencial (TF CPU) | PyTorch CPU | PyTorch GPU | Speedup GPU vs TF CPU | Speedup GPU vs PyTorch CPU |
|---|---|---|---|---|---|
| CNN1 | 3.89 s | 6.99 s | 1.14 s | 3.42× | 6.15× |
| CNN2 | 4.59 s | 9.24 s | 1.28 s | 3.60× | 7.24× |

| Accuracy val. (10 ép.) | Secuencial TF | PyTorch GPU | PyTorch CPU |
|---|---|---|---|
| CNN1 | 0.9226 | 0.9066 ± 0.0147 | 0.9046 |
| CNN2 | 0.9667 | 0.9701 ± 0.0053 | 0.9625 |

| Corrida | Total secuencial s | Total paralelo s | Speedup |
|---|---|---|---|
| 1 | 110.47 | 25.78 | 4.29× |
| 2 | 88.98 | 25.91 | 3.43× |
| 3 | 88.45 | 26.04 | 3.40× |
| 4 | 88.18 | 25.99 | 3.39× |
| 5 | 89.58 | 26.10 | 3.43× |

| Procesos | Tiempo (s) | Speedup | Eficiencia |
|---|---|---|---|
| 1 | 5.04 ± 0.02 | 1.00× | 100% |
| 2 | 2.89 ± 0.02 | 1.74× | 87% |
| 4 | 1.67 ± 0.02 | 3.03× | 76% |
| 8 | 1.17 ± 0.00 | 4.31× | 54% |
| 12 | 1.00 ± 0.01 | 5.03× | 42% |
| 16 | 1.00 ± 0.02 | 5.06× | 32% |

**Monitor (secuencial)**

| Fase | Muestras | CPU prom % | Núcleos >50 % | GPU util prom % | GPU util máx % | Mem GPU máx MiB | Potencia GPU W |
|---|---|---|---|---|---|---|---|
| F1 listado+split | 1 | 21 | 2.0 | 6 | 6 | 382 | 7 |
| F2 preproc secuencial | 21 | 18 | 2.0 | 4 | 11 | 382 | 7 |
| F3 entrenamiento CNN1 | 88 | 69 | 15.0 | 5 | 34 | 449 | 8 |
| F3 entrenamiento CNN2 | 102 | 74 | 15.2 | 5 | 31 | 421 | 8 |
| F4 Random Forest | 25 | 97 | 15.6 | 1 | 6 | 421 | 8 |

**Monitor (paralelo)**

| Fase | Muestras | CPU prom % | Núcleos >50 % | GPU util prom % | GPU util máx % | Mem GPU máx MiB | Potencia GPU W |
|---|---|---|---|---|---|---|---|
| F1 listado+split | 5 | 9 | 1.0 | 0 | 0 | 1711 | 47 |
| F2 preproc CPU paralelo | 10 | 72 | 12.8 | 0 | 0 | 1711 | 16 |
| F3 entrenamiento CNN1 | 103 | 10 | 1.3 | 96 | 99 | 1711 | 59 |
| F5 inferencia CNN1 | 2 | 9 | 1.5 | 98 | 99 | 1711 | 60 |
| F3 entrenamiento CNN2 | 115 | 10 | 1.4 | 95 | 99 | 1711 | 60 |
| F5 inferencia CNN2 | 4 | 12 | 1.8 | 99 | 99 | 1711 | 60 |
