# Control de la receta y diagnóstico del deterioro

El control con150pasos de datos originales conservó el MAE histórico: **20,287
frente a20,302** del checkpoint inicial. Los dos pilotos mixtos siguen rechazados.
Este resultado permite centrar el siguiente diagnóstico en cómo se incorporó la
supervisión nueva; no demuestra que las fotos sean malas ni identifica por sí
solo una pérdida o hiperparámetro causal.

## Control original-only150

Se fijó [SPEC007](../sdd/007-controles-entrenamiento/SPEC.md), se revisó el código
y se aprobaron siete pruebas CPU antes de la GPU. El control usó exactamente el
orden de las150exposiciones originales del piloto mixto de200pasos:16recortes
cíclicos de8secuencias train. Mismo checkpoint original, semilla20261025, cabeza
den1 entrenable129parámetros, Adam nuevo LR1e-5, lote1 y clipping1.

Al quitar50pasos nuevos cambian el reloj y los momentos Adam. No se igualan a la
vez exposiciones y actualizaciones; no se ejecutó original-only200 ni control
conv4. Sólo cambiaron los dos tensores de la cabeza; el resto quedó igual por hash.
La suma de conteos de train16 pasó de300,843 a297,169: razón0,98779, dentro de la
banda previa0,5–1,5. Es un guard de estabilidad, no evaluación de generalización.

| Pesos | MAE,30 imágenes | RMSE | Sesgo |
| --- | ---: | ---: | ---: |
| Original | 20,3018 | 23,8159 | −3,2176 |
| Original-only150, cabeza | 20,2872 | 23,9490 | −4,7126 |
| Mixto200, cabeza | 35,0225 | 42,0966 | +33,9497 |
| Mixto200, conv4+cabeza | 112,8739 | 118,9455 | +112,8739 |

El control150 cumple el umbral previo MAE24,3621808, con RMSE y sesgo algo peores:
no se presenta como mejora. La cohorte de30imágenes es test oficial ya observado,
sin participación en gradientes ni selección de checkpoints de este control.
Una sola evaluación final del checkpoint150; ninguna búsqueda de hiperparámetros.
Sigue sin validar conteo local de una manifestación.

## Gradientes iniciales sin actualizar pesos

Se midieron los129componentes en los mismos16ejemplos originales y4nuevos del
piloto congelado v2, antes de usar correcciones v3/v4. El estado completo quedó
idéntico por hash. Los positivos parciales dieron normas3,156 y4,117, reducidas
a1 por clipping; sus componentes fueron todos no positivos. Los originales
variaron entre0,0052 y0,4992; los dos negativos nuevos dieron0,0168 y0,0251.

| Fuente | Frecuencia del mixto | Norma de suma vectorial inicial | Tras clipping por ejemplo |
| --- | ---: | ---: | ---: |
| Originales | 150 | 5,237 | 5,237 |
| Positivos parciales | 26 | 94,512 | 25,989 |
| Negativos nuevos | 24 | 0,498 | 0,498 |

Estos vectores incorporan direcciones y cancelaciones en el punto inicial;
**no son porcentajes de influencia ni pasos de Adam**. La pérdida parcial y las
features no negativas producen presión ascendente bajo descenso simple; Adam
mezclado acumula otra dinámica. El diagnóstico respalda que frecuencia1:3 no
garantizaba equilibrio. No distribuye causalidad entre kernel, máscara, contexto,
ponderación, optimizador o calidad de las etiquetas.

## Comparación espacial de blancos sobre GT completo

Se verificaron hashes de8imágenes/MAT originales y coincidencia exacta entre sus
coordenadas completas y replay. Los mapas adaptativos reconstruidos coincidieron
bit a bit con el auditor histórico en las8fuentes. Se comparó con sigma fijo2
manteniendo iguales rejilla booleana, truncamiento de coordenadas, soporte finito
y pérdida de masa en bordes; los16recortes se extrajeron después del mapa completo.
No se escribieron HDF5 ni se usaron vecinos de anotaciones parciales.

Las medianas de sigma histórico por fuente fueron3,00–9,16celdas de salida, frente
a2 fijo. La mediana por fuente de pico aislado adaptativo/fijo fue0,048–0,444;
hay también cabezas históricas con sigma menor que2 y relación de pico mayor que1.
Por eso no vale afirmar que el kernel fijo sea siempre más estrecho. El coseno
espacial entre mapas completos estuvo entre0,477 y0,861: la distribución cambia
apreciablemente aun conservando convenciones de generación.

La comparación aísla sigma sobre originales completos. No reproduce todos los
blancos nuevos, que además normalizan bordes y preservan puntos coincidentes.
No prueba incapacidad del modelo para generar picos ni calidad superior de un
kernel. La respuesta espacial aprendida y una pérdida alternativa requieren otro
ensayo con hipótesis y criterios propios.

## Recursos, reproducción y evidencia

GPU RTX3060, ejecuciones secuenciales con límite2GiB RAM,512MiB swap y120segundos;
reservas previas20GiB disco/2GiB RAM. Gradientes:2,48s, RSS1,413GiB, CUDA234MiB.
Control150:9,84s, RSS1,683GiB, CUDA235MiB. Retención:9,96s, RSS1,002GiB,
CUDA426MiB. Kernels:CPU0,30s. Ninguna expansión de cuotas ni regeneración masiva.

Entrenamiento y gradientes se ejecutaron con `ops/python-ml.sh`, PyTorch2.14.1+cu130,
Python3.13.13 y semilla20261025. El diagnóstico de kernels usa Python con
NumPy/SciPy/Pillow, sin Torch: Python3.13.13, NumPy2.5.4, SciPy1.18.1 y
Pillow13.0.0 en el entorno CPU usado. Temporales y cachés en `~/home.codex/tmp`.

Entradas congeladas: replay
`f929b0209cc7e9879e9627eebb3a2b51a61e5ec7b7fe6f0c22081b4afbb420e7`,
controles nuevos v2
`098fe0d33e0ff16cb4337955c99bf9b7405f7a84388cbb69d5433aee33196002`,
cohorte histórica
`f94dd4a09e014b57cae0b365fd4dbbccea8598599157f18e40a718521d9e2f8e`.
Checkpoint final150:
`0af9bbc6a06e7d126646800d454650f9a1dda60e8f7aa51fdb48608f39514898`.

| Informe privado | SHA-256 |
| --- | --- |
| `outputs/ajuste-dominio/diagnostico-inicial/report.json` | `14248ae04a5c31f7ed3f56d87b800ae0af42adf1afcce5008633007b3bb31620` |
| `outputs/ajuste-dominio/control-original150/training.json` | `d51348775ba6233ee0f04b67743d2e72a6d2312b6edb30f619bb3ce0c3d06b43` |
| `outputs/ajuste-dominio/control-original150-retencion/evaluation.json` | `608047b59a247b68af3a272eaefba4a6464f6f096e9530493435c1a4dbebc710` |
| `outputs/ajuste-dominio/diagnostico-kernels/report.json` | `c741fa73c00bab56c1cffc0c93682cae6e854a794b5c8645bfc0d0503fafb8ff` |

Los JSON registran código, entradas y configuración; pesos y datos quedan privados.
Reproducción: `diagnose_domain_gradients.py` con replay/controls/checkpoint y sus
hashes; `train_domain_pilot.py --original-only150 --steps 150` con replay/checkpoint;
`evaluate_calibration.py --mode quadrants` índices0–29; `diagnose_target_kernels.py`
con replay y hash. Todos requieren directorio de salida nuevo y las cuotas anteriores.

Implementación:Luna; revisión independiente:Codex, modelo no identificado.
El revisor comprobó orden, hashes, métricas y límites de interpretación del
control150 y de los gradientes. La comparación de kernels tuvo además pruebas
de bordes, colisiones y equivalencia con el auditor antes de ejecutarse.

Siguiente decisión: conservar el original; completar regiones anotadas, definir
la máscara del área de interés y aislar el objetivo antes de otro ajuste. Los
candidatos bibliográficos pueden auditarse por separado; no heredan esta receta.
