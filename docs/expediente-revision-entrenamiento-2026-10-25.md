# Expediente para revisión crítica del entrenamiento STNNet

Fecha: 25/10/2026. Base de código y resultados anteriores: commit `274253f`.
Solicitud del autor: revisar si el entrenamiento estuvo bien diseñado, pedir
dictámenes a Astra y Fable y hacerlos discutir sobre evidencia común.

Este documento reúne los hechos del piloto. Las secciones de hipótesis no son
causas demostradas. Se pide evaluar también las decisiones del coordinador y de
las revisiones previas: que el código cumpla una SPEC no valida esa SPEC.

## 1. Pregunta y alcance real

Objetivo práctico: estimar personas en videos aéreos de una marcha usando STNNet,
y explorar si nuevas marcas de cabezas más replay del dataset original ayudan.
Se dispone de videos y fotos del mismo acto, sin anotaciones humanas completas
para certificar precisión. Se evaluaron resolución nativa, interpolación,
controles con MAT originales y referencias visuales provisionales antes de
entrenar. Las cifras globales de video no son GT ni cantidades únicas de asistentes.

Se hicieron dos pilotos de 200 pasos, uno con129 parámetros y otro con147.713.
No fue un reentrenamiento completo ni una reproducción del entrenamiento publicado.
Ambos empeoraron los controles y se rechazaron. Esto permite rechazar esos pesos;
no permite concluir que el ajuste de dominio sea inviable o que el diseño fuera
adecuado. Los pesos originales no se modificaron.

## 2. Modelo, procedencia y entorno

Arquitectura local: `STNNet/stnmodel.py`, modo `use_loc=False,use_trk=False`,
inicialización offline y carga estricta de nombres, formas, tipos y finitud.
Se cargan41 tensores. La salida principal es den1 a stride2; no se decodifican
detecciones o trayectorias. Normalización RGB/ImageNet.

El archivo `models.zip` fue aportado por el usuario desde el enlace del README;
se verificaron CRC y SHA-256. El checkpoint den se identifica por hash exacto;
se utiliza `torch.load(weights_only=True)`. Para los dos hashes oficiales
inspeccionados existe una excepción mínima y temporal para metadata NumPy antigua;
no se habilita deserialización arbitraria.

| Artefacto | SHA-256 |
| --- | --- |
| ZIP original | `2caa2483e1ea237b7cacf86df7ed7cbb30edb36c4b650ee74d8f643ae9f205c0` |
| Den original | `be407d2dcdee6775a342c343523cd17399310d91acb548172d8f7000606d501d` |
| Piloto cabeza | `52ce16a4854041edae67b549388ada82b07f9d686c7b5722f3176e1f91cf2a7c` |
| Piloto conv4+cabeza | `81a5218cd60ec5967c73a4b6b87e60db5af37ed524bf5e3e05e7e346c851e1e8` |

Los checkpoints experimentales contienen `state_dict` y metadata nativa del
piloto, sin reutilizar el estado de optimizador antiguo. No se descargó VGG16
ni se entrenó tracking. Conv4 afecta g1→s_weight1→den1: den2 y den3 reciben g2
y f3 separados antes de conv4. Afirmar que conv4 cambia esas dos salidas sería
incorrecto para esta arquitectura.

Ejecución: Python3.13.13, PyTorch2.14.1+cu130, RTX3060 de12GiB. El README histórico
indica Python3.6/PyTorch1.6/CUDA10.2: no se presenta el entorno actual como una
reproducción binaria de aquél. Se conservó el venv compartido, sin instalaciones.
Trabajos con lote1, High1,5GiB/Max2GiB RAM, swap0,5GiB y timeout120s, sin elevar
límites. Los dos entrenamientos duraron9,51/10,44s internos y usaron234,8/236,5MiB
CUDA; RSS1,686/1,740GiB. Estas duraciones sólo describen la ejecución.

## 3. Control del dominio original antes de entrenar

Una imagen central disponible por cada una de30 secuencias `test_data`, elegida
determinísticamente antes de inferir; en esta copia, frame152. Anotaciones humanas
originales MAT: filas `(x,y,id)` y `number`, sin sumar los IDs. GT total4105;
43–252 personas por imagen. Esta muestra no es todo el benchmark de4500 imágenes.

| Receta/salida original | MAE | RMSE | Sesgo predicción−GT |
| --- | ---: | ---: | ---: |
| Cuatro cuadrantes960×540, den1 principal | 20,3018173 | 23,8159302 | −3,2175793 |
| Mosaicos con halo, den1 | 19,4460165 | 22,9875817 | −1,0546473 |
| Cuatro cuadrantes, den2 | 22,9981528 | — | −3,9906221 |
| Cuatro cuadrantes, den3 | 22,1442390 | — | −6,6223119 |

Conteo histórico: sumar float32 por cuadrante, clamp inferior de cada suma,
sumar las cuatro y dividir por100. Los mapas crudos guardados mantienen negativos.
El mosaico usa núcleo320×128, halo96 y cobertura única, con las mismas cuatro
sumas globales. No se ajustó un multiplicador a los resultados.

La muestra dio evidencia contra un error grueso de unidades en inferencia,
sin descartar defectos más sutiles de compatibilidad. Un total cercano al GT
tampoco acredita localización correcta de cada persona. Las30 imágenes ya fueron
observadas y se reutilizaron después como control de retención, no como test final
ciego ni evidencia de generalización a la marcha.

## 4. Datos efectivamente usados para ajustar

### Replay original

`train_data` contiene82 secuencias y12.300 imágenes. Se seleccionaron ocho IDs
equidistantes en la lista ordenada, un frame central disponible por secuencia y
dos recortes fijos512×512 nativos, sin resize ni aumentos:
`[192,128,704,640]` y `[1152,512,1664,1024]`.

| Secuencia/frame | Cabezas originales en imagen completa | Recortes |
| --- | ---: | ---: |
| 001/152 | 105 | 2 |
| 014/152 | 289 | 2 |
| 032/152 | 91 | 2 |
| 049/152 | 43 | 2 |
| 060/152 | 102 | 2 |
| 080/152 | 52 | 2 |
| 093/152 | 74 | 2 |
| 110/152 | 366 | 2 |

No son16 imágenes independientes sino16 recortes de ocho imágenes. Sus secuencias
son disjuntas de test/val; val comparte secuencias con test en la copia original.
La repetición de estos16 recortes es determinista y cíclica, sin shuffle.
No se incorporaron gradientes de las30 imágenes de evaluación.

### Etiquetas nuevas

Siete ROIs totales, sólo cuatro entran en gradientes. Las marcas fueron propuestas
por agentes y revisadas por otro agente; no hay revisión humana externa. Las
referencias de las primeras fotos habían sido vistas durante controles anteriores.
Los puntos nuevos de la foto reservada se fijaron antes de sus predicciones.

| Región | Fuente/grupo | Partición | Supervisión | Confirmados nominales | Centros supervisados | Masa objetivo supervisada | Píxeles válidos |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| foto-dispersas | Foto8418 | Train | Parcial positiva | 7 | 2 | 2,06069 | 703 |
| foto-grupo | Foto8418 | Train | Parcial positiva | 12 | 9 | 7,95940 | 1805 |
| edificio-b | Video0265 | Train | Negativa | 0 | 0 | 0 | 9216 |
| foto-calle-vacia | Foto8418 | Train | Negativa | 0 | 0 | 0 | 9216 |
| edificio-a | Video0269 | Test repetido | Negativa | 0 | 0 | 0 | 50176 |
| test-peaton-fuente | Foto8415 | Test | Parcial positiva | 3 | 2 | 1,84375 | 349 |
| test-suelo-fuente | Foto8415 | Test | Negativa | 0 | 0 | 0 | 256 |

Los valores de las tres filas test sólo describen máscaras diagnósticas: nunca
se usaron para entrenar. Se guardan nueve candidatos dudosos en todo el conjunto.
Los bordes y candidatos dudosos explican que19 marcas nominales de entrenamiento
se reduzcan a11 centros efectivamente supervisados. Parte de las colas de los
kernels de centros excluidos aún puede contribuir donde la máscara es válida.

Los recortes de entrenamiento foto-dispersas y foto-grupo miden256×256; se añade
contexto128 alrededor. El contexto de entrenamiento nunca supera512 por lado.
La fachada reservada mide512×512 y su contexto768×768 se evalúa con mosaicos.
No se anotaron positivos exhaustivos nuevos. Los grupos densos irresolubles de
video quedaron excluidos, sin convertir cabezas no marcadas en ceros.

Se separaron hashes y videos entre train/test, pero todas las fuentes son del
mismo acto y pueden tener dependencia visual. Esta separación no es una prueba
fuerte de generalización geográfica o entre eventos.

## 5. Blancos, máscaras y pérdidas exactas

### Blanco histórico de replay

Se reconstruye a resolución540×960 desde todos los puntos de la imagen1920×1080,
antes de recortar. Los centros se cuantizan mediante floor(x/2),floor(y/2) en un
mapa booleano: colisiones de grilla se fusionan, como en el generador auditado.
Para cuatro o más centros ocupados se usa sigma=0,1×suma de distancias a los tres
vecinos más cercanos. Gaussianas con soporte truncado a4sigma, float32; en los
bordes se conserva la truncación histórica, sin renormalizar cada cabeza.
Caché RAM de dos mapas completos. No se usaron ni recrearon HDF5 para el piloto.

El auditor había contrastado esta regeneración contra los archivos históricos
antes de su retirada. Ese contraste no valida automáticamente el nuevo blanco
fijo para etiquetas de la marcha.

### Blanco nuevo

Un kernel por punto confirmado, sigma fijo2 en el mapa stride2, soporte±8.
Se normaliza cada kernel para sumar1 incluso si se recorta en el borde; puntos
coincidentes se suman. A diferencia del generador original, aquí se conserva
masa por anotación y no se calcula sigma adaptativo.

Se excluyen32 píxeles de borde del ROI y discos de radio16 píxeles de entrada
alrededor de candidatos dudosos. Para parciales se supervisa sólo donde la suma
de kernels positivos es mayor que cero; el resto del ROI y el contexto no
aportan pérdida. Los negativos claros usan toda la zona interior válida.

Si `p` es la salida den1 en unidades del modelo, `t` el blanco en personas/píxel
y `M` la máscara, las pérdidas por ejemplo son:

```
Replay o negativo: L = mean_M((100*t - p)^2)
Parcial positivo:  L = mean_M(max(100*t - p, 0)^2)
```

Cada ejemplo tiene peso1; no hay coeficientes distintos por fuente, sigma o área.
La parcial no penaliza exceso de densidad. Matemáticamente, su derivada respecto
a cada salida supervisada es <=0: puede pedir aumentar esa salida o no cambiarla.
Esto evita castigar personas adicionales sin marcar, pero no prueba que sea un
objetivo suficiente para adaptar conteos. La dirección de cada parámetro y la
actualización Adam no se deducen sólo de esa desigualdad.

### Optimizador y parámetros

Ambos pilotos parten del mismo den original, semilla20261025, lote1, Adam nuevo
LR1e-5, betas por defecto(0,9;0,999), eps por defecto1e-8, sin weight decay;
gradiente limitado a norma global1. No se restaura optimizador ni se usa scheduler,
early stopping por métrica intermedia, regularización de pesos o distillation.

Piloto1: sólo `den_output_layer1.weight/bias`,129 parámetros. Piloto2:
`conv4.weight/bias` más los anteriores,147.713. Los otros tensores se congelan
y comparan por hash antes/después. Sólo cambian los permitidos.

Orden: nuevo, original, original, original, repetido50 veces. Hay150 pasos
originales y50 nuevos; los nuevos rotan por cuatro ROIs:26 pasos de parciales
positivos y24 negativos. Es una mezcla1:3 por ejemplo, no por píxel ni por norma
de gradiente. Se entrena den1; no se añaden pérdidas para den2/den3.

## 6. Diferencias respecto del código histórico local

| Aspecto | Ruta histórica leída | Piloto ejecutado |
| --- | --- | --- |
| LR | 1e-6 | 1e-5, diez veces mayor |
| Adam beta1 | 0,5 | 0,9 por defecto |
| Pérdida den1 | Suma MSE con coeficiente y normalizaciones históricas | Media sobre máscara de cada ejemplo |
| Otras salidas | Pérdidas den2/den3 ponderadas | Sólo den1 |
| Recortes | Cuadrantes960×540, flip aleatorio | Fijos512×512/contextos nuevos |
| Datos por paso | Clips temporales, dos frames por defecto | Una imagen |
| Parámetros | Filtro histórico de parámetros habilitados | Última cabeza o conv4+cabeza |
| Blanco nuevo | No forma parte de receta histórica | Sigma fijo2 y parcial unilateral |
| Estado Adam | Checkpoint histórico puede contenerlo | Adam reiniciado |

El código `mytrain.py` contiene `if loss > den_loss` antes de backward/step:
en modo den puro puede omitir actualizaciones. Se evitó en la nueva ruta y se
probó que ésta actualiza pesos. No se infiere que los pesos publicados provengan
de ejecutar exactamente esa rama; no hay historial de entrenamiento que lo pruebe.
La receta histórica local tampoco es una verdad experimental que deba copiarse
sin comprobarla. Los cambios anteriores son confusores a controlar, no defectos
causales demostrados por su sola diferencia.

## 7. Resultados completos y criterios previos

Umbral fijado antes del ajuste: MAE histórico <=1,20×20,3018173=24,3621808.
Se detiene ante no finitos, cambios de congelados o colapso total a cero.
El guardia de colapso examina tres recortes originales con masa>=1; no mide
inflación de conteo ni sustituye a la evaluación de retención.

| Modelo | MAE | RMSE | Sesgo | Total predicho / GT4105 | Retención |
| --- | ---: | ---: | ---: | ---: | --- |
| Original | 20,3018173 | 23,8159302 | −3,2175793 | 4008,4726 | Referencia |
| Cabeza | 35,0225391 | 42,0966026 | +33,9497372 | 5123,4921 | Falla |
| Conv4+cabeza | 112,8739169 | 118,9454745 | +112,8739169 | 7491,2175 | Falla |

En conv4+cabeza, sesgo y MAE coinciden: los30 errores son no negativos.
Las otras dos salidas no se usan para elegir un resultado después de verlo.

| ROI / división | Referencia provisional | Original | Cabeza | Conv4+cabeza |
| --- | --- | ---: | ---: | ---: |
| Dispersas / train | 7–8 candidatos | 9,57514 | 11,75386 | 18,38016 |
| Grupo / train | 12–19 candidatos | 21,05571 | 23,67726 | 31,99082 |
| Techo / train | 0 discernibles | 2,07362 | 2,82612 | 4,39707 |
| Calle / train | 0 discernibles | 1,38777 | 2,16936 | 3,81351 |
| Fachada / test repetido | 0 discernibles | 28,12121 | 35,25111 | 54,14265 |
| Peatones / test | 3 claros y1 dudoso | 10,24518 | 11,99088 | 16,74042 |
| Suelo / test | 0 discernibles | 0,24708 | 0,35703 | 0,57459 |

ROIs: resolución nativa, contexto y mosaicos idénticos entre variantes; suma
float64/100 y clamp después de sumar el ROI. Son masas de densidad, no detecciones
individuales ni tasas de falsos positivos. La Gaussiana puede cruzar límites del
recorte. Los valores no son MAE certificado de la marcha.

| Supervisión | Pasos | Pérdida media cabeza | Pérdida media conv4+cabeza |
| --- | ---: | ---: | ---: |
| Replay | 150 | 0,01403488 | 0,01447642 |
| Parciales positivos | 26 | 0,55158977 | 0,51292370 |
| Negativos nuevos | 24 | 0,00014108 | 0,00045239 |

Estas medias agregan pasos sobre ejemplos distintos; no son una curva de descenso
comparable ni normas de gradiente. En cabeza, el primer paso de dispersas registró
norma previa a clipping3,156 y pérdida0,4164; no se midieron todos los gradientes
por fuente en un mismo estado inicial. No hay una ablación que cuantifique el
efecto individual de LR, betas, kernels, máscaras o mezcla.

## 8. Qué comprobamos y qué faltó

Comprobado: hashes de fuentes/pesos, MAT compatible, factor100, masa de blancos
nuevos, máscaras, gradiente unilateral, real optimizer.step, tensores congelados,
separación de fuentes y guardia de colapso. Doce pruebas CPU y smoke GPU original
de cinco pasos. El smoke cambió sólo los dos tensores den1 y mantuvo conteos
positivos en tres recortes originales. No midió retención tras200 pasos.

Controles ausentes relevantes:

1. Original-only de150/200 pasos con la misma ruta, replay, LR y Adam, evaluado
   exactamente igual. Sin él no se separa deterioro de receta base y adaptación.
2. Gradientes por fuente en un checkpoint común: signo del bias, norma y
   alineación antes/después de clipping. No equivalen directamente a pasos Adam.
3. Blanco histórico y nuevo comparados en una misma anotación compatible, para
   aislar escala espacial, normalización y colisiones sin cambiar personas.
4. Regiones nuevas exhaustivas y revisión humana; más diversidad de personas,
   fondos y eventos. Hoy sólo11 centros nuevos están efectivamente supervisados.
5. Validación de entrenamiento separada del test histórico repetido; varias
   semillas y un conjunto final no consultado. No hay estimación de incertidumbre.
6. Checkpoints intermedios o criterio de inflación, sin elegirlos mirando el test
   final. Sólo se guardó el final y el guardia evita cero, no sobreconteo.

La evidencia permite afirmar deterioro de los pesos nuevos bajo estos controles.
No permite atribuirlo con certeza a pocas etiquetas, incompatibilidad de Python,
olvido catastrófico, sigma, unilateralidad o capacidad del modelo por separado.

## 9. Hipótesis a discutir, con observaciones que las podrían refutar

| Hipótesis | Motivo para considerarla | Control discriminante |
| --- | --- | --- |
| Parciales estrechos inducen presión ascendente desbalanceada | Derivada unilateral, pocas zonas positivas y áreas distintas; aumentan todas las masas ROI | Gradientes iniciales por fuente y comparación de objetivo espacial con anotaciones comunes |
| La ruta base ya deteriora el modelo | Cambios de LR/betas/crops/reducción/Adam y falta de control largo | Original-only150/200; si conserva capacidad, esta explicación pierde peso |
| El bias global explica parte del incremento | La cabeza final es conv1×1 con bias por todos los píxeles | Delta exacto del bias y contribución uniforme sin clamp; distinguirla del cambio de pesos/features |
| Pocos datos y repetición permiten un ajuste no transferible | Ocho frames originales y dos ROIs positivos de una sola foto | Comparar train/validación con más regiones completas y fuentes reservadas |
| Diferencia train512/evaluación960 o mosaicos importa | Contexto, bordes y mezcla multiescala dependen de geometría | Evaluación del original y control original-only con ambas geometrías fijas |
| Error de datos/unidades pese a verificaciones | Las verificaciones no cubren todas las rutas posibles | Contrastar blancos y crops contra referencia, reproducir por archivo, usar otras cabezas como testigo |

Para el bias final, una variación uniforme delta_b añade5184×delta_b personas
crudas a una imagen1920×1080 a stride2 antes del clamp. El clamp por cuadrante
puede romper la igualdad en el conteo final. Esa identidad algebraica no estima
por sí sola el efecto total, porque también cambian pesos y, en piloto2, conv4.

## 10. Propuesta mínima para revisar, todavía no ejecutada

Primero pedir dictámenes independientes sobre este expediente y el código;
después discutir desacuerdos. No añadir barridos ni más datos mientras se
revisa el diseño. Un posible siguiente control es original-only con150 pasos
para igualar exposiciones originales y/o200 para igualar número total de pasos;
los dos emparejamientos responden preguntas distintas y deben declararse.
Fijar particiones, métricas, condición de parada y presupuesto antes de ejecutarlo.
Medir gradientes iniciales sin actualizar puede ser un diagnóstico más barato.

Si la receta original-only falla, corregirla antes de atribuir el deterioro a las
nuevas marcas. Si conserva, aislar una modificación a la vez. Los revisores deben
proponer el orden y justificar qué resultado respaldaría o refutaría cada hipótesis.

## 11. Evidencia para reproducir y auditar

Todo es local. Los revisores pueden leer las fuentes y figuras; no se subirán
fotos/videos a servicios ni se instalarán herramientas para esta discusión.

| Qué | Ruta relativa al repositorio |
| --- | --- |
| SPEC del piloto | `sdd/005-ajuste-dominio/SPEC.md` |
| Entrenador y blancos nuevos | `scripts/train_domain_pilot.py`, `scripts/domain_targets.py` |
| Preparador de replay | `scripts/prepare_domain_replay.py` |
| Generador histórico auditado | `scripts/audit_hdf5.py:reconstruct_density` |
| Arquitectura y receta histórica | `STNNet/stnmodel.py`, `STNNet/mytrain.py`, `STNNet/image_train.py` |
| Carga restringida / normalización | `scripts/count_video.py` |
| Evaluadores | `scripts/evaluate_calibration.py`, `scripts/evaluate_roi_controls.py` |
| Cohorte test congelada | `outputs/validacion-controles/historico-validado/cohort.json` |
| Baseline histórico completo | `outputs/validacion-controles/reporte-historico/summary.json` |
| Puntos nuevos congelados | `outputs/ajuste-dominio/anotaciones-v2.json` |
| Contextos y controles | `outputs/ajuste-dominio/controles-preparados/controls.json` |
| Replay y hashes fuentes | `outputs/ajuste-dominio/replay-original/replay.json` |
| Entrenamientos | `outputs/ajuste-dominio/{mixto-head,mixto-conv4-head}/training.json` |
| Evaluación histórica posterior | `outputs/ajuste-dominio/{mixto-head-retencion,mixto-conv4-retencion}/evaluation.json` |
| Baseline y evaluación ROI | `outputs/ajuste-dominio/{baseline-rois,mixto-head-rois,mixto-conv4-rois}/evaluation.json` |
| Resultados/figura comparativa | `outputs/ajuste-dominio/reporte/{summary.json,controles-reservados.png}` |
| Máscaras efectivas | `outputs/ajuste-dominio/supervision-efectiva.json` |
| Smoke y código exacto antiguo | `outputs/ajuste-dominio/smoke-original/training.json`, `outputs/ajuste-dominio/executed-code/` |
| Pruebas nuevas | `tests/test_domain_targets.py`, `tests/test_domain_annotations.py`, `tests/test_train_domain_pilot.py` |

Hashes de manifiestos: cohorte histórica
`f94dd4a09e014b57cae0b365fd4dbbccea8598599157f18e40a718521d9e2f8e`;
anotaciones nuevas
`b7b37952b31d2efd3c3c3e6079587be88096f9b5c368bc56c46ccd2e0f6cfa1b`;
controles preparados
`098fe0d33e0ff16cb4337955c99bf9b7405f7a84388cbb69d5433aee33196002`;
replay
`f929b0209cc7e9879e9627eebb3a2b51a61e5ec7b7fe6f0c22081b4afbb420e7`.

## 12. Preguntas y protocolo de discusión

Cada revisor responde primero sin leer el dictamen del otro:

1. ¿Qué conclusiones están respaldadas y cuáles fueron excesivas?
2. ¿Hay un bug verificable, un diseño insuficiente o ambos? Citar código/datos.
3. ¿Qué hipótesis explica mejor la dirección y magnitud del deterioro? Separar
   mecanismo matemático de inferencia causal y señalar posibles refutaciones.
4. ¿Qué control mínimo haría antes de volver a ajustar con fotos nuevas?
5. ¿Mantendría la pérdida unilateral y el kernel elegido? ¿Bajo qué condiciones?
6. ¿Qué debe corregirse en el informe previo y qué está bien comprobado?

Luego intercambian sus dictámenes, cuestionan al menos una afirmación concreta
del otro y responden a sus objeciones. Registrar consensos, desacuerdos persistentes
y evidencia que falta; no forzar consenso ni presentar hipótesis como medidas.
Se conserva la identidad real de cada modelo. Fable permanece pendiente de que
el autor indique su identificador/acceso; ninguna revisión se le atribuye todavía.

Este expediente inicial será seguido por un acta de dictámenes y discusión.
