# Anotaciones y primer piloto de ajuste

Los dos ajustes exploratorios empeoraron los controles. Se conservan como
experimentos rechazados; el checkpoint original continúa como referencia.
Este resultado no demuestra que ajustar el modelo sea imposible: muestra que
esta pequeña combinación de etiquetas, blancos y pérdidas no produjo una mejora.

| Variante | Parámetros ajustados | MAE histórico, 30 imágenes | Decisión |
| --- | ---: | ---: | --- |
| Original | 0 | 20,30 | Referencia sin modificar |
| Cabeza den1 | 129 | 35,02 | Rechazado |
| Conv4 + den1 | 147.713 | 112,87 | Rechazado |

Umbral de retención fijado antes de entrenar: MAE <=24,3621808, equivalente
a 1,20 veces el baseline. La cohorte ya se había observado; es un control
repetido contra anotaciones originales del dataset, no una evaluación final ciega
ni una medida de precisión en la marcha.

| Región reservada de la marcha | Referencia provisional | Original | Cabeza | Conv4 + cabeza |
| --- | --- | ---: | ---: | ---: |
| Fachada, video0269 | Sin personas discernibles | 28,12 | 35,25 | 54,14 |
| Peatones, foto8415 | 3 cabezas claras y 1 candidato dudoso en el borde | 10,25 | 11,99 | 16,74 |
| Suelo, foto8415 | Sin personas discernibles | 0,25 | 0,36 | 0,57 |

Las salidas son integrales de densidad en unidades de personas, no detecciones
individuales ni tasas de falsos positivos. Las referencias nuevas son marcas
visuales provisionales de agentes; no constituyen GT humano certificado.
La densidad puede cruzar bordes de los recortes.

## Qué se etiquetó y cómo revisarlo

Se congelaron siete regiones: cuatro de entrenamiento y tres reservadas.
Entrenamiento: dos recortes de la foto8418 con positivos parciales, un techo del
video0265 y una calle vacía de la foto8418. Evaluación: fachada del video0269 y
peatones/suelo de la foto8415. No se mezclan hashes ni videos entre grupos,
aunque las fuentes siguen relacionadas por pertenecer al mismo acto.
Los grupos densos irresolubles quedaron excluidos, sin convertir personas sin
marcar en fondo. Hay 22 puntos confirmados visualmente y nueve candidatos
dudosos en el conjunto completo. Tras las máscaras, los positivos de entrenamiento
tienen 11 centros supervisados y unas 10,02 unidades de masa objetivo; los puntos
nominales cercanos a bordes no equivalen a supervisión efectiva.

- [Revisor HTML privado y offline](../outputs/ajuste-dominio/revision-v2/revisar-puntos.html): abrir como archivo local, agregar/quitar puntos, separar dudosos y exportar JSON. Sin servidor ni red.
- [Etiquetas congeladas](../outputs/ajuste-dominio/anotaciones-v2.json), SHA-256 `b7b37952b31d2efd3c3c3e6079587be88096f9b5c368bc56c46ccd2e0f6cfa1b`.
- Siete MAT en `outputs/ajuste-dominio/revision-v2/*-provisional.mat`: estructura original `image_info.location=[x,y,id_local]`, `number=N`, más metadatos de supervisión. Coordenadas locales, cero inicial; los IDs no son trayectorias. No reemplazan los MAT del dataset.
- [Comparación visual](../outputs/ajuste-dominio/reporte/controles-reservados.png), [tabla](../outputs/ajuste-dominio/reporte/counts.csv) e [informe trazable](../outputs/ajuste-dominio/reporte/summary.json).

Las ediciones invalidan la revisión previa. Para volver a exportar MAT, calcular
el nuevo SHA-256 y ejecutar `scripts/domain_annotations.py --manifest RUTA
--sha256 HASH --output-dir DIRECTORIO_NUEVO`; nunca sobreescribir el conjunto
congelado. El HTML pasó sintaxis JavaScript; falta una prueba interactiva de
navegador. Las imágenes y artefactos privados permanecen fuera de Git.

## Experimento y alcance de la conclusión

Ambos pilotos arrancaron desde el den original verificado, con Adam nuevo,
LR1e-5, lote1, semilla20261025, norma de gradiente limitada a1 y 200 pasos:
150 originales y 50 nuevos. Replay fijo de ocho secuencias de `train_data`,
midpoint disponible y dos recortes512×512 por secuencia, disjuntos de test/val.
Se reconstruyó cada blanco histórico completo en RAM antes del recorte; caché
de dos mapas. No se recrearon HDF5.

Los blancos nuevos conservan una persona por punto, incluso para pocos puntos,
bordes y colisiones, con sigma2 a stride2. Los dudosos se excluyen en radio4sigma
y se ignoran32 píxeles de borde. Para positivos parciales se usa pérdida
unilateral: no penaliza personas adicionales sin marcar. Los negativos claros y
el replay usan MSE bilateral, siempre con factor100 y media por píxel válido.

La pérdida parcial positiva media fue0,552/0,513 frente a0,0140/0,0145 en replay
y0,000141/0,000452 en negativos, respectivamente. Esto evidencia escalas distintas
entre términos: una proporción de muestras1:3 no equilibra por sí sola su efecto.
Los blancos nuevos más concentrados y los pocos puntos disponibles son diferencias
relevantes de esta prueba. No se aisló causalmente cuál explica el deterioro;
las pérdidas tampoco son precisión ni equivalen al aporte efectivo de gradiente
tras Adam y clipping. No se hicieron más barridos de hiperparámetros.

Antes de nuevos ajustes conviene ampliar regiones exhaustivamente revisadas,
incluir más fondos difíciles y fijar un balance de supervisión defendible.
Después corresponde repetir el control histórico y evaluar fuentes reservadas.
No alcanza con observar que sube el total de personas predicho.

## Ejecución, revisión y recursos

Doce pruebas CPU nuevas: masa, colisiones/bordes, máscaras, gradientes unilaterales,
compatibilidad MAT, separación de fuentes, actualización real y conservación de
congelados, y guardia de colapso. Smoke original de cinco pasos en GPU completo.
En cada piloto cambiaron exclusivamente los tensores permitidos. Los checkpoints
se cargaron en modo restringido y con hash; no se reutilizó estado de optimizador.

RTX3060, trabajos acotados a120s, scope High1,5GiB/Max2GiB/swap0,5GiB sin aumentar
cuotas. Entrenamientos:9,51s y10,44s internos; picos CUDA234,8/236,5MiB y
RSS1,686/1,740GiB. El preflight inicial estimaba1,25GiB; después del smoke se
actualizó a1,75GiB y volvió a admitirse. Salidas nuevas totales:262MiB, inferiores
al presupuesto512MiB. Quedan aproximadamente410GiB libres: el SSD externo no
es necesario para este piloto ni para reconstruir blancos por lote.

La rutina histórica de entrenamiento puro podía omitir `optimizer.step` por su
condición `loss > den_loss`; se conservó ese código y se creó una ruta explícita
comprobada. No se infiere de ese defecto la procedencia de los pesos publicados.

Se aplicó la SPEC005 con commits locales atómicos. Revisor independiente existente:
herramienta Codex, modelo no identificado. Codex GPT-6 Luna, con contexto acotado,
preparó el replay y revisó formato/MAT/HTML y aritmética de resultados. La revisión
del coordinador corrigió su selección inicial de secuencias y sus primeras
coordenadas visuales; el revisor independiente aprobó las etiquetas finales.
Esta delegación resulta útil para tareas mecánicas; no se midió un ahorro total
de tokens ni se aceptan etiquetas visuales sin revisión por usar un modelo barato.

Checkpoints experimentales privados, sin sustituir el original:
`mixto-head/den-domain-pilot.pth.tar`, SHA-256
`52ce16a4854041edae67b549388ada82b07f9d686c7b5722f3176e1f91cf2a7c`;
`mixto-conv4-head/den-domain-pilot.pth.tar`, SHA-256
`81a5218cd60ec5967c73a4b6b87e60db5af37ed524bf5e3e05e7e346c851e1e8`.
Los manifiestos, logs JSON y hashes de código ejecutado quedan bajo
`outputs/ajuste-dominio/`; no se modificaron originales ni el entorno compartido.
