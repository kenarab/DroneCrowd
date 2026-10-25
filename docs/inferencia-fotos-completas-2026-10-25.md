# Modelo original aplicado a las ocho fotos completas

Se ejecutó el checkpoint original sobre las8fotos aportadas, incluida la
panorámica de5200×12480, sin esperar más etiquetas y sin entrenamiento.
Hay predicciones y vistas privadas de todas. Las anotaciones son necesarias para
evaluar y corregir errores; no son un requisito para ejecutar la inferencia.

## Salidas exploratorias

| Foto en la vista | Descripción/identificador | Suma de densidad | Máximos propuestos |
| --- | --- | ---: | ---: |
| 00 | Amplia, IMG_8418 | 2609,36 | 2939 |
| 01 | Aérea0257 | 3402,55 | 4458 |
| 02 | Aérea0258 | 3665,25 | 3058 |
| 03 | Aérea0260 | 3432,32 | 2062 |
| 04 | Panorámica0262 | 2459,00 | 2271 |
| 05 | Aérea0266 | 1415,05 | 1768 |
| 06 | Plaza, IMG_8415 | 7126,77 | 5359 |
| 07 | IMG_8416 | 3230,92 | 2546 |

Son medidas diferentes: la suma integra densidad; los máximos provienen del
postprocesado histórico `calc_denpt`, no de personas verificadas. Los conteos no
son cifras de asistencia y no se suman: las fotos tienen campos superpuestos.
Las figuras muestran activaciones en fachadas/techos. Falta fijar y validar la
máscara de calle, vereda y plaza que excluya terrazas y otras zonas fuera del área
de interés. No se aplicó una máscara silenciosamente ni se afirma precisión local.

## Resolución, partición y recursos

Entradas a resolución nativa, sin interpolación ni píxeles agregados. Se excluyeron
3columnas derechas de la foto de5159×3804 para obtener5156×3804, por alineación a4;
las otras7conservaron sus dimensiones. Todas tenían orientación EXIF1, registrada
junto con la política de trabajar sobre los píxeles RGB almacenados, sin giro
automático. Las imágenes originales no se modificaron.

Las fotos se dividieron en21bandas de núcleo máximo2048filas, halo96 y origen
alineado a8. Dentro de cada banda se usaron tiles de núcleo320×128 y contexto
máximo512×320. Se conservaron los núcleos disjuntos; ningún píxel se contó dos
veces al unirlos. Se sumó densidad cruda y se aplicó el truncamiento a cero una
sola vez por cada uno de los4cuadrantes de la foto completa, dividiendo por100.
No se sumaron conteos truncados por banda. Se registró también la masa negativa.
El contexto finito es una aproximación; no garantiza igualdad con ejecutar toda
la foto de una sola pasada.

Se prepararon559,89MiB RGB,21paneles y5111tiles. Las21ejecuciones GPU fueron
secuenciales, con preflight,2GiB RAM/512MiB swap y120s cada una. Tiempo sumado
reportado132,36s, incluyendo carga y cómputo; no es una medición de tiempo de
kernel CUDA. Pico RSS1,109GiB y CUDA161,09MiB. Todas finalizaron dentro de las
cuotas. Agregación y vistas se hicieron en CPU; no hubo HDF5 ni entrenamiento.

## Reproducción y evidencia

[SPEC011](../sdd/011-fotos-completas/SPEC.md) se escribió antes de preparar datos.
`scripts/count_photo_panels.py` ofrece tres comandos: `prepare`, `infer` por índice
de panel y `aggregate` por fuente. Sus ayudas describen los argumentos exactos.
`prepare` usa Python CPU con Pillow/NumPy; `infer` usa el entorno ML mediante
`ops/python-ml.sh`, con Torch importado sólo en ese modo. No se instaló ni modificó
ninguna dependencia compartida. Cada informe conserva fuente, dimensiones,
transformación, hashes de modelo/cargador/código, versiones y recursos.

Manifiesto privado:
`outputs/ajuste-dominio/fotos-completas/inputs/inputs.json`, SHA-256
`63efe52f9d188a3fde525302437f2338f0193f84250e2dd6dd5771d7d4cbf8f8`.
Predicciones: `predictions/panel-000/` hasta `panel-020/`, bajo ese directorio.
Agregados: `aggregate/photo-000/` hasta `photo-007/`, con `report.json` y
`density-full.npy`. El checkpoint original conserva SHA-256
`be407d2dcdee6775a342c343523cd17399310d91acb548172d8f7000606d501d`.

`scripts/render_photo_predictions.py` recibe el manifiesto/hash y los8reportes;
verifica procedencia y vuelve a calcular los conteos antes de mostrar originales
reducidos, propuestas en magenta y brillo de densidad normalizado por foto.
Salida privada: `revision/fotos-originales/`, con JPEG menores de1,1MiB y un índice
de hashes. Las vistas están reducidas para navegar; las entradas del modelo eran
nativas. La portada `revision/index.md` enlaza estas fotos y los19recortes revisados.
Se comprobaron página e imágenes por HTTPS autenticado. No se publican originales,
mapas ni etiquetas privadas en GitHub.

Implementación del contador:Luna; render y coordinación:Codex. Revisión independiente
con Codex, modelo no identificado:4pruebas geométricas aprobadas,21núcleos y8mapas
comprobados, cobertura única y conteos recalculados, fuentes/EXIF/hashes verificados
y figuras00/04/06 inspeccionadas. Sin bloqueantes. Entorno de inferencia:
Python3.13.13, Torch2.14.1+cu130 y NumPy2.5.4; CPU con Pillow13.0.0 y SciPy1.18.1.

El siguiente paso de evaluación es delimitar el suelo de interés y revisar regiones
resolubles, conservando negativos arquitectónicos para medir falsos positivos.
Más puntos automáticos permiten acelerar esa revisión, sin reemplazarla.
