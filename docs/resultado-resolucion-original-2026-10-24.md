# Ensayo de resolución original y Lanczos

Se probaron los dos MOV en su resolución nativa 3840×2160 y el MP4 de 960×540
con interpolación Lanczos 2× y 4×. El detalle nativo produce una respuesta más
extendida que ampliar la imagen reducida. La revisión visual sigue mostrando
respuesta insuficiente en la multitud y activaciones en arquitectura; ninguno
de estos valores constituye una estimación validada de la convocatoria.

Un fotograma por video, con los mismos instantes que la prueba anterior:

| Video / PTS | Resolución nativa | Lanczos 2× desde 960×540 | Lanczos 4× desde 960×540 |
|---|---:|---:|---:|
|0265 / 20,570550 s|378,5 (4K)|—|135,4 (4K interpolado)|
|0269 / 4,704700 s|503,4 (4K)|—|87,0 (4K interpolado)|
|trim / 5,496667 s|61,6 (960×540)|84,7 (1920×1080)|128,5 (4K interpolado)|

Son salidas del modelo, en personas por fotograma, sin anotaciones de referencia.
La diferencia entre nativo e interpolado no mide precisión. En el MP4 sólo crece
el tamaño de entrada; la interpolación no recupera personas o detalles ausentes.
No se aplicó un factor multiplicativo al conteo para compensar resolución.

## Comparación visual y método

- [Fotograma0265 y sus mapas](../outputs/resolucion-original/comparaciones/0265.png).
- [Fotograma0269 y sus mapas](../outputs/resolucion-original/comparaciones/0269.png).
- [Fotograma trim y sus mapas](../outputs/resolucion-original/comparaciones/trim.png).
- [Tabla CSV](../outputs/resolucion-original/comparaciones/counts.csv).

Revisé las tres láminas: en 0265 nativo hay más respuesta dispersa en la zona de
la multitud, con áreas extensas débiles; en 0269 también se activan fachadas y
cúpulas. Las ampliaciones del MP4 cambian la respuesta, sin resolver visualmente
el conteo. No se enumeraron personas ni se atribuyeron picos a identidades.

Los controles de los MOV parten de las imágenes anteriores de 960×540,
ampliadas con FFmpeg `scale=3840:2160:flags=lanczos`. Comparten geometría,
partición, modelo, pesos y PTS con el ensayo nativo. Permiten observar el efecto
del detalle perdido al reducir la imagen. Comparar directamente contra los
conteos de la primera prueba mezcla también un cambio de contexto y partición.

Se ejecutó STNNet con la carga restringida ya validada, RGB/ImageNet y primera
cabeza de densidad. Núcleos 320×128, halo de contexto 96, tile máximo 512×320,
lote 1. Se conserva sólo la salida de cada núcleo para cubrir exactamente una
vez cada píxel; los solapes de contexto no se suman. No se garantiza equivalencia
con inferencia del fotograma completo. En los bordes se usa el contexto disponible,
sin agregar padding de entrada. Conteo final: las cuatro sumas globales de
cuadrantes, limitadas inferiormente a cero y divididas por 100.

Los NPY originales conservan los valores negativos. Para visualizar se reducen
a una grilla espacial común de 480×270, con promedio BOX multiplicado por la
relación de áreas, que conserva la masa salvo redondeo numérico. La escala de
color es compartida: máximo de los percentiles 99,7 de cada mapa; valores mayores
saturan y negativos aparecen negros. Esta transformación no altera los conteos.

## Evidencia y recursos

Entradas nativas: `outputs/resolucion-original/{0265,0269}/entrada-native/`.
Los PPM conservan todos los píxeles 4K decodificados; las previews PNG son 960×540.
Hash de video y PTS coinciden con la extracción anterior. Imágenes ampliadas sólo
en memoria, con SHA-256 del RGB transformado registrado en cada ejecución.
Manifiestos y mapas: `outputs/resolucion-original/<video>/<variante>/`.
Procedencia de pesos: `outputs/video-breve/20261024-pesos.json`.
Los originales se conservaron intactos y no se generaron HDF5.

Siete ejecuciones secuenciales completas, entre 1,02 y 5,71 s por configuración
dentro del proceso de conteo, excluyendo arranque de Python y extracción.
204 tiles por imagen 4K, 54 por 1920×1080 y 15 por 960×540. Pico RSS máximo
1,077 GiB y memoria CUDA asignada máxima 161,09 MiB; límites kernel efectivos
High 1,5 GiB / Max 2 GiB / swap 0,5 GiB, con timeout externo de 120 s. Preflight
Nexo y ocupación GPU comprobados; cuotas persistentes y entorno sin cambios.
Artefactos nuevos: aproximadamente 95 MiB, bajo el presupuesto de 128 MiB.

17 tests pasaron, incluidos cobertura única, bordes parciales, alineación espacial,
conservación de sumas, límites, salida no finita, timeout y decodificación nativa.
Revisor independiente: agente Codex autorizado `/root/revision_retencion_hdf5`,
modelo efectivo no expuesto. Aprobó plan, implementación y las siete ejecuciones;
verificó hashes, PTS, sumas, recursos y conservación de masa del render, e
inspeccionó las tres láminas.

## Reproducción y decisión

[SPEC del ensayo](../sdd/003-resolucion-original/SPEC.md). Los siete comandos con
sus entradas, escalas e índices están en `outputs/resolucion-original/commands.json`.
Ejemplo para 0265 nativo, desde el checkout:

```sh
export TMPDIR=/home/alephbet/home.codex/tmp TMP=/home/alephbet/home.codex/tmp TEMP=/home/alephbet/home.codex/tmp
systemd-run --user --scope --collect --slice=nexo-projects.slice \
  -p MemoryHigh=1610612736 -p MemoryMax=2147483648 \
  -p MemorySwapMax=536870912 -p OOMPolicy=kill -- \
  timeout --signal=TERM --kill-after=5s 120s \
  bash ops/python-ml.sh scripts/count_video_tiles.py \
  --input outputs/resolucion-original/0265/entrada-native/input.json \
  --frame-index 0 --scale 1 \
  --checkpoint STNNet/models/den_model_best.pth.tar \
  --checkpoint-sha256 be407d2dcdee6775a342c343523cd17399310d91acb548172d8f7000606d501d \
  --checkpoint-source 'https://drive.google.com/file/d/1H-GrqyYxERLzVM5wQblUDeX24Ht2lD6q/view' \
  --output-dir outputs/resolucion-original/0265/otra-ejecucion
```

Conservar la resolución nativa para próximos ensayos y estos resultados negativos.
No procesar todavía videos completos ni usar las cifras como totales de asistentes.
La siguiente validación útil requiere zonas anotadas manualmente para medir el
subconteo y las respuestas en edificios. No se inició otra búsqueda de modelos,
entrenamiento ni ampliación adicional del ensayo.
