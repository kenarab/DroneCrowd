# Prueba STNNet con los videos de la marcha

La ejecución del modelo entrenado funciona en el entorno actual. La configuración
inicial a 960×540 resulta insuficiente para estimar la multitud visible: las nueve
salidas oscilan entre 20 y 82 personas, con zonas densas casi sin respuesta y
activaciones sobre edificios. Es una conclusión cualitativa al comparar imágenes
y mapas; no se dispone de anotaciones ni se calculó un error porcentual.

Se muestrearon tres instantes de cada video, próximos al 10%, 50% y 90% de duración.
Cada fotograma se redujo a 960×540 y se dividió en cuatro cuadrantes secuenciales,
lote 1. Se usó la primera cabeza de densidad, RGB/ImageNet y la suma de cada
cuadrante limitada inferiormente a cero, dividida por 100 como en `mytest.py`.
El modelo de densidad procesa cada imagen por separado; no se suman fotogramas
como personas únicas. Los dos MOV originales son 4K; el MP4 ya es 960×540.

| Video | PTS de los tres fotogramas (s) | Salidas del modelo (personas) |
|---|---|---|
|0265|2,286 /11,428 /20,571|20,0 /28,9 /62,0|
|0269|0,951 /4,705 /8,458|36,7 /46,0 /36,4|
|trim|1,110 /5,497 /9,902|82,4 /62,9 /67,3|

Comparaciones locales, conservadas fuera de Git:

- [Video0265: imágenes y mapas](../outputs/video-breve/20261024-resultados/0265.png).
- [Video0269: imágenes y mapas](../outputs/video-breve/20261024-resultados/0269.png).
- [Video trim: imágenes y mapas](../outputs/video-breve/20261024-resultados/trim.png).
- [Conteos con timestamps](../outputs/video-breve/20261024-resultados/counts.csv).
- Manifiestos de entrada, NPY y ejecución:
  `outputs/video-breve/20261024-validado/{0265,0269,trim}/`.

Las figuras usan una misma escala de color: su máximo es el mayor de los
percentiles 99,7 calculados por mapa. Valores superiores saturan el color. Los
valores negativos aparecen negros; se conservan sin recortar en los NPY y se usan crudos
para calcular las sumas. El color es una visualización de densidad, no una
enumeración de detecciones individuales. No permite atribuir cada unidad del
conteo a una persona identificable.

## Pesos, entorno y recursos

El autor aportó `~/home.agents/data/models.zip`, cuya integridad CRC completa se
comprobó. Se extrajo sólo `den_model_best.pth.tar` y se conservó una copia en
`STNNet/models/`, excluida de Git. Hashes y procedencia:
`outputs/video-breve/20261024-pesos.json` y [SPEC](../sdd/001-video-breve/SPEC.md).
No hay una firma ni hash publicado por el proveedor con el que autenticarlo;
los hashes registrados identifican la copia aportada y permiten reproducirla.

La primera carga falló por metadata NumPy heredada. Se conservó ese intento en
`0265/inferencia/inference.json`. La excepción mínima por hash mantiene
`weights_only=True`; las ejecuciones completas están en `inferencia-validada/`.
Los 41 tensores cargaron con nombres, formas, tipos y finitud estrictos. No se
descargó VGG16 ni se usaron pesos aleatorios para producir estos resultados.

Python 3.13.13, PyTorch 2.14.1+cu130 y NumPy 2.5.4 del venv compartido; GPU RTX 3060.
No se modificó ese entorno. Cada video tardó entre 1,05 y 2,24 s dentro del proceso
de conteo, sin incluir arranque de Python, muestreo ni renderizado. Pico RSS
máximo 1,009 GiB y memoria CUDA asignada máxima 136,72 MiB. Los límites kernel fueron
High 1,5 GiB/Max 2 GiB/swap 0,5 GiB, con timeout externo de 120 s, sin alterar cuotas
persistentes. Preflight de disco/RAM aprobado y GPU libre antes del ensayo.

Trece tests de los adaptadores pasaron: cuatro del muestreador y nueve de conteo.
El revisor independiente Codex `/root/revision_retencion_hdf5` aprobó la carga
restringida y verificó los nueve mapas, conteos, geometrías, hashes y timestamps;
su modelo efectivo no está expuesto. El coordinador inspeccionó visualmente las
tres láminas de resultados. No se ejecutó entrenamiento ni seguimiento.

Para regenerar las figuras con el Python base que ya tiene Matplotlib/Pillow:

```sh
export TMPDIR=/home/alephbet/home.codex/tmp TMP=/home/alephbet/home.codex/tmp TEMP=/home/alephbet/home.codex/tmp
export MPLCONFIGDIR="$TMPDIR/dronecrowd-matplotlib" XDG_CACHE_HOME="$TMPDIR/dronecrowd-xdg-cache"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
python3 scripts/render_video_report.py --root outputs/video-breve/20261024-validado --output-dir outputs/video-breve/otra-comparacion
```

## Decisión y siguiente ensayo

Conservar videos, pesos y esta evidencia pequeña; no procesar videos completos
con esta configuración. La reducción de resolución y el cambio de perspectiva,
densidad y tamaño aparente son posibles causas, aún sin aislar experimentalmente.
El siguiente ensayo propuesto es comparar recortes de los MOV a resolución nativa
con sus versiones reducidas y anotar manualmente algunas zonas para medir el
error. Requiere especificar geometría y presupuesto antes de ampliar esta prueba.

Los mapas HDF5 históricos no fueron necesarios en ninguna etapa de inferencia.
No se requiere el SSD externo para este ensayo breve; el disco local dispone de
espacio suficiente tras la retirada auditada de mapas regenerables.
