# Prueba breve de STNNet con los videos de la marcha

## Objetivo y encargo vigente

El autor aportó videos en **~/home.agents/data/marchaUniversitaria241002** y
encarga el 24/10/2026 continuar las pruebas desde una nueva sesión de DroneCrowd.
Este expediente pertenece al fork personal git@github.com:kenarab/DroneCrowd.git.
Nexo entrega la preparación y coordina recursos; la nueva sesión de DroneCrowd
es escritora y responsable del adaptador, la prueba y su evaluación.

Objetivo práctico: aplicar un modelo con pesos entrenados a un fragmento real,
ver si el resultado justifica seguir y aportar evidencia para decidir la
conservación del dataset histórico. Primera opción: inferencia en GPU, sin
reentrenar. [Plan](PLAN.md), [traspaso](../../docs/traspaso-video-marcha-2026-10-24.md)
y [entorno comprobado](../../docs/entorno.md).

El autor informó que Juan Roleri le compartió el material el **2/10/2024**.
No confundirlo con el video de abril que apareció durante una búsqueda anterior.
La identidad concreta y características de cada archivo se comprobarán al recibirlo.
No iniciar otra búsqueda de copias ni descargar el video público mientras estos
originales están siendo aportados.

## Base ya verificada

| Incremento | Commit y alcance |
| --- | --- |
| Nombre correcto de requirements | d0442e4; renombre previo del autor |
| Pines históricos | 5f73ed0; versiones del intento anterior conservadas |
| Infraestructura y diagnóstico | b06f1c0; AGENTS/CLAUDE, exclusiones y lanzador ml |
| Compatibilidad básica del modelo | 8c52c73; arranque offline y pruebas sintéticas |

Se reutilizó **~/home.agents/venvs/ml** sin instalar paquetes. Diagnóstico:
Python3.13.13, PyTorch2.14.1+cu130, CUDA13.0, RTX3060 de12GiB. Conteo y
localización pasaron dos pruebas en CPU y GPU con dos tensores32x32, suite2,377s.
Se verificaron límites kernel de un scope propio: High1,5GiB/Max2GiB/swap0,5GiB
dentro del pool existente de8GiB. Los valores y versiones deben reconsultarse.

STNNet admite pretrained_backbone=False: evita descargar VGG16 al cargar un
checkpoint completo. Los imports de correlación y PointConv sólo se requieren
para seguimiento. El valor por defecto conserva la inicialización histórica.
Esto **no demuestra calidad, pesos cargados ni inferencia sobre video**.
STANet y el seguimiento quedan fuera del primer incremento.

## Entradas y recepción de archivos

- Videos originales: ~/home.agents/data/marchaUniversitaria241002, lectura solamente.
- dataset: enlace existente a ~/data/DroneCrowd; preservarlo y no usarlo como salida.
- Pesos: artefacto local separado de Git, con procedencia y SHA256.
- Parámetros: video elegido, variante del modelo, muestreo, tamaño/crops,
  normalización y escala de las salidas, todos explícitos en el manifiesto.

Al preparar este traspaso la carpeta de subida aún no existía. Es una observación
de ese momento, no prueba de fallo de transferencia. Confirmar finalización por
el mecanismo de subida o por el autor. Tamaño/mtime estables y ffprobe son
comprobaciones adicionales; una pausa de la transferencia no demuestra finalización.
No mover, renombrar, borrar, recodificar ni escribir junto a los originales.

Inventariar nombre, tamaño, duración, dimensiones, FPS/base de tiempos, rotación,
codec e integridad de lectura. Registrar SHA256 después de la subida; si cambian
los bytes, invalidar el manifiesto y esperar. No cargar el video entero en RAM
ni extraer todos sus fotogramas.

## Recursos y autorización

El autor autoriza preparar y ejecutar esta prueba local acotada. No hace falta
otra autorización ritual para el adaptador o la inferencia dentro de este contrato.
No crear una nueva sesión automáticamente: el autor abrirá la sesión del repo.

Antes de nuevas descargas, instalaciones o cargas grandes: estimar disco
transitorio/retención, RAM y GPU, consultar el preflight de Nexo y comprobar
cuotas. Mantener la reserva de20GiB en el disco usado y2GiB RAM disponible.
La retirada verificada de mapas históricos liberó unos404GiB; se observaron
413GiB disponibles al retomar esta prueba. Reconsultar recursos antes de cada
carga. No borrar más datos como efecto implícito de la evaluación.

Primera ejecución: **un video, máximo30 fotogramas, lote1, máximo120s**.
Usar resolución/crops pequeños, declarados y compatibles con el modelo.
RAM inicial High1,5GiB/Max2GiB/swap0,5GiB, bajo el pool Nexo existente;
si no alcanza, registrar fallo y revisar tamaño/consumo, sin ampliar por defecto.
GPU: comprobar ocupación, empezar con tamaño pequeño; no hay cuota VRAM verificada,
por lo que no prometer aislamiento o reserva. No lanzar trabajos en paralelo.

Entorno ml compartido: sólo lectura/ejecución. Si falta decodificación u otra
dependencia, crear entorno propio y fijar sus versiones sin actualizar ml.
Todos los temporales/cachés bajo ~/home.agents/tmp; no /tmp ni /root ni cambios
en HOME/CODEX_HOME. No drivers, paquetes globales, servicios, APIs facturadas,
LLM adicionales, entrenamiento completo o campañas por esta tarea.

## Pesos y adaptación

El README de STNNet enlaza el archivo público de modelos:
https://drive.google.com/file/d/1H-GrqyYxERLzVM5wQblUDeX24Ht2lD6q/view
Su página respondió200 como models.zip en la preparación; **no fue descargado
ni se comprobó su contenido**. Comprobar variante, tamaño y presupuesto antes de
descargar; mantener artefactos y descompresión fuera de Git, con límites de tamaño
y rutas. No descargar el dataset de entrenamiento.

Cargar checkpoint con deserialización restringida, map_location explícito y
validación de nombres/formas. No sustituir pesos por parámetros aleatorios ni
asignarlos por posición para forzar compatibilidad. Un prefijo conocido puede
normalizarse mediante una regla explícita y probada. Si falta una clave, variante
o procedencia, informar incompatibilidad en vez de producir conteos aparentes.

Reusar el modelo adaptado con pretrained_backbone=False sólo al cargar sus pesos
completos. Revisar los loaders y el evaluador histórico para conservar RGB,
normalización, geometría de entrada y factor de entrenamiento de densidad.
Derivar la escala del código/checkpoint; no inventar una constante para que el
conteo parezca plausible. Separar entrada/salida de video del evaluador benchmark,
que todavía exige anotaciones. No generar HDF5 masivos para esta inferencia.

## Entrega verificable

1. Adaptador CLI reproducible con argumentos de video, pesos y límites, sin rutas
   personales hardcodeadas. Resumen de recursos antes de ejecutar.
2. Muestreo temporal explícito, con timestamps reales; un conteo por fotograma.
   No sumar fotogramas para afirmar personas únicas en toda la marcha.
3. Muestra visual de fotogramas y mapas, tiempos y conteos, sin métricas de
   precisión hasta disponer de anotaciones. Declarar qué región se procesó y
   límites por resolución, oclusión y diferencia de dominio.
4. Manifiesto privado: hashes de video/pesos/código, versiones, parámetros,
   duración, recursos observados y fallos. Conservar resultados negativos.
5. Pruebas con entrada sintética, límites, checkpoint incompatible y salida
   reproducible; luego una prueba con el archivo real bajo las cuotas.

Resultados locales en outputs/video-breve/<ejecucion>/, excluidos de Git;
presupuesto inicial de salidas128MiB. Evitar video de salida completo en este paso.
Mantener código, pruebas y documentación de reproducción en commits atómicos.
No subir videos, fotogramas, pesos o resultados privados a Git, web o terceros.
Los permisos locales no autorizan push al fork ni publicación.

## Tiempo y condiciones de parada

La evaluación inicial del entorno ya quedó hecha. Acotar la siguiente revisión
de pesos/adaptador a aproximadamente1h de trabajo antes de reevaluar si vale la
pena seguir. La demo podría requerir2–4h si los pesos son compatibles; no es una
promesa ni una medición de GPU. No extenderlo a un port de varios días por defecto.

Detener el paso afectado si la subida no terminó, faltan pesos fiables, no alcanza
la reserva/cuota, hay incompatibilidad o se exceden tiempo/salidas. Conservar
evidencia y continuar sólo tareas independientes. No entrenar para resolver
rápidamente la ausencia de pesos; registrar esa alternativa como fase aparte.

## Decisión sobre el dataset

El 24/10/2026 se completó la [validación y retirada de mapas históricos](../002-retencion-hdf5/SPEC.md),
autorizada por el autor como tarea separada. Se retiraron 50.239 HDF5 tras
leer y comparar todos sus valores con su regeneración; se liberaron unos
404 GiB. Se conservaron seis archivos incompletos/dañados, todas las imágenes,
anotaciones, código y resultados propios. Los MAT de secuencia tienen copia
adicional y un manifiesto permite regenerar mapas incluso después del borrado.
La inferencia adaptada de este video no necesita esos mapas de entrenamiento.

Ejecutar la prueba no borra datos. Una ejecución fallida tampoco demuestra que
ningún modelo pueda servir. Entregar recomendación de seguir/ajustar/retirar,
esfuerzo restante y posible liberación de espacio, separados del resultado técnico.

## Aceptación y traspaso

La sesión nueva puede empezar por PLAN.md. Debe cerrar con comando reproducible,
evidencia CPU/GPU y de video diferenciadas, revisión independiente cuando esté
disponible, commits por incremento y estado preciso de los bloqueos.
Si falta una entrada, el cierre es parcial; no presentar la SPEC como demo terminada.

## Continuación del 24/10/2026

El autor pide probar el modelo en los videos recibidos. Se identificaron dos MOV
4K (22,856 s y 9,393 s) y un MP4 de 960×540 (10,985 s). Primera muestra:
un video, tres timestamps al 10/50/90 % de duración, entrada de 960×540 y cuatro
cuadrantes secuenciales. Mantener máximo30fotogramas/120s/lote1. Sin anotaciones
de estos videos, los resultados son conteos exploratorios, no precisión medida.

La implementación separa recepción/muestreo (FFmpeg/FFprobe del sistema) y
conteo (ml compartido, PyTorch/NumPy). No instalar Pillow/OpenCV/torchvision en ml.
Guardar timestamps PTS reales, hashes, RGB uint8 y geometría explícita; la
normalización se aplica en memoria al entrar al modelo.
El evaluador histórico usa `sum(max(0, sum(cuadrante))) / 100`, cabeza den1;
no sumar cabezas/fotogramas ni aplicar una corrección de área por resize.

Preflight Nexo consultado sin modificar su repositorio: scratch1GiB,
retención0,5GiB, RAM1GiB, paralelismo1; permitido con margen30 %. La GPU mostró
148MiB usados sobre12GiB. Aplicar además scope con límites kernel, sin ampliar
las cuotas existentes. El venv ml permanece sin modificaciones.
La duración dura de120s la impone `timeout` externo dentro del scope; los
chequeos internos entre fotogramas no interrumpen por sí solos una llamada larga.

Bloqueo de pesos observado: el enlace original de los autores muestra
`models.zip` (349 MB), pero la confirmación de descarga responde **Google Drive
Quota exceeded**. No se descargó un checkpoint ni se ejecutó inferencia real.
No hay pesos STNNet en este checkout ni en las rutas locales consultadas.
Se pidió al autor una copia/ruta alternativa y se revisaron releases/incidencias
del upstream sin encontrar una distribución alternativa verificable.

### Entrada y muestreo comprobados

`scripts/sample_video.py` usa FFmpeg/FFprobe y biblioteca estándar. Cuatro pruebas
del muestreador pasaron, incluyendo decodificación real de una entrada sintética,
timestamps, preservación del original y límite de espacio que contempla PNG
incompresible. Los tres videos aportados se decodificaron en tres posiciones:

| Video | PTS reales de las muestras (segundos) |
| --- | --- |
| DJI 0265 | 2,285617; 11,428083; 20,570550 |
| DJI 0269 | 0,950950; 4,704700; 8,458450 |
| trim | 1,110000; 5,496667; 9,901667 |

Se observaron y registraron en kernel High1,5GiB/Max2GiB/swap0,5GiB en un scope
propio de DroneCrowd bajo `nexo-projects.slice` (pool8GiB). El lanzador genérico
con proyecto `nexo` rechazó una admisión por su propio uso contabilizado; no se
amplió esa cuota. Se aplicaron los límites específicos ya previstos para
DroneCrowd directamente bajo el pool compartido.

La extracción final, con el código revisado, tomó 4,53s/5,00s/1,49s por video;
los nueve hashes RGB coinciden con la primera extracción. Esto prueba lectura
de las muestras seleccionadas, no decodificación exhaustiva ni precisión.
Los originales se mantuvieron sin modificaciones. El venv ml tampoco se alteró.

Manifiestos privados: `outputs/video-breve/20261024-validado/{0265,0269,trim}/entrada/input.json`.
Lámina de entrada: `outputs/video-breve/20261024-entradas.png`, rotulada como
inferencia pendiente. Estado: `outputs/video-breve/20261024-estado.json`.
Revisión independiente: Codex `/root/revision_retencion_hdf5`, modelo efectivo
no expuesto; aprobó muestreador, guardas, pruebas y separación de evidencia.
El lanzador ml fija `~/home.codex/tmp`, alias físico del temporal autorizado.

### Adaptador de conteo preparado; inferencia pendiente

`scripts/count_video.py` valida el manifiesto y los RGB antes de leerlos, acepta
sólo un checkpoint completo de densidad por nombres/formas/tipos, verifica
SHA-256 y usa `torch.load(weights_only=True, map_location="cpu")`. Normaliza
únicamente el prefijo inicial `module.` con rechazo de colisiones. No copia
pesos por posición ni descarta silenciosamente claves de otras variantes.

Aplica RGB/ImageNet, cuatro cuadrantes secuenciales y el factor100 del evaluador
histórico. Guarda conteos por PTS, sumas crudas por cuadrante, mapas NPY y un
manifiesto de ejecución/fallo; no suma personas únicas ni mide precisión. Rechaza
entradas fuera de960×540, más de30fotogramas y valores/sumas no finitos.

Doce pruebas de muestreo/conteo pasaron en CPU, bajo límites efectivos de
High1,5GiB/Max2GiB/swap0,5GiB. Incluyen un estado sintético completo con prefijo
`module.`, errores de claves/formas/tipos/NaN, hash, geometría, RGB, overflow de
sumas, timestamps y preservación del original. Los estados sintéticos se usan
únicamente en tests; no produjeron estimaciones sobre los videos aportados.
El revisor independiente aprobó las correcciones y ambos adaptadores.

La siguiente ejecución necesita el ZIP oficial o un checkpoint den compatible,
con procedencia y hash. El autor aportó `~/home.agents/data/models.zip` y confirmó
la finalización de su subida. Se conserva como entrada; la verificación del
archivo y la inferencia se registran como el siguiente incremento.

Una vez verificados sus pesos, el comando reproducible de la primera muestra
es el siguiente (sustituir ruta y SHA-256 por los valores comprobados):

```sh
systemd-run --user --scope --collect --slice=nexo-projects.slice \
  -p MemoryHigh=1610612736 -p MemoryMax=2147483648 \
  -p MemorySwapMax=536870912 -p OOMPolicy=kill -- \
  timeout --signal=TERM --kill-after=5s 120s \
  bash ops/python-ml.sh scripts/count_video.py \
  --input outputs/video-breve/20261024-validado/0265/entrada/input.json \
  --checkpoint /ruta/comprobada/den_model_best.pth.tar \
  --checkpoint-sha256 SHA256_COMPROBADO \
  --checkpoint-source 'https://drive.google.com/file/d/1H-GrqyYxERLzVM5wQblUDeX24Ht2lD6q/view' \
  --output-dir outputs/video-breve/20261024-validado/0265/inferencia \
  --device cuda
```

Reconsultar preflight y ocupación GPU antes de lanzarlo; no ejecutarlo con pesos
incompletos, aleatorios o procedencia desconocida. El comando no crea cuotas
persistentes ni amplía los límites del pool compartido.
