# Prueba breve de STNNet con los videos de la marcha

## Objetivo y encargo vigente

El autor está subiendo videos a **~/home.agents/data/marchauniversitaria** y
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

- Videos originales: ~/home.agents/data/marchauniversitaria, lectura solamente.
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
En el traspaso se observaron unos9,4GiB libres: preparar código/metadatos sigue
siendo posible; no sortear la reserva ni iniciar una carga grande por esta SPEC.
Nexo coordina la recuperación de espacio. No borrar el dataset para desbloquearse.

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
