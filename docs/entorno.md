# Entorno para una primera prueba

## Estado observado el 24/10/2026

El venv compartido ~/home.agents/venvs/ml contiene Python3.13.13,
PyTorch2.14.1 (runtime declarado CUDA13.0), NumPy2.5.4, SciPy1.18.1 y
scikit-learn1.9.1 según metadatos. No contiene torchvision, Pillow, h5py,
OpenCV ni spatial-correlation-sampler. Esto no reproduce el entorno histórico.

STNNet documenta Python3.6, PyTorch1.6 y CUDA10.2. Los pines locales conservados
y los wheels CPython3.7/CUDA10.1 prueban un intento anterior, no que funcionara.
STANet documenta Python2.7 y PyTorch0.4; requiere una adaptación distinta.

La RTX3060 de12GiB se comprobó mediante nvidia-smi fuera del aislamiento.
Un error de acceso dentro del sandbox no demuestra un fallo del driver.

## Diagnóstico reproducible sin instalaciones

```sh
bash ops/python-ml.sh scripts/check_environment.py
bash ops/python-ml.sh scripts/check_environment.py --runtime
```

El primer comando lee metadatos. --runtime importa PyTorch y consulta CUDA,
sin pesos ni descargas. --gpu-smoke añade una operación de cuatro valores
en GPU; es un diagnóstico sintético, no inferencia del modelo. Ejecutarlo
con las cuotas del coordinador y permiso de acceso al dispositivo cuando proceda.
No instalar dependencias antiguas en ml. El lanzador prepara sólo temporales
propios; no cambia el venv, HOME ni CODEX_HOME.

## Reproducción y adaptación

Reproducir exactamente Python3.6/3.7 y dependencias2020 no es lo mismo que adaptar
el modelo al entorno actual. El camino corto a evaluar es STNNet con conteo,
pesos completos y PyTorch existente. El módulo importa torchvision y extensiones
de tracking incluso si sólo queremos conteo: hay que separar esas dependencias.
También inicializa VGG16 con descarga automática; una carga de checkpoint completo
debe poder omitir ese paso. El diagnóstico base posterior se describe al final de este documento.

El README de los autores enlaza un ZIP público de modelos; la página respondió200
con título models.zip. No se descargaron ni inspeccionaron pesos y no se conoce
aún qué variantes contiene. La prueba de video depende además del archivo que
está buscando el autor. No apareció un enlace Rolandi ni un video en las rutas
del proyecto y carpetas habituales consultadas; no se inspeccionó historial web
ni cuentas personales.

Fuentes:
- https://github.com/VisDrone/DroneCrowd/tree/master/STNNet
- https://pytorch.org/get-started/previous-versions/
- https://drive.google.com/file/d/1H-GrqyYxERLzVM5wQblUDeX24Ht2lD6q/view

## Diagnóstico real verificado

La importación de PyTorch y una operación de cuatro valores en CUDA pasaron
con el venv ml: Python3.13.13, PyTorch2.14.1+cu130, CUDA13.0 y RTX3060.
Se ejecutó mediante la cuota Nexo existente, con estimación0,6GiB, margen30%
y límite aproximado1,28GiB para el proceso. Una primera admisión con0,75GiB
fue rechazada por la ocupación existente de la slice; no se ampliaron límites.
No se instalaron paquetes ni se cargó el modelo DroneCrowd. Faltan las
dependencias específicas, la adaptación, pesos verificados y el video.

Revisión independiente de la infraestructura: Codex /root/revision_archivos,
modelo efectivo desconocido; sin bloqueantes. Revisó exclusiones Git, sintaxis
del wrapper y separación entre diagnóstico, futura inferencia y entrenamiento.
La comprobación GPU la realizó el coordinador, no el revisor.

## Núcleo STNNet compatible con ml

Se añadió pretrained_backbone=False para iniciar desde un checkpoint completo
sin descargar VGG16. El valor por defecto sigue siendo True para conservar
la inicialización histórica. Las dependencias de correlación y PointConv se
importan únicamente si se usa seguimiento. Así el conteo y la localización
pueden arrancar con PyTorch y NumPy del venv ml sin instalar otros paquetes.

Dos pruebas reales pasaron en CPU y RTX3060: conteo y localización, cada una con
dos imágenes sintéticas32x32. Comprobaron forma de todas las salidas, valores
finitos y ausencia de importaciones de tracking/torchvision en modo conteo.
Tiempo de la suite:2,377s. El scope propio del diagnóstico quedó dentro del pool
Nexo existente, con MemoryHigh1,5GiB, MemoryMax2GiB y swap0,5GiB; límites kernel
verificados. No se ampliaron límites del pool ni de otros proyectos.

Esto valida ejecución básica del núcleo con pesos aleatorios. No reproduce
el entorno2020 ni valida checkpoints de los autores, conteos, entrenamiento,
seguimiento o el video. Los scripts originales de benchmark siguen dependiendo
de sus loaders y anotaciones; el adaptador a video está pendiente.

Para repetir la suite, el coordinador debe aplicar un scope de recursos con
esos límites antes de ejecutar ops/python-ml.sh -m unittest discover -s tests -v;
DRONECROWD_TEST_CUDA=1 añade la GPU a las dos pruebas. No usar la suite para
decidir exactitud ni generar resultados científicos con parámetros aleatorios.

Revisión independiente del incremento de modelo: Codex /root/revision_archivos,
modelo efectivo desconocido; sin bloqueantes. Conserva firma previa, capas y
claves del state_dict y el valor predeterminado de inicialización. Revisión
estática; no ejecutó GPU ni certificó pesos o calidad.

## Entrada de video aportada por el autor

El autor informa una subida a ~/home.agents/data/marchauniversitaria y pide
continuar en una nueva sesión del fork. Reemplaza la búsqueda de copias antiguas
como siguiente paso. Al preparar el traspaso la ruta aún no existía; confirmar
finalización y verificar archivos. Pesos/adaptador e inferencia real siguen
pendientes. [SPEC vigente](../sdd/001-video-breve/SPEC.md).
