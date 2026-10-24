# DroneCrowd: recuperación del experimento

El autor autoriza el 24/10/2026 trabajar y commitear en su fork personal
git@github.com:kenarab/DroneCrowd.git, trasladado al sandbox. El upstream es
https://github.com/VisDrone/DroneCrowd. Este proyecto conserva el código científico,
los adaptadores y la aceptación sobre el video. Nexo conserva capacidad, cuotas y
coordinación; el traslado no registra automáticamente un productor Telegram,
ejecutor o proyecto OpenProject.

Primero se conservaron por separado el renombre de requirements y los pines
históricos del autor. Los dos wheels locales corresponden a CPython3.7 y
CUDA10.1, son instaladores (no pesos del modelo) y permanecen fuera de Git.
El enlace dataset apunta a los datos existentes, que permanecen intactos.

El objetivo práctico es evaluar un video con pesos existentes. Entrenar aprende
parámetros desde ejemplos anotados; inferir aplica pesos existentes a otro video.
Los mapas de entrenamiento HDF5 no son un requisito de una futura inferencia
adaptada a video. Los scripts mytest originales también calculan métricas del
benchmark y no son todavía una interfaz de video independiente.

[Plan y criterio de cierre](../sdd/001-video-breve/SPEC.md).
[Entorno y diagnóstico](entorno.md).

Pedido vigente del autor: continuar desde este repo con videos que están subiendo
a ~/home.agents/data/marchauniversitaria. [Traspaso operativo](traspaso-video-marcha-2026-10-24.md).
La nueva sesión comprobará finalización de la subida, pesos y recursos antes de
ejecutar la muestra; no se lanzó inferencia real desde Nexo.
