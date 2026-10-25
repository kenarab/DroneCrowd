# DroneCrowd: recuperación del experimento

## Investigación vigente

El alcance se amplía a investigación reproducible sobre medición de multitudes
en manifestaciones argentinas, inicialmente Buenos Aires. Se compararán métodos
recientes con STNNet bajo protocolos explícitos y revisión independiente. Ningún
modelo está validado todavía como estimador de la asistencia total a una marcha.

- [Marco SDD y criterios de reproducibilidad](../sdd/000-investigacion-reproducible/SPEC.md).
- [Bibliografía y selección de candidatos](investigacion/README.md).
- [Revisión conjunta de los ajustes rechazados](revision-conjunta-entrenamiento-2026-10-25.md).
- [Cómo mirar y corregir las anotaciones provisionales](revision-visual-anotaciones-2026-10-25.md).
- [Modelo original sobre las ocho fotos completas](inferencia-fotos-completas-2026-10-25.md).
- [Primera prueba SR, tamaños de pesos y tiempos de video](prueba-sr-2026-10-25.md).

## Recuperación y primeras pruebas

El autor autoriza el 24/10/2026 trabajar y commitear en su fork personal
git@github.com:kenarab/DroneCrowd.git, trasladado al sandbox. El upstream es
https://github.com/VisDrone/DroneCrowd. Este proyecto conserva el código científico,
los adaptadores y la aceptación sobre el video. Nexo conserva capacidad, cuotas y
coordinación; el traslado no registra automáticamente un productor Telegram,
ejecutor o proyecto OpenProject.

Primero se conservaron por separado el renombre de requirements y los pines
históricos del autor. Los dos wheels locales corresponden a CPython3.7 y
CUDA10.1, son instaladores (no pesos del modelo) y permanecen fuera de Git.
El enlace dataset apunta a los datos existentes; se conservan las imágenes,
anotaciones y resultados propios.

El objetivo práctico es evaluar un video con pesos existentes. Entrenar aprende
parámetros desde ejemplos anotados; inferir aplica pesos existentes a otro video.
Los mapas de entrenamiento HDF5 no son un requisito de una futura inferencia
adaptada a video. Los scripts mytest originales también calculan métricas del
benchmark y no son todavía una interfaz de video independiente.

[Plan y criterio de cierre](../sdd/001-video-breve/SPEC.md).
[Entorno y diagnóstico](entorno.md).

Pedido vigente del autor: probar STNNet sobre los videos aportados en
~/home.agents/data/marchaUniversitaria241002. Se prepararon nueve fotogramas
verificados y el adaptador de conteo, con pruebas y límites de memoria/tiempo.
El autor aportó los pesos tras la cuota excedida en Drive. La inferencia
entrenada funcionó en nueve fotogramas; la configuración a 960×540 devolvió 20–82
personas y resulta visualmente insuficiente para esa multitud. No usar estos
valores como estimación de convocatoria.
[Resultados](resultado-video-marcha-2026-10-24.md),
[estado y reproducción](../sdd/001-video-breve/SPEC.md).

Continuación autorizada: prueba nativa 4K y Lanczos 2×/4×, siete configuraciones
completas. Hay más respuesta con detalle nativo, pero los conteos siguen siendo
visualmente insuficientes y no están validados como convocatoria.
[Comparación de resolución](resultado-resolucion-original-2026-10-24.md).

## Retención de mapas históricos

El 24/10/2026 se verificaron y retiraron 50.239 HDF5 generados para entrenamiento,
liberando unos 404 GiB. Se conservaron seis excepciones y todos los originales,
anotaciones y resultados propios. La prueba de video no requiere esos mapas;
para volver a entrenar habrá que regenerarlos. Hay copia de los MAT y un
manifiesto privado con hashes, y se probó regenerar después del borrado.
[Especificación, evidencia y procedimiento](../sdd/002-retencion-hdf5/SPEC.md).
