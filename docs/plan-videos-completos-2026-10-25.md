# Plan: tres modelos sobre todos los cuadros disponibles

25/10/2026. **Reservar 6–8 horas de ejecución para los tres clips**, más la espera
en cola, una vez adaptado y validado el ejecutor. Con los procesos de dos cuadros
actuales serían aproximadamente 12–15 horas. Estas son proyecciones desde tiempos
medidos, no una campaña ya iniciada; implementar y revisar el ejecutor lleva un
trabajo previo no incluido en esas horas.

## Qué se procesaría

| Archivo | Duración | Resolución original | Cuadros declarados |
| --- | ---: | --- | ---: |
| 0265 | 22,856 s | 3840×2160 | 1370 |
| 0269 | 9,393 s | 3840×2160 | 563 |
| trim | 10,985 s | 960×540 | 659 |
| Total | 43,234 s | | 2592 |

Se procesarían los tres archivos enteros: **2592 cuadros × 3 modelos = 7776
predicciones**. La prueba completada cubrió 44 instantes a 1 Hz; esta propuesta
cubre cada cuadro nativo. Los tres modelos son SR, CLIP-EBC RN50 y MPCount,
con los mismos pesos y recetas de esa prueba. El trim conserva su recorte inferior
de 12 filas para comparar 960×528. No se incorpora upscaling ni entrenamiento.
Antes de ejecutar se confirmará el número de cuadros decodificando los archivos;
por ahora procede del metadato del contenedor, contrastado con sus hashes.

## De dónde sale el tiempo

Los 69 procesos anteriores permiten estimar por separado inferencia por cuadro
y carga/preparación por proceso. La inferencia extrapolada suma 4,51 h.
Agrupar 16 cuadros consecutivos por carga de pesos, procesados **uno por vez en
GPU**, reduce la cantidad de procesos de 3891 a 492. El presupuesto base pasa de
10,51 h a 5,27 h, antes de márgenes operativos.

| Modelo | Base estimada con bloques de 16 |
| --- | ---: |
| SR | 2 h 29 min |
| CLIP-EBC RN50 | 1 h 48 min |
| MPCount | 59 min |
| Total | 5 h 16 min |

Recomiendo reservar **6–8 h**, contemplando preparación, escritura, admisión y
videos de revisión. No incluye espera por otras tareas de GPU, implementación ni
revisión del ejecutor. El tiempo puede cambiar con el piloto o con la carga del
disco; el número de plazas globales del dispatcher no multiplica los recursos
permitidos a este proyecto.

## Secuencia de trabajo

1. Adaptar el lector para mantener sólo un RGB y su mapa en RAM, conservando el
   modelo durante 16 cuadros. El ejecutor actual todavía no permite esta campaña.
2. Congelar inventario por ordinal/PTS real y hashes. Preparar un bloque en CPU
   y terminar el decodificador antes de comenzar GPU.
3. Revisar y proponer nueve perfiles fijos, uno por archivo/modelo, con las mismas
   cuotas de RAM y tiempo. El dispatcher ya reconoce al propietario `DroneCrowd`;
   falta registrar los perfiles del ejecutor futuro. El paquete antiguo de
   promoción del registro ya no corresponde a la integración actual.
4. Probar un bloque 4K de 16 cuadros con los tres modelos. Comprobar memoria,
   tiempo, cancelación/reanudación, admisión entre procesos y resultados contra
   los anclajes idénticos anteriores. Es uno de los 164 bloques: sus tres trabajos
   ya están incluidos en los 492. Ajustar la estimación con esa medición.
5. Si pasa la revisión, completar los 164 bloques: hasta tres solicitudes por
   bloque, secuenciales por proyecto, con recibos y reanudación verificable.
6. Revisar mapas y curvas por escena, y generar tres videos locales con los tres
   conteos superpuestos, un CSV por cuadro y un informe de límites y discrepancias.

## Disco y utilidad del resultado

Conservar todos los mapas float32 para auditoría consumiría unos **63 GiB**.
El plan reserva 80 GiB para entregables, más scratch, margen y 20 GiB intocables:
exige **128 GiB libres** antes de empezar. Había unos 388 GiB libres al calcularlo;
se volverá a comprobar. El SSD externo no hace falta para estos tres clips.
Los RGB temporales se mantienen por bloque (menos de 380 MiB sin comprimir),
evitando materializar otros 46 GiB de imágenes. No se borran datos anteriores.

La salida responde «cuántas personas estima cada modelo en este encuadre e
instante». Los cortes, movimientos de cámara y zoom se marcarán para revisar las
variaciones. No se suman conteos de distintos cuadros o planos como asistentes.
Procesar más cuadros permite estudiar estabilidad temporal; para elegir un modelo
por precisión sigue haciendo falta una referencia manual exhaustiva.

La [SPEC017](../sdd/017-videos-completos/SPEC.md) contiene fórmulas, recursos,
aceptación y revisión independiente. El inventario, estimador CPU y hashes de los
69 informes usados quedan localmente en `outputs/plan-completo-20261025/`.
Videos, mapas, pesos y anotaciones personales permanecen fuera de Git.
