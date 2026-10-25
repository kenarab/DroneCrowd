# Procesamiento de todos los cuadros de los tres clips

25/10/2026. Estado: **planificado; ejecutor nuevo y campaña sin ejecutar**.
Pedido del autor: planificar los videos completos con SR, CLIP-EBC RN50 y
MPCount, e informar cuántas horas llevaría. Esta especificación prepara esa
campaña; no inicia GPU ni aplica perfiles compartidos.

## Alcance y entradas

Procesar cada cuadro nativo decodificado de los mismos tres archivos de SPEC014,
con sus mismos pesos y recetas científicas. Conservar originales y salidas
anteriores. Los archivos disponibles suman 43,233884 s, no horas de filmación.
«Completo» significa todos sus cuadros, frente al muestreo anterior de 1 Hz.

| Fuente | Duración (s) | Cuadros declarados | Geometría de inferencia | Bloques de hasta 16 |
| --- | ---: | ---: | --- | ---: |
| 0265 | 22,856167 | 1370 | 3840×2160 | 86 |
| 0269 | 9,392717 | 563 | 3840×2160 | 36 |
| trim | 10,985000 | 659 | 960×528 | 42 |
| Total | 43,233884 | 2592 | | 164 |

Los 659 cuadros del trim son 960×540 originales con las mismas 12 filas inferiores
excluidas en la prueba anterior. No agregar superresolución ni cambiar recortes,
ventanas o normalizaciones. Los totales proceden de `nb_frames` del contenedor;
la decodificación completa debe confirmarlos antes de congelar la campaña.
Si discrepan, actualizar inventario, estimación y revisión; no omitir cuadros.

Se verificaron nuevamente los SHA-256 originales contra el plan congelado
93405aba21aaef66f4a9bc85615d3de3e34ce267c382725b3e7455170adbd73b.
El inventario privado y la estimación quedan en
`outputs/plan-completo-20261025/estimate.json`, SHA-256
c6290336c9b9cd302b8a797378818ae0cd9f92ac9119c87baa3a44d7c20f19f0.
Su generador CPU se conserva como `estimator.py` en esa carpeta, SHA-256
754eb30de65e2db1f8790558cae75bc4027e64c49b21c3056568d399f0ba9e0f.
No se materializaron cuadros nuevos ni se lanzó GPU para calcular el presupuesto.

## Presupuesto reproducible

Base: 69 procesos completos, 44 cuadros por modelo y recibos de SPEC016.
Para cada fuente y modelo se toma la media medida de inferencia por cuadro.
El coste fijo estimado por proceso es la media de su duración de unidad menos
la suma de sus tiempos de inferencia: SR 4,375428 s; CLIP-EBC 7,304366 s;
MPCount 4,986582 s. Incluye carga/importaciones, preparación y escritura medidas;
tratarlo como fijo es una aproximación que debe contrastar el piloto.

| Fuente | SR (s/cuadro) | CLIP-EBC (s/cuadro) | MPCount (s/cuadro) |
| --- | ---: | ---: | ---: |
| 0265 | 4,154329 | 2,644527 | 1,376988 |
| 0269 | 4,174080 | 2,697337 | 1,365861 |
| trim | 0,256020 | 0,227860 | 0,112701 |

`T = suma_fuente,modelo(N_cuadros × media_inferencia + ceil(N_cuadros/B) × coste_fijo)`.
El lote GPU sigue siendo 1; B es el número de cuadros consecutivos por proceso.
Extrapolar únicamente la inferencia da 4,50879 h. Con B=16 son 492 procesos,
5,26803 h de base: SR 2,47992 h; CLIP-EBC 1,80269 h; MPCount 0,98542 h.
Reservar **6–8 h de ejecución**, incluyendo margen para preparación, escritura,
admisión y montaje de entregables. La espera por otros proyectos se añade.
No es una medición del ejecutor futuro, ni incluye su implementación y revisión.

Con B=2, como la tanda anterior, serían 3891 procesos y 10,51331 h de base;
reservar 12–15 h. El mayor cuadro SR medido tardó 4,221189 s: 16 equivalen a
67,54 s de inferencia, antes de preparación y escritura. El margen de tiempo
parece suficiente, pero sólo un piloto puede confirmar el presupuesto de 120 s.

## Ejecutor y cola propuestos

1. Implementar un lector acotado y un modelo persistente durante cada bloque.
   El contador actual admite hasta tres cuadros en su manifiesto; elevar ese
   límite no basta. Mantener en RAM sólo un RGB y su mapa, escribir/verificar
   antes de liberarlos y continuar. Conservar lote GPU 1 y un hilo/núcleo CPU.
2. Preparar en CPU un bloque de hasta 16 cuadros consecutivos con ordinal,
   PTS original, `time_base`, geometría y SHA RGB. Terminar el decodificador
   antes de admitir GPU; no compartir su pico con el proceso de inferencia.
   Usar decodificación acotada por bloque/GOP, evitando buscar por cada cuadro.
3. Usar el dispatcher activo, propietario exacto **`DroneCrowd`**, ya reconocido
   mediante el registro de recursos. El identificador científico histórico
   `dronecrowd` del plan anterior no exige renombrar el propietario de la cola.
   No aplicar la promoción/retiro del paquete de alta histórico de SPEC015.
4. Proponer nueve perfiles fijos: uno por fuente/modelo, con `max_runs` 86, 36 o
   42 según fuente. Cada perfil fija entrypoint, lock, revisión de 40 caracteres,
   argv y hash del nuevo plan. El wrapper seleccionará únicamente un bloque
   aprobado mediante `NEXO_GPU_REQUEST_ID`; sin argumentos libres ni shell.
   No crear 492 perfiles: se excedería el límite de tamaño de la política.
5. Solicitar los tres modelos del bloque preparado y esperar sus resultados
   verificados antes de liberar sus RGB y preparar el siguiente. Máximo tres
   solicitudes pendientes, por debajo del límite de 32. Ejecución secuencial
   por las cuotas del proyecto; cuatro plazas globales no cuadruplican su RAM.
   Registrar aceptación, estado y resultado real; un recibo no acredita CUDA.

Los perfiles sólo se podrán generar tras implementar, probar, revisar y commitear
el ejecutor. Infraestructura aplica la configuración; este plan no cambia registros
ni unidades compartidas. Preservar HEAD y archivos rastreados durante la campaña.

## Recursos y retención

Mantener RAM High 1,5 GiB/Max 2 GiB, swap 512 MiB, timeout externo 120 s más
terminación hasta 10 s e interno 115 s; lote GPU 1. Propuesta de admisión:
RAM estimada 1,5 GiB, VRAM 1024 MiB, scratch 0,5 GiB y salidas 1 GiB por trabajo.
Confirmar estas estimaciones y picos en el piloto; si no cumplen, detener y
rediseñar bloques con revisión, sin ampliar límites automáticamente.

Los RGB de 16 cuadros 4K suman 379,69 MiB en disco, pero no se acumulan en RAM.
Los 16 mapas MPCount float32 suman 506,25 MiB: el perfil anterior de salidas
0,25 GiB resulta insuficiente. Mantener el límite de 64 MiB por archivo y logs
acotados. Temporales, cachés y staging exclusivamente en `~/home.codex/tmp`.
Retener los mapas científicos en `outputs/`, con informes atómicos y hashes.

Todos los 7776 mapas representan unos **62,88 GiB** de payload sin compresión;
headers, recibos, CSV y videos añaden espacio. Reservar 80 GiB para entregables,
2 GiB de scratch agregado, margen del 30 % y 20 GiB libres intocables:
preflight mínimo redondeado de **128 GiB libres**, además de 2 GiB RAM host
disponible más margen de admisión. El disco tenía unos 388 GiB libres al preparar
el plan; repetir medición antes de ejecutar. El SSD externo no es necesario para
estos tres clips con esta política, aunque puede alojar una copia posterior.

Conservar todos los mapas hasta revisión de calidad; decidir después una retención
curada. Materializar todos los RGB supondría otros 45,73 GiB y se evita mediante
la caché de un bloque. Suprimir sólo esa caché regenerable tras verificar los tres
resultados del bloque; nunca originales, anotaciones ni resultados anteriores.

Antes de cada admisión verificar recursos y cuotas efectivas. Resolver con el
operador la caché de la slice vacía observada en SPEC016: mantener comprobaciones
de procesos/anon/shmem, reclamación propia acotada y admisión posterior, sin
`drop_caches` global ni ampliar cuotas. Probarlo con el dispatcher antes de una
campaña larga; no sustituir la cola por un lanzador directo.

## Tiempo, escenas y entregables

Preservar ordinal y PTS real; no asumir CFR, eliminar PTS repetidos ni usar un
filtro `fps`/`-r` que duplique o descarte cuadros. Confirmar identidad de los RGB
contra los 44 anclajes previos usando la misma conversión. FFmpeg documenta
`fps_mode=passthrough` y los posibles cambios del muxer; conservar además un
manifiesto de PTS independiente. Referencias:
[FFmpeg](https://ffmpeg.org/ffmpeg.html) y
[FFprobe: cuadros y conteo](https://ffmpeg.org/ffprobe.html).

Entregar CSV con 2592 filas y tres conteos por fila (7776 predicciones), ordinal,
PTS, fuente/escena y procedencia de código, RGB y pesos; tres videos locales
anotados con los tres conteos por instante; curvas y mapas verificables.
Marcar cortes, transiciones y zoom para interpretar las curvas, conservando sus
cuadros. Nunca sumar planos, cuadros ni clips como personas distintas o asistencia.
Un recorrido completo permite examinar estabilidad; no prueba precisión sin GT.

## Puertas de aceptación antes de ampliar

- Inventario por decodificación real, correspondencia ordinal/PTS/RGB, hashes de
  fuentes/pesos/lock y nuevo plan inmutable comprobados en CPU.
- Pruebas CPU de límites, selección de solicitudes, salida parcial, reanudación,
  integridad y ausencia de omisiones/duplicados en fronteras de bloques.
- Piloto de un bloque 4K de 16 cuadros con los tres modelos por dispatcher;
  elegir uno de los 164 bloques congelados: sus tres solicitudes cuentan dentro
  de las 492 y de `max_runs`, no constituyen una tanda adicional.
  Comprobar scope, tiempos, picos, contadores de memoria y mapas finitos/sumas.
  Comparar anclajes idénticos con la receta anterior: tolerancia de conteo
  `max(0,01; 1e-5 × abs(conteo_referencia))`, sólo tras igualdad de hash RGB.
- Cancelación y reanudación verificadas: resultado completo íntegro se omite;
  parcial se conserva, no se sobrescribe ni se reintenta automáticamente.
  Probar admisión entre bloques y recuperación tras interrupción del coordinador.
- Revisión independiente del ejecutor/perfiles y actualización de tiempos con
  el piloto; ante desviaciones detener antes de admitir los otros bloques.
- Cierre futuro: todos los ordinales y modelos presentes una sola vez, todos
  los mapas verificados, recursos/tiempos reales y límites científicos publicados.

## Revisión de esta planificación

Revisor independiente Codex, agente existente `revision_retencion_hdf5`, modelo
efectivo no identificado. Reprodujo los cálculos desde los 69 informes y recibos:
4,50879 h de inferencia, 5,26803 h con bloques de 16, 10,51331 h con bloques de dos
y 62,87762 GiB de mapas. Señaló preservar PTS, no acumular RGB en RAM y ampliar
la estimación de salidas por trabajo, incorporados arriba. En la revisión final
confirmó hashes, fórmulas y contrato vigente del dispatcher; pidió explicitar que
el piloto consume tres de las 492 solicitudes y marcar todo el procedimiento de
alta anterior como histórico. Incorporado. Sin bloqueantes de planificación;
esto no aprueba todavía un ejecutor ni una campaña GPU.

[Plan de lectura y presupuesto](../../docs/plan-videos-completos-2026-10-25.md).
