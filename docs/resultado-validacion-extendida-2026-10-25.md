# Tres modelos durante los tres clips disponibles

25/10/2026. **Completados 69 trabajos GPU, 44 cuadros y 132 predicciones/mapas**, con
verificación de hashes, formas, finitud y sumas. No es una medición
validada de asistencia: no hay referencia exhaustiva de personas por cuadro.

## Cobertura y reproducción

Se reutiliza el plan de [SPEC014](../sdd/014-validacion-larga-en-cola/SPEC.md),
SHA-256 `93405aba21aaef66f4a9bc85615d3de3e34ce267c382725b3e7455170adbd73b`:
44 instantes a 1 Hz, 23 lotes y 69 trabajos, uno por modelo/lote. Son 132 predicciones
por cuadro y sus mapas, no 2592 cuadros de video ni 69 escenas distintas.

| Video | Cuadros | Geometría común | Tiempos pedidos |
| --- | ---: | --- | --- |
| 0265 | 23 | 3840×2160 | 0–22 s |
| 0269 | 10 | 3840×2160 | 0–9 s |
| trim | 11 | 960×528 | 0–10 s |

El trim excluye 12 filas inferiores; conserva sus píxeles originales. No hubo
upscaling ni nuevas muestras elegidas por sus resultados. Los 69 descriptores se
definieron antes de ejecutar. Se conservan PTS efectivos y hashes RGB/entrada.

Pesos, normalizaciones y contextos idénticos a la
[comparación anterior](comparacion-tres-modelos-video-2026-10-25.md): SR con núcleos
256/halo 160, CLIP-EBC RN50 con ventanas 224 y promedio de solapes, MPCount con
parches disjuntos 512 y densidad dividida por 1000. Carga restringida, claves y
formas estrictas; sin entrenamiento ni selección de pesos por conteos deseados.

El snapshot del contador y adaptadores corresponde a 5471334 y sus hashes. El
control de ejecución procede de 848e5ee, con lanzadores operativos revisados y
conservados localmente. Python 3.13.13, Torch 2.14.1+cu130, NumPy 2.5.4 y el lock
de SPEC015. No se modificó el entorno ML compartido.

## Ejecución y evidencia

El dispatcher seguía sin el alta aplicada. El autor pidió ejecutar en GPU mientras
se integra el pipeline: [SPEC016](../sdd/016-prueba-directa-acotada/SPEC.md). La
ruta fue directa y acotada, sin recibos ni reservas ficticias de la cola.

Tres trabajos iniciales; después los 66 restantes. Cada uno usa una unidad propia
en la slice de DroneCrowd, un hilo/núcleo CPU, RAM High 1,5/Max 2 GiB, swap 512 MiB,
timeout 120 s y hasta 10 s de terminación; el contador limita su trabajo a 115 s.
Preflight fresco de disco, RAM, cuotas y GPU antes de cada proceso. Logs hasta
16 MiB por stream y archivos del hijo hasta 64 MiB. Las cuotas no se ampliaron.

Los primeros intentos del controlador se detuvieron entre procesos por admisión
de RAM y por una guarda de caché excesiva; no se repitieron inferencias completas.
La telemetría identificó 8192 bytes pendientes de escritura, sin procesos,
memoria anónima ni compartida, que la guarda exigía llevar a cero.

La corrección usa reclamación normal del kernel sólo sobre la slice propia
vacía, con `anon=0` y `shmem=0`, un pedido hasta 2 GiB y revalidación completa de
admisión después. Registra páginas pendientes antes/después; el kernel puede
recuperar más o menos bytes o devolver EAGAIN. Fsync de informes/densidades
propios verificados al cerrar cada proceso. Referencias primarias:
[contrato cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory) y
[implementación de reclamación](https://github.com/torvalds/linux/blob/master/mm/vmscan.c).
No se descartaron archivos ni se alteraron cuotas para resolver esa espera.

Artefactos privados, fuera de Git:

- `outputs/validacion-extendida-cola-20261025/results/`: informes atómicos y mapas
  por modelo/lote, con códigos, entradas, pesos, PTS, geometría y límites efectivos.
- `outputs/validacion-extendida-cola-20261025/summary/`: CSV por cuadro, resumen y
  gráficos; se escriben sólo si los 69 informes y 132 mapas verifican completos.
- `outputs/alta-dispatcher-20261025/`: recibos de ejecución directa, logs,
  reclamaciones y copias de los lanzadores revisados con hashes.

Los resultados completos se omiten únicamente tras comprobar hash de entrada,
pesos, código, todos los cuadros/PTS, métricas finitas y mapas. Una salida parcial
no se sobrescribe ni se reintenta automáticamente. La preparación congelada sigue
declarando «no enviada»; los recibos directos documentan lo ocurrido después.

## Interpretación

Predicciones por cuadro, redondeadas a personas enteras. La mediana resume los
instantes muestreados; el rango refleja escenas/encuadres, no un intervalo de
confianza ni una estimación de toda la marcha.

| Video | Modelo | Mínimo | Mediana | Máximo |
| --- | --- | ---: | ---: | ---: |
| 0265 | SR | 226 | 307 | 456 |
| 0265 | CLIP-EBC RN50 | 281 | 451 | 782 |
| 0265 | MPCount | 1249 | 1816 | 13677 |
| 0269 | SR | 183 | 217 | 267 |
| 0269 | CLIP-EBC RN50 | 955 | 1940 | 3179 |
| 0269 | MPCount | 1696 | 4879 | 5768 |
| trim | SR | 20 | 29 | 35 |
| trim | CLIP-EBC RN50 | 572 | 858 | 1149 |
| trim | MPCount | 560 | 625 | 766 |

MPCount cambia casi 11 veces entre mínimo y máximo del 0265, con una subida marcada
durante el acercamiento. La estabilidad observada antes en tres cuadros cercanos
en el tiempo no describe toda la secuencia. CLIP-EBC también cambia mucho en 0269;
en el trim da estimaciones mayores que MPCount, mientras SR sigue dando valores pequeños. Estos
resultados no seleccionan un ganador en precisión sin referencias exhaustivas.

El video 0265 pasa de una panorámica a un acercamiento marcado. Se comprobó el
mismo SHA de video que en la prueba previa, pero los PTS nuevos son distintos:
los tres cuadros previos estaban en 18,585/19,586/20,571 s. No son repeticiones
exactas de los cuadros a segundos enteros de esta tanda.

Analizar el conteo junto con encuadre/escala. Una curva creciente por sí sola no
prueba llegada de personas. Cambios de campo visual, escala de las cabezas, oclusiones y
posición respecto de los parches pueden cambiar las predicciones. Asociar un
cambio al zoom por inspección no identifica por sí solo su causa interna.

Las vistas de los tres videos tampoco prueban simultaneidad ni son sectores
disjuntos: no sumar cuadros, medianas, máximos ni clips como asistentes. Siguen
faltando máscaras de calle/área de interés, referencias locales exhaustivas y
comprobaciones controladas de sensibilidad a escala/parches. Un modelo que
responde con números mayores no queda validado por ese hecho.

## Tiempos, recursos y comprobación final

| Modelo | Procesos | Tiempo acumulado del contador | RSS máximo | Pico CUDA registrado |
| --- | ---: | ---: | ---: | ---: |
| SR | 23 | 189,27 s | 1,009 GiB | 336,58 MiB |
| CLIP-EBC RN50 | 23 | 201,60 s | 1,412 GiB | 705,99 MiB |
| MPCount | 23 | 109,36 s | 1,235 GiB | 362,22 MiB |

Los contadores suman 500,24 s, **8 min 20 s**, incluyendo validaciones, carga y
calentamiento además de inferencia. Las unidades completas suman 660,31 s,
**11 min** con importación y lanzamiento. El intervalo desde la primera ejecución
hasta la última fue 2425,26 s, **40 min 25 s**: incluye las pausas, diagnóstico y revisión
del controlador. Tras corregir la guarda, los 47 pendientes tardaron 437,18 s,
**7 min 17 s de pared**, sin nuevos rechazos de la guarda. Esos tiempos se midieron
en esta máquina; no son garantías para otras entradas o cargas compartidas.

Se comprobaron 69 IDs únicos ejecutados con éxito, sin repetir los completos;
23 procesos por modelo. Los 132 archivos NPY verificaron hash, float32, forma
esperada, finitud, suma y masa negativa. Diferencia máxima entre suma recalculada
y conteo del informe: **0**. Se comprobaron los límites efectivos High 1,5 GiB,
Max 2 GiB y swap 512 MiB en los 69 informes. RSS y el pico registrado por PyTorch no
representan toda la contabilidad de caché del cgroup o del driver.

La GPU quedó sin procesos CUDA de esta tanda. No hubo inferencias fallidas; los
rechazos del controlador ocurrieron antes de iniciar el siguiente trabajo. No se
entrenó ni se amplió el presupuesto por proceso. El alta compartida sigue
pendiente: estos resultados proceden de la ruta directa autorizada.

Revisión independiente del alcance, lanzadores y correcciones: Codex, modelo
efectivo no identificado. Los lanzadores aprobados y sus hashes se conservaron
en `reviewed-launchers/` junto a los recibos privados. El último lanzador tiene
SHA-256 `b4c216fffae86568d483d458b6708dd64a5fac8cb8cc86f29b730c8e3fdbfb31`.
La revisión final recalculó independientemente los 132 mapas, las nueve filas de
estadísticas, los tiempos y los 69 IDs únicos antes de publicar este informe.
