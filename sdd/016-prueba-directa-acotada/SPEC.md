# Prueba GPU breve mientras se integra el dispatcher

25/10/2026. El autor reiteró: «Mandalos a la GPU, cuanto van a tardar? En el
medio arreglamos el pipeline». El alta sigue sin aplicar. Ejecutar esta tanda
breve directamente con cuotas efectivas, conservando la distinción con envío al
dispatcher. No modificar registros compartidos ni afirmar que hubo encolado.

Usar exactamente los tres trabajos iniciales de SPEC015: dos cuadros de
`0265-batch-000` por modelo, plan congelado93405… y pesos/código de SPEC013/014.
Un intento por modelo, secuenciales. Sin entrenamiento, descargas ni cambios de
receta. Después el autor pidió una prueba más grande: continuar con los otros 66
trabajos de SPEC014 sólo si los tres iniciales terminan y se verifican completos.
Alcance total máximo 69 intentos/44 cuadros, sin repetir los tres iniciales.

Antes de cada ejecución: verificar hashes, entorno, cuotas del proyecto/pool,
20 GiB retenidos más disco previsto,2 GiB RAM host más margen estimado, GPU libre
de procesos CUDA externos y al menos7 GiB libres. Si cambian, detener la tanda.

Cada ejecución usa una unidad propia `dronecrowd-inference-*.scope` dentro de la
slice existente de DroneCrowd, con High 1,5 GiB, Max 2 GiB, swap 512 MiB, un núcleo y
un hilo. Timeout120 s y hasta 10 s de terminación; límite interno del contador115 s.
Logs acotados16 MiB por stream y escrituras del hijo64 MiB por archivo. Temporales
y cachés exclusivamente en `~/home.codex/tmp`; sin modificar el venv compartido.

Conservar comandos, preflight, estado y logs locales, además de los informes y
mapas del contador. Verificar resultados contra todos los cuadros/PTS, hashes de
código, entrada y pesos. Detener ante fallo sin ampliar límites ni repetir.
Preservar salidas parciales. No usar un nombre de unidad del dispatcher ni crear
una reserva ficticia en su base. La comprobación fresca reduce colisiones, pero
no equivale a una reserva transaccional; la integración de la cola sigue pendiente.

Tras SR la admisión detuvo el siguiente proceso: quedaban unos240 MiB de caché
limpia imputados a la slice vacía. Antes de reanudar se permite un único pedido
acotado de `memory.reclaim` sobre esa slice exacta, hasta 2 GiB, sólo con
`populated=0`, `anon=0`, `shmem=0`. Registrar `file_dirty` y `file_writeback` antes
y después: no exigir cero tras comprobar que esa condición bloqueó por 8192 bytes
residuales sin procesos ni memoria activa. La reclamación normal del kernel
conserva o escribe las páginas pendientes, no descarta sus datos. Registrar
medidas y EAGAIN si ocurre; revalidar cuotas/admisión completas después. No
repetir SR ni ampliar límites. Referencia:
[documentación del kernel](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory).
Sincronizar con fsync sólo informes/densidades propios verificados al cerrar cada
proceso o reutilizar un resultado. Esperar hasta 30 s a que las guardas de slice
vacía/sin memoria anónima se cumplan; si no, detener sin GPU. El tope2 GiB corresponde al pedido
de recuperación, no garantiza una cantidad exacta devuelta por el kernel.

Revisión independiente del alcance y lanzador antes de GPU, también para ampliar
la tanda inicial. Cierre con tiempos
reales y resultados por cuadro, sin MAE ni suma de asistentes porque no hay GT
exhaustivo. Commit atómico de especificación, procedimiento y resultados después
de la tanda para conservar la revisión fijada durante su ejecución.

## Cierre

El autor amplió a 44 cuadros/69 trabajos. Completados los 69 IDs únicos,23 por
modelo, sin repetir inferencias completas ni ampliar cuotas. Conformes 132 mapas
por hash, forma, float32, finitud y suma; diferencia de conteo recalculado 0.
Límites efectivos1,5/2 GiB RAM y512 MiB swap verificados en todos los informes.

El control se detuvo antes de trabajos nuevos por admisión y después por exigir
dirty/writeback0. Telemetría mostró 8192 bytes pendientes, sin procesos ni anon/shmem;
se corrigió esa condición con revisión y contrato del kernel. Permanecieron ruta
exacta, slice vacía, pedido acotado y admisión posterior. Ningún fallo de inferencia.

Tiempos: contador500,24 s; unidades660,31 s; pared2425,26 s con pausas/diagnóstico.
Los 47 trabajos posteriores a la corrección final tardaron437,18 s de pared.
Revisor independiente Codex, modelo efectivo no identificado; aprobó cada cambio
del lanzador antes de alterar la admisión. Código y pesos del plan congelado
intactos. El último lanzador tiene SHA b4c216fffae86568d483d458b6708dd64a5fac8cb8cc86f29b730c8e3fdbfb31.

[Resultados, límites y reproducción](../../docs/resultado-validacion-extendida-2026-10-25.md).
Datos, mapas, CSV, gráficos y recibos permanecen fuera de Git. No hubo encolado;
la integración de alta/perfiles compartidos sigue pendiente. No se informa MAE ni
asistencia total por ausencia de referencia exhaustiva y cobertura disjunta.
