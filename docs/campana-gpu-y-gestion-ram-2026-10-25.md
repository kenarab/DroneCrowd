# Cómo se procesa el video y por qué la GPU queda esperando RAM

25/10/2026, explicación solicitada por el autor. Este documento conserva la
explicación de la sesión; los contadores siguientes son una muestra fechada,
no un panel de estado en vivo.

## Trabajo científico

La campaña aplica SR, CLIP-EBC RN50 y MPCount a todos los cuadros del video0269:
9,392717 segundos,563 fotogramas nativos3840×2160 y1689 predicciones. No entrena
ni modifica pesos. Cada modelo produce un mapa de densidad; su integral es el
conteo estimado del encuadre. No se suman cuadros como asistentes distintos.

El recorrido se divide en36 bloques de hasta16 cuadros y108 trabajos GPU:

1. La CPU decodifica un bloque y comprueba ordinal, PTS real y hash de sus píxeles.
2. El dispatcher admite un modelo. Se cargan sus pesos una vez y se procesan los
   cuadros uno por vez en GPU. Después hacen lo mismo los otros dos modelos.
3. Se verifican informes y mapas: hashes, dimensiones, float32, finitud, integral
   y masa negativa. En el piloto, además se comparan anclajes RGB idénticos.
4. Se conservan los resultados y se eliminan sólo las imágenes temporales propias
   del bloque, regenerables desde el original. Los originales y anotaciones siguen
   intactos. A continuación se prepara el siguiente bloque.

Los16 RGB ocupan menos de380MiB en disco, pero el contador conserva en RAM sólo
una imagen y su mapa. Los mapas de todo el video ocuparían aproximadamente18GiB.
Datos y videos permanecen locales y fuera de Git.

## El límite que frena la admisión

Los12GiB de VRAM de la tarjeta y la RAM del sistema son recursos distintos.
La VRAM aloja pesos y tensores de GPU; la RAM también sostiene Python, carga de
pesos, decodificación, buffers, coordinadores y caché del sistema de archivos.
No existe una proporción universal que obligue a reservar12GiB de RAM para esta
tarjeta. Para esta carga, en cambio, el margen del contrato actual es insuficiente.

| Control | Valor vigente |
| --- | ---: |
| High del proyecto, usado para admisión | 2GiB |
| Max del proyecto | 4GiB |
| Estimación RAM por trabajo GPU | 1,5GiB |
| Reserva de admisión con30% de margen | 1,95GiB |
| Margen restante para control y memoria ya imputada | 51,2MiB |
| High/Max del trabajo durante inferencia | 1,5/2GiB |

High es un umbral de control y admisión; Max es el límite duro. La cuota del
proyecto no se resume en «máximo2GiB». El problema concreto es que el planificador
exige encajar la nueva reserva bajo High, junto con la memoria ya contabilizada.

Linux conserva páginas de archivos después de que termina un proceso. Nexo ya
descuenta `inactive_file`, pero permanece caché clasificada como `active_file`.
En una muestra había unos46MiB activos más unos15MiB anónimos del coordinador y
otra memoria del kernel: unos65MiB para admisión, por encima de los51,2MiB libres
en el contrato. La GPU podía estar vacía y la máquina tener RAM disponible, pero
el siguiente trabajo permanecía `queued / waiting_resources_or_external_process`.
La clase activa no demuestra por sí sola que un modelo siga vivo ni identifica
qué archivo posee cada página.

## Qué se probó y qué demuestra

El piloto completó16 cuadros por modelo:48 mapas verificados y un anclaje idéntico
por modelo con diferencia de conteo0. Tiempos del contador: SR69,18s;
CLIP-EBC49,75s; MPCount39,27s. RSS pico observado:1001,20/1431,91/1244,15MiB,
respectivamente. Esto verifica ejecución y consistencia; no precisión sin GT.

Primero se aconsejó al kernel liberar caché de los16 PPM temporales verificados,
con descriptores de sólo lectura, `fsync` y `POSIX_FADV_DONTNEED`. Bajó la caché
inactiva, pero la activa se mantuvo en46,39MB: no resolvió el gate de admisión.
Por ello no se automatizó esa intervención.

Después se revisó y realizó una única solicitud de recuperación de32MiB sobre
la slice exacta de DroneCrowd, con `swappiness=0`. Se verificaron código/plan,
PID y comando del único coordinador, ausencia de modelo/decodificador,
memoria anónima≤32MiB, shmem0 y cuotas intactas. El resultado observado fue:

| Medida | Antes | Después |
| --- | ---: | ---: |
| Memoria usada para admisión, bytes | 66.338.816 | 20.127.744 |
| Memoria anónima, bytes | 15.171.584 | 15.171.584 |
| Caché activa de archivos, bytes | 46.383.104 | 172.032 |

El kernel recuperó una cantidad diferente de la solicitada, comportamiento previsto
por la interfaz. Nexo admitió otro trabajo. No se borraron archivos, no se usó
`drop_caches` global y no se ampliaron High, Max, swap o perfiles.
La comprobación previa no constituye una reserva transaccional: un trabajo
encolado puede ser admitido mientras el kernel realiza la recuperación normal.

Fuentes primarias: [cgroup v2: memoria](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory)
y [posix_fadvise](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html).
La recuperación acotada es una contingencia operativa, no la solución estructural.

## Estado, tiempo y siguiente paso

Muestra inicial de las22:00−03:7/108 trabajos completos,112/1689 predicciones;
dos solicitudes del tercer bloque esperando RAM y GPU ociosa. La proyección de
modelos restantes ronda1½h; las esperas impiden fijar una hora final confiable.
El presupuesto operativo2–3h sólo aplica con admisión estable y sin colas externas.
No presentar la GPU libre como ejecución en curso.

Se preparó un auxiliar operativo para revisión, aún sin activación en esta muestra:
una recuperación de32MiB por firma de solicitudes, presupuesto durable de108
acciones/8h, identidades comprobadas, sin worker/decodificador y con parada si no
resuelve el gate. No ejecuta GPU ni cambia el plan científico.

Actualización posterior: el auxiliar se revisó y activó con High32/Max64MiB,
swap0 yCPU10%, conservando presupuestos persistentes y guardas. La campaña
alcanzó12 trabajos completos y192 predicciones/mapas verificados:64 de563 cuadros
con los tres modelos. Después una recuperación no resolvió el gate en45s;
el auxiliar detuvo al coordinador y éste canceló tres solicitudes sin iniciar.
Los12 resultados completos se conservan; no hubo informes de inferencia fallidos.
No se relajan guardas ni se reenvían automáticamente los IDs cancelados.

En esa espera la caché activa llegó a210.296.832 bytes. La acción liberó caché
inactiva, manteniendo el uso para admisión por encima de240MB. El caso de46MiB
era una muestra inicial; no describe un máximo del flujo ni identifica sus archivos.
Esto refuerza la necesidad de presupuestar las fases completas.

Quedan1497 predicciones, aproximadamente1h20min de procesos del contador por
extrapolación, incluyendo carga, inferencia y escritura; no es tiempo puro de GPU.
No hay hora de finalización operativa confiable hasta que Nexo
resuelva presupuesto/admisión y se revise la transición de las solicitudes
pendientes. El presupuesto2–3h requiere un flujo estable; no es un compromiso
de finalización mientras la campaña está detenida.

La sesión responsable de Nexo confirmó que vio el encargo de RAM en su backlog;
la propuesta y su adopción siguen pendientes. Reanudar también requiere resolver
las tres solicitudes canceladas sin iniciar: la API actual devuelve su estado
cancelado al reenviar el mismo ID y no tiene una operación de requeue. No basta
subir High y reiniciar el cliente; hace falta una transición revisada que conserve
los12 resultados completos y no repita inferencias ya ejecutadas.

El autor encargó a Nexo una SPEC para revisar presupuestos por fase, una cuota de
proyecto adecuada a GPU y motivos de espera observables. Candidato a evaluar:
High3GiB/Max4GiB para este consumidor, conservando topes del pool/cuenta y las
cuotas de cada trabajo; no es una configuración aplicada ni una regla universal.

## Pedido posterior: aprovechar VRAM y evaluar paralelismo

El autor pidió que Nexo compare uno, dos y tres procesos al observar600–800MiB
de VRAM ocupada. Ese porcentaje mide capacidad de memoria; la actividad GPU se
mide aparte. Una muestra de SR indicó95% de tiempo con kernels activos y804MiB.
Eso tampoco equivale a95% de FLOPS máximos ni prueba que otra instancia acelere
el conjunto. Se deben comparar throughput y tiempos reales.

La SPEC de Nexo incorpora un benchmark serie y por pares SR+CLIP, SR+MP,
CLIP+MP, y tres procesos sólo si admiten todos los recursos. Dos reservas actuales
suman3,9GiB antes de control/caché; tres suman5,85GiB y superan High6GiB del pool
al agregar esos componentes. La propuesta serie High3/Max4GiB requiere otra
evaluación para paralelismo. Conservar6GiB de reserva VRAM para atención y medir
consumo real de procesos/contextos, además de los tensores informados por Torch.

Como alternativa posterior del producto se pueden comparar lotes de parches o
cuadros dentro de un modelo persistente, con regresión numérica. Esta campaña
conserva lote1 y perfiles revisados; no se activa MPS ni se cambia el código
científico por cargar la propuesta.

Fuentes: [métricas NVIDIA](https://docs.nvidia.com/deploy/nvidia-smi/index.html) y
[CUDA: medición y concurrencia](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

Evidencia privada: `outputs/video-completo-0269-20261025/`, con plan inmutable,
recibos de la cola, informes, mapas y pruebas de admisión. Código operativo fijado
a4a668ce; [SPEC018](../sdd/018-campana-video-0269/SPEC.md). Este documento se publica
en una rama documental aislada para conservar el HEAD exigido por la campaña.

Revisión independiente Codex, modelo efectivo no identificado: mediciones, parada, límites científicos y propuesta de paralelismo comprobados; sin bloqueantes documentales.
