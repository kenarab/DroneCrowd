# Primera campaña completa: video 0269

25/10/2026. El autor autorizó iniciar una campaña y delegó elegir el video.
Se elige 0269: 9,392717 s, 563 cuadros nativos 3840×2160, SHA-256
9a20838aed3912205eb2e19f8f70d53346328731df214c194bd148b8a8432199.
Aplicar SR, CLIP-EBC RN50 y MPCount con las mismas recetas y pesos previos.
Resultado previsto: 1689 predicciones, 36 bloques por modelo, 108 solicitudes.

Deriva de [SPEC017](../017-videos-completos/SPEC.md). Presupuesto base medido
extrapolado: 1,455 h; reservar 2–3 h, más cola. El primer bloque es parte de los
108 trabajos; no consumir intentos extra con reintentos automáticos.

## Implementación y puertas

- `full_video_campaign.py inventory` decodifica en CPU todos los cuadros sin
  guardar todos los RGB; congela ordinal, PTS entero, time_base y hash PPM nativo.
  Confirma el total mediante FFprobe y FFmpeg y verifica el original por hash.
- `freeze` fija manifiestos, recetas/código/lock/pesos y 108 IDs de contenido.
  Sólo se retiene un bloque de hasta 16 RGB en `~/home.codex/tmp`.
- El contador admite bloques de 16 únicamente con la opción explícita y campos
  temporales completos. Un modelo por proceso, lote GPU 1; libera RGB/mapa entre
  cuadros. No cambia la agregación, normalización, resolución ni pesos.
- `run_full_video_job.py` exige proyecto/perfil/request de Nexo, plan SHA fijado,
  código/lock/inputs/pesos válidos y scope real del dispatcher. Reutiliza el
  lanzador acotado, fija CPUQuota100%, RAM High1,5/Max2 GiB, swap512 MiB;
  contador115 s dentro del job120 s, más terminación hasta10 s. Logs16 MiB y
  archivos64 MiB máximo. Admisión RAM1,5 GiB/VRAM1024 MiB, scratch0,5/salidas1 GiB.
- Tres perfiles fijos, max_runs36 cada uno, fijados al commit limpio revisado.
  Infraestructura debe registrar la propuesta; no editar hermanos sin excepción
  explícita. Un estado waiting_profile no significa ejecución GPU.
- El coordinador es cliente de submit/status/cancel, sin reservas ni ejecutor
  alternativo. Prepara en CPU un bloque bajo cuotas, termina ese proceso y admite
  sus tres solicitudes. Máximo tres pendientes. Reanuda sólo salidas completas
  verificadas; parcial o terminal fallido requiere revisión, no un ID nuevo.
- Al fallar/cancelar/interrumpirse, solicita cancelar sólo los IDs propios enviados
  o reconciliados del mismo plan, con plazo global de150 s que incluye llamadas
  de estado/cancelación y espera, para constatar que están en
  estado terminal, incluyendo comprobar que no quedan queued/waiting_profile.
  Conserva
  salidas y caché de diagnóstico. No interpretar cancelación solicitada como GPU
  liberada; el estado deja constancia de la verificación o su ausencia.
- Verificar cache existente antes de cualquier solicitud: directorio confinado
  al temporal, sin enlaces, conjunto exacto de PPM regulares, tamaños y hashes.
  Rechazar un bloque parcial conservándolo; no consumir GPU para descubrirlo.
  El cargador coteja cada bloque con el inventario completo y las unidades de PTS.
- Piloto: primer bloque, tres modelos, dentro del presupuesto. Verificar los mapas
  por hash, forma, float32, finitud, suma y masa negativa en CPU. Comparar al menos
  un anclaje RGB exacto por modelo con el experimento previo, tolerancia
  max(0,01; abs(referencia)×1e-5). Exigir tiempo del contador<110 s y RSS<1,5 GiB.
  Verificar dos admisiones sucesivas, límites efectivos y cancelación/reanudación
  mediante pruebas CPU antes de continuar automáticamente con los otros bloques.
- La infraestructura ahora descuenta caché de archivo inactiva al admitir;
  comprobarlo con admisiones sucesivas. No añadir drop_caches ni reclamaciones
  propias, ni ampliar límites para sortear una espera.

## Recursos y evidencia

Conservar originales, anotaciones y resultados previos. Preflight128 GiB libres,
RAM host3 GiB para CPU (2 GiB más su margen), y admisión Nexo para cada scope.
Mapas estimados de este video: unos18 GiB; temporales RGB por bloque<380 MiB.
Eliminar sólo los PPM propios del bloque después de verificar sus tres modelos.
No borrar cachés parciales automáticamente.

Datos, entradas, informes, mapas, estado y recibos quedan fuera de Git en
`outputs/video-completo-0269-20261025/`; temporales en `~/home.codex/tmp`.
Guardar CSV de563 filas con ordinal/PTS/hashRGB y tres conteos. Preparar después
video local con los tres conteos y curvas por escena; no servir ni subir imágenes.
No sumar cuadros como asistentes ni seleccionar un ganador de precisión sin GT.

## Revisión y estado previo al piloto

Revisión independiente Codex, agente existente `revision_retencion_hdf5`, modelo
efectivo no identificado. Aprobó código/commit/congelado/perfiles propuestos y el
piloto tras registro válido. Se incorporaron sus bloqueantes: cachés parciales y
enlaces, cancelación fallida o pendiente, adopción de solicitudes tras reinicio y
coherencia PTS/inventario. Pasaron29 pruebas CPU:10 de campaña,11 del wrapper y8
del contador; el revisor reprodujo independientemente las10 de campaña.

Inventario real:563 ordinales,563 RGB únicos, PTS0..562562 en pasos1001 con
time_base1/60000; los10 anclajes RGB anteriores coinciden exactamente. Bloque0
preparado y verificado en CPU bajo cuotas, sin GPU.

Coordinador persistente único en la slice del proyecto: High64/Max128 MiB,
swap0, CPU50%, Restart=no, RuntimeMaxSec8 h y TimeoutStopSec180. Logs temporales
en `~/home.codex/tmp`, estado científico en outputs. Su RSS consume parte del
margen de admisión (~51 MiB) del perfil; el piloto se ejecutará con ese coordinador
residente y el decodificador terminado. Verificar dos admisiones sucesivas antes
de ampliar. Si recursos bloquean, registrar memoria/caché y detener sin ampliar.

Ejecutar primero con `--pilot-only`, comprobar y someter sus resultados a revisión
independiente. Reanudar luego el mismo plan omitiendo los tres resultados completos;
no consume tres intentos adicionales. Estado previo: listo para congelar y proponer
perfiles; aún no acredita GPU ni configuración compartida aplicada.
