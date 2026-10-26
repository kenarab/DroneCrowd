# Campaña nocturna de los tres clips

26/10/2026. El autor autoriza el uso exclusivo nocturno de GPU y pide ejecutar
todo el plan de SPEC017. Alcance: 0269, 0265 y trim, en ese orden; SR,
CLIP-EBC RN50 y MPCount, 2592 cuadros y 7776 predicciones. No entrenamiento ni
cambios de resolución, pesos o receta. Estimación previa: 6–8 horas de ejecución,
además de preparación de infraestructura; no constituye un resultado medido.

## Continuidad y ejecución

Conservar el plan de 0269 y sus archivos científicos congelados en 4a668ce.
Verificar y omitir sus 12 trabajos completos/192 mapas. Tres solicitudes
canceladas nunca iniciadas requieren reemplazo transaccional mediante API Nexo,
con vínculo supersedes y manifiesto de migración fijado por SHA en los perfiles.
No editar SQLite, reactivar terminales, sobrescribir parciales ni repetir inicios.
Los otros 93 trabajos de 0269 conservan sus IDs. Mantener 36 inicios/modelo,
incluidos los cuatro anteriores, aunque se revise el pin del entrypoint.

Agregar un cliente/entrypoint independiente sin modificar el código congelado.
Para 0265 y trim confirmar inventario completo por decodificación CPU, ordinal,
PTS entero, time_base y SHA PPM. En trim conservar el recorte previo de 12 filas
inferiores: 960×528. Comparar anclajes RGB y conteos de las pruebas anteriores.
Planes y seis perfiles nuevos fijan hashes, revisión y presupuestos86/42 por
modelo; el piloto pertenece a esos presupuestos. Emitir como máximo tres
solicitudes y preparar un solo bloque de16 RGB, batchGPU1, CPU100%, timeout120s.

La cola es el único lanzador GPU. Nexo implementa y activa la política revisada
de RAM de proyecto3/4/1GiB; se conserva workerHigh1,5/Max2/swap0,5GiB y reserva
1,95GiB. Sin reclamación periódica ni cambios a cuotas de padres. Concurrencia
inicial1; usar la GPU en exclusiva no demuestra ventaja de ejecutar en paralelo.
Una prueba de concurrencia exige presupuesto y medición independientes.

## Aceptación y entrega

- Pruebas CPU de identidad, reemplazo solo de cancelados sin inicio, integridad
  del inventario/PTS, último bloque corto, parciales y cancelación/reinicio.
- Revisión independiente y commit atómico antes de fijar perfiles. Medir dos
  ciclos completos de0269 tras el cambio; conservar evidencia de recursos.
- Verificar todos los mapas por hash, forma, float32, finitud e integral.
  Eliminar únicamente PPM regenerables propios tras verificar los tres modelos.
- CSV por video y combinado con2592filas/7776predicciones; curvas, videos
  locales anotados y resumen de tiempos, picos y límites. Datos y medios fuera
  de Git. No sumar cuadros o planos como asistentes distintos ni afirmar
  precisión sin anotaciones de referencia.
- Ventana exclusiva y servicio acotados a8h desde activación Nexo, sin reinicios
  ni reintentos automáticos; en fallo
  cancelar únicamente solicitudes propias y registrar liberación comprobada.

La autorización nocturna permite ejecutar la campaña, pero no desactivar guardas
ni apropiarse de procesos ajenos. La configuración compartida sigue a cargo de
Nexo. No se considera iniciada hasta observar un worker y un resultado nuevo.

## Comprobaciones previas y revisión

Inventarios CPU completos:0265=1370cuadros y23anclajes RGB previos exactos;
trim=659 y11anclajes exactos. El inventario monolítico de0265 excedió120s sin
lanzar GPU; se dividió en metadatos y segmentos128 con seek por PTS original,
copias de timestamps y verificación SHA. Se conserva el límite por proceso.
Preflight privado en `outputs/campana-nocturna-20261026/preflight.json`:
380,34GiB libres/19,63GiB RAM disponible, mínimo128GiB y80GiB para entregables.

Revisión independiente: agente existente Codex `revision_retencion_hdf5`, modelo
efectivo no identificado. Sin bloqueantes para commit, freeze y propuesta de
perfiles. Verificó14pruebas nuevas CPU y una prueba VFR con B-frames/GOP:
17/17cuadros de cinco segmentos bit-exactos frente a decodificación completa.
Pasan además10pruebas de campaña anterior y11del wrapper; prueba sintética de
etiquetas FFmpeg y6/6PTS racionales de codificación exactos. Se corrigieron
validación del ledger12/192, errores de preflight, gate de dos bloques nuevos,
CSV canónico y procedencia, y timestamps del video de revisión. La activación
requiere todavía paquete concreto, revisión/política/API Nexo y ventana efectiva.
