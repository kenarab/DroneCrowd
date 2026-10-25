# Validación extendida preparada para la cola GPU

25/10/2026. Las entradas están materializadas y verificadas: **44 cuadros,
23 lotes y 69 descriptores de trabajo** para SR, CLIP-EBC RN50 y MPCount. No se
enviaron al dispatcher ni se ejecutó GPU en esta fase. Después apareció la CLI
real del dispatcher; se prepara el alta del proyecto y sus perfiles en SPEC015.

La similitud observada antes correspondía a los tres cuadros de MPCount, no a
acuerdo entre modelos. La extensión examinará respuesta y variación en toda la
duración disponible. Sin referencia exhaustiva, estabilidad no equivale a precisión.

## Cobertura y entradas

| Video | Duración | Muestreo solicitado | Cuadros | Lotes | Trabajos GPU preparados |
| --- | ---: | --- | ---: | ---: | ---: |
| 0265 | 22,856 s | 0–22 s, cada segundo | 23 | 12 | 36 |
| 0269 | 9,393 s | 0–9 s, cada segundo | 10 | 5 | 15 |
| trim | 10,985 s | 0–10 s, cada segundo | 11 | 6 | 18 |
| Total | 43,234 s | 1 Hz | 44 | 23 | 69 |

Cada lote contiene uno o dos cuadros y genera tres trabajos, uno por modelo.
Se inspeccionarán instantes a lo largo de todos los clips; no se procesan los
2592 cuadros ni se observa continuamente lo ocurrido entre muestras.

Los dos videos grandes conservan 3840×2160. El trim conserva píxeles originales
en 960×528, excluyendo las últimas 12 filas para alinear los tres modelos; no
se amplió ni interpoló. Se registran PTS efectivos y hashes de originales y RGB.
Desvío máximo comprobado respecto del segundo solicitado: 0,016 s.

Los RGB ocupan 837.873.366 bytes, unos 838 MB. El muestreo acumuló 88,58 s en CPU,
dentro del scope de 2 GiB/120 s. Las cuotas de cada lote del muestreador y la
reserva de disco/RAM se mantuvieron. La estimación de 4 GiB adicionales incluye
previews, futuros mapas y resúmenes; no representa espacio ya consumido.

## Artefactos locales y estado real

Directorio privado: `outputs/validacion-extendida-cola-20261025/`.

- `plan.json`: muestras, revisión, hashes de código/pesos, argumentos tipados,
  recursos e identificadores idempotentes de los 69 trabajos.
- `batches/`: 23 manifiestos y sus 44 RGB/previews completos y verificados.
- `preparation-check.json`: comprobación final de datos, geometría, PTS, IDs y
  correspondencia con el código; sin directorios de resultados GPU.

El plan declara `preparation_state=prepared`,
`submission_state=prepared_not_submitted`, `not_submitted=true` y
`executor_profile_registered=false`. Tener este JSON **no acredita encolado**.
No existe recibo del dispatcher ni ejecución pendiente en un proceso nuestro.

Hashes SHA-256:

- Plan: `93405aba21aaef66f4a9bc85615d3de3e34ce267c382725b3e7455170adbd73b`.
- Los manifiestos/RGB y los checkpoints se identifican individualmente dentro
  del plan; los pesos son exactamente los de SPEC013, sin nueva selección.

El snapshot de ejecución corresponde al commit `5471334`; los hashes del código
se verificaron nuevamente contra los archivos actuales. Cada ID de trabajo liga
manifiesto de entrada, modelo, checkpoint y código. Reenviar el mismo descriptor
debe conservar su ID. Crear de nuevo muestras con metadata distinta puede producir
otro manifiesto/ID: reutilizar el plan congelado al integrar, sin reconstruirlo
para eludir deduplicación ni repetir trabajos completados.

## Contrato propuesto para integrar

El nombre inicial `dronecrowd-native-count-v1` era una propuesta. La CLI real
admite perfiles con argv fijo, sin argumentos dinámicos en la solicitud. SPEC015
prepara tres perfiles iniciales `dronecrowd-smoke-{sr,clip-ebc,mpcount}-v1`, con
un ID de trabajo fijo cada uno y revisión Git completa. El wrapper valida el plan
y llama al contador tipado. Los datos y rutas de origen permanecen del lado del
proyecto. No enviar videos, cuadros o el plan privado completo a servicios externos.

Cada descriptor contiene `model`, `input`, `expected_input_sha256`, `checkpoint`,
`checkpoint_sha256` y `output_dir`; CLIP añade su árbol fuente local verificado.
Las tres variantes y sus normalizaciones, unidades y contextos siguen siendo
las de la [comparación previa](comparacion-tres-modelos-video-2026-10-25.md).

Por trabajo: lote uno, un hilo CPU, RAM máxima 2 GiB, swap 512 MiB, tiempo máximo
120 s; estimación inicial de RAM 1,5 GiB y VRAM 1 GiB. El dispatcher deberá
comprobar recursos frescos, sus márgenes y exclusividad antes de admitirlo. Las
estimaciones no reservan físicamente la GPU ni autorizan ampliar cuotas ante fallo.
La prioridad propuesta es validación breve que desbloquea la evaluación local.

El plan conserva la referencia al entorno temporal original; el alta usa su
equivalente persistente `.venv/gpu-candidates`, con versiones verificadas contra
`ops/gpu-candidates.lock.json`. No cambia el plan ni sus IDs. El árbol CLIP
continúa en el temporal autorizado y es un prerrequisito verificado. Si desaparece
o cambia, esperar su reconstrucción/verificación en CPU; no descargar
inicializaciones durante el trabajo GPU. No se modificó el venv compartido.

El alta y los perfiles siguen pendientes de aplicación por infraestructura. La
primera tanda contiene sólo tres trabajos sobre los dos cuadros iniciales del
video0265: un intento por perfil. Los otros66 quedan preparados. Tras el alta,
conservar recibos reales y verificar admisión/ejecución antes de ampliar la tanda.
Los resultados fallidos deben preservarse; cualquier reintento tendrá revisión y
un intento explícito acotado. [Paquete de alta](alta-dispatcher-2026-10-25.md).

## Preparación, reanudación y comprobaciones

[SPEC014](../sdd/014-validacion-larga-en-cola/SPEC.md) se escribió antes de preparar.
`scripts/prepare_long_video_jobs.py` planifica por defecto; `--prepare` materializa
las muestras en CPU. El staging se limita a `~/home.codex/tmp`; sólo se promueve
un lote completo tras validar fuente, hash, PPM, geometría y PTS. Un rename entre
filesystems falla cerrado, sin sustituir la ubicación temporal por otra.

El plan inicial y el progreso se guardan atómicamente. Reanudar verifica de nuevo
cada RGB y el hash del manifiesto congelado; no sobreescribe entradas modificadas
ni salidas. Antes de preparar exige 20 GiB retenidos más los 4 GiB previstos con
30 % de margen y al menos 2 GiB de RAM host disponible.

Comando de preparación, dentro de un scope CPU con los límites anteriores y con
TMPDIR/TMP/TEMP exportados al temporal autorizado:

```sh
python3 scripts/prepare_long_video_jobs.py \
  --output-dir outputs/validacion-extendida-cola-20261025 --prepare
```

Ese comando no envía ni ejecuta GPU. El contador acepta un manifiesto nuevo sólo
con su SHA explícito, mantiene máximo tres cuadros por proceso y registra estado
inicial/progreso por cuadro; estos lotes usan un máximo de dos. Las fórmulas de
densidad no cambiaron ni se repitió inferencia para comprobar esta preparación.

Pasaron 9 pruebas del preparador y 7 del ejecutor en CPU: partición, recorte,
identidades, cambios de contenido, reanudación/interrupción, PTS, reservas,
EXDEV, geometría y conservación del último informe válido. Implementación del
preparador: GPT-6 Luna; ejecutor y coordinación: Codex. Revisión independiente
Codex, modelo efectivo no expuesto, aprobó la materialización antes de ejecutarla.
La comprobación final verificó 44 RGB y 69 IDs únicos/recalculados, sin salidas GPU.

Los conteos futuros describirán cada instante/encuadre. No sumar cuadros ni clips
como asistentes, ni suponer simultaneidad o cobertura de toda la marcha. Aún faltan
referencias exhaustivas, máscara del área de interés y revisión de carteles,
estructuras y sensibilidad a parches, especialmente para MPCount.
