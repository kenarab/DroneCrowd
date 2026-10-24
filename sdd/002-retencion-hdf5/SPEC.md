# Validación, regeneración y retirada de mapas HDF5 históricos

Fecha: 24/10/2026. Estado: en ejecución.

## Encargo y autorización

El autor pide comprobar qué contienen los HDF5 que ocupan casi la mitad del
disco y borrarlos si resultan prescindibles. Recuerda que fueron generados
localmente y pide probar su regeneración. Ofrece un SSD externo de 1 TB si
hiciera falta espacio temporal. Autoriza expresamente un único agente revisor.
Esta autorización es condicional: no permite borrar archivos sin evidencia.

El encargo anterior de commits locales atómicos sigue vigente. Separar esta
especificación, la herramienta validada y el resultado de la retirada. No hacer
push ni publicar datos. La revisión independiente es de solo lectura.

## Estado inicial y alcance

- Dataset: `/home/alephbet/data/DroneCrowd`; `dataset` es un enlace a él.
- Inventario inicial: 50.245 HDF5, unos 404 GiB asignados en disco.
- Densidad: 16.800 archivos, unos 32,5 GiB.
- Localización/seguimiento: 33.445 archivos `_max.h5`, unos 371,5 GiB.
- Conservar imágenes, XML, MAT, listas, código, bash y resultados propios.
- Conservar ambos directorios de material personal de la marcha. Los videos
  actuales están en `~/home.agents/data/marchaUniversitaria241002`.
- No entrenar ni inferir sobre videos como parte de esta limpieza.

## Evidencia necesaria antes de retirar un archivo

1. Abrirlo en modo lectura e inventariar todos sus objetos. Rechazar enlaces,
   atributos, grupos, tipos, dimensiones o almacenamiento no contemplados.
2. Leer todos sus valores; rechazar errores y valores no finitos. Registrar
   SHA-256 del contenido de cada array, tipo, forma y almacenamiento.
3. Reconstruir todos los arrays desde `annotations/NNNNN.mat`, conservando
   orden, centros de cajas, identificadores y correspondencia de fotogramas.
   Si existe `GT_imgNNNNNN.mat`, comprobar también su igualdad con esa fuente.
4. Comparar todos los valores con igualdad numérica exacta, sin tolerancias. Esto
   no exige identidad binaria del contenedor ni distingue el signo de cero. Los archivos
   que difieran se conservan, incluso si su nombre o extensión parecen conocidos.
5. Conservar los MAT de secuencia y los XML originales. La mera existencia del
   XML no equivale a haber comprobado la conversión XML → MAT.
6. Guardar manifiesto permanente antes de borrar: ruta relativa, dispositivo,
   inode, tamaño, fecha, hashes y resultado. Verificar que las fuentes no hayan
   cambiado y revalidar cada archivo justo antes de retirarlo.
7. Limitar la retirada a archivos regulares enumerados por el manifiesto,
   `payload_verified == true`, sin diferencias ni errores, bajo
   `train_data/ground_truth` o `test_data/ground_truth`. No usar comodines de
   borrado ni seguir enlaces.

Los mapas se consumen durante el entrenamiento histórico. La prueba de video
adaptada utiliza imágenes y un checkpoint entrenado; no necesita estos mapas.
Su retirada implica tener que regenerarlos antes de volver a entrenar.

## Regeneración de prueba

Además de la reconstrucción exhaustiva en memoria, guardar una muestra nueva
de HDF5 en el temporal autorizado, abrirla de nuevo y comparar arrays y hashes
con los originales. Incluir densidad, identidad y los tres niveles de offsets,
varias secuencias y un fotograma impar que ya no tenga imagen/MAT individual.
No ejecutar los generadores históricos completos: crean procesos y escriben
sobre los destinos originales. Conservar sus fuentes y hashes como procedencia.

Los scripts bash encontrados en el dataset eliminan fotogramas impares; no son
generadores y no se ejecutan. La generación está en `make_data_density.py` y
`make_data_localization.py`.

## Recursos y aislamiento

Todos los temporales, cachés y logs transitorios van a `~/home.codex/tmp`, alias
del temporal autorizado `~/home.agents/tmp`. No modificar HOME ni CODEX_HOME.

La auditoría secuencial carga un archivo pequeño cada vez, sin GPU, instalación
de paquetes ni entorno compartido modificado. Estimación de RAM menor a 256 MiB,
con `RLIMIT_AS=1,5 GiB`; máximo 16 MiB por dataset numérico. Limitar cada archivo
de reporte a 128 MiB, una CPU efectiva por variables de threads y duración
inicial máxima de 3.600 s. Usar prioridad baja de CPU/IO. La lectura acumulada
recorre unos 404 GiB, pero no crea una copia de ese volumen.

La reserva vigente de 20 GiB para cargas grandes sigue intacta: no lanzar
entrenamiento, inferencia, descargas o una regeneración masiva con los 9 GiB
libres observados. Para una copia grande, exigir preflight, cuotas y destino
externo identificado con espacio suficiente. El ofrecimiento del SSD no
autoriza montar, formatear ni usar un volumen de identidad dudosa.

## Aceptación y registro

- Suite del auditor: referencias densas independientes para las matemáticas;
  lectura sin modificación, datos no finitos y enlaces rechazados.
- Auditoría exhaustiva terminada o subconjunto verificable identificado con
  claridad. Nunca afirmar que una muestra demuestra la regenerabilidad total.
- Regeneración a disco de la muestra, seguida de lectura e igualdad exacta.
- Revisión independiente del auditor y del alcance concreto del borrado.
- Evidencia local excluida de Git; resultado resumido en esta especificación.
- Inventario de archivos conservados antes/después y espacio efectivamente
  liberado. Registrar diferencias, archivos retenidos y cualquier bloqueo.
- Commits locales de propósito único y checkout limpio al cierre.

## Evidencia de implementación

- `scripts/audit_hdf5.py`: lector de solo lectura mediante la biblioteca HDF5
  del sistema y comparación exhaustiva con reconstrucción desde MAT.
- `scripts/regenerate_hdf5.py`: piloto de escritura exclusiva, destino vacío
  separado del dataset y dentro del temporal autorizado; máximo 12 archivos.
- Nueve pruebas del auditor/regenerador pasaron en CPU, sin instalaciones.
- Piloto: 11 archivos nuevos de cinco secuencias (001, 011, 100, 107, 110),
  81.942.016 bytes en total; todos los SHA-256 de arrays, tipos y formas
  coinciden tras escribirlos y abrirlos de nuevo. Incluye un fotograma impar
  sin imagen y secuencias con diferencias XML/MAT.
- Revisión independiente: Codex `/root/revision_retencion_hdf5`, modelo exacto
  no expuesto; inspección de código, cabeceras C locales y metadata exhaustiva.
  Solicitó proteger destinos del regenerador contra enlaces; corrección revisada.
- El complemento de metadata encontró un HDF5 con cabecera inválida:
  `train_data/ground_truth/img091271_max.h5`. Se conserva. Los otros 50.244
  carecen de comentarios de objetos y userblock. Este complemento no sustituye
  la lectura/comparación de todos los valores, todavía en ejecución.
- La revisión de XML/MAT encontró seis secuencias con una diferencia histórica
  de coordenadas: 004, 006, 007, 100, 107 y 110. No se determinó su origen.
  Conservar los MAT es necesario; regenerar desde XML podría producir mapas
  distintos de los que esta auditoría logra reproducir desde los MAT vigentes.
- `scripts/retire_verified_hdf5.py`: plan de solo lectura por defecto; ejecutar
  exige cantidad y SHA-256 del manifiesto revisado. Cruza ambas auditorías,
  comprueba fuentes y copias, y opera con `dir_fd`/`O_NOFOLLOW` sobre cada nombre
  validado, con diario de operaciones y detención ante cambios. Comprueba de
  nuevo los hashes y los 34.585 archivos ajenos al borrado al terminar.
- Diecinueve pruebas de auditor/regenerador/retirada pasaron en CPU. La retirada
  se probó exclusivamente con fixtures desechables. El mismo revisor aprobó
  su implementación; la ejecución real depende todavía del resultado completo
  y del manifiesto concreto. No se autoriza retirar el archivo con cabecera
  inválida por la mera existencia de un generador.
- El regenerador también admite `--reference-evidence`: verifica fuentes y
  reconstruye mapas que ya no existen, comparando contra los hashes conservados
  en el manifiesto. Un test CLI real elimina un HDF5 de prueba y lo regenera sin
  el original. Veinte pruebas pasaron; cambio revisado por el mismo revisor.

## Repetir una regeneración después de la retirada

Con el entorno temporal configurado, crear un directorio nuevo y vacío y usar
los registros permanentes. Este ejemplo reconstruye un único mapa de densidad:

```sh
destino=$(mktemp -d "$TMPDIR/dronecrowd-regenerado-XXXXXX")
python3 scripts/regenerate_hdf5.py \
  --root /home/alephbet/data/DroneCrowd \
  --output-dir "$destino" \
  --reference-evidence "$PWD/.private/hdf5-retencion-2026-10-24" \
  --source train_data/ground_truth/img001002.h5
```

Admite hasta 12 mapas por prueba, incluyendo `_max.h5`, sin escribir sobre los
datos originales. Una regeneración masiva requiere otro preflight y presupuesto
de almacenamiento: producir todo el conjunto denso ocuparía otra vez unos
404 GiB. La muestra pequeña no requiere el SSD ofrecido.
