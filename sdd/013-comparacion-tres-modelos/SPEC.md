# Tres contadores sobre un tramo breve de video

Encargo del autor, 25/10/2026: descargar SR, CLIP-EBC y MPCount y probarlos con
datos locales. Aclaración posterior: priorizar unos segundos del mismo video.

## Selección previa y pregunta

Tres cuadros 4K del video 0265, en tiempos solicitados 18,57055, 19,57055 y
20,57055 segundos: un tramo de dos segundos, muestreo aproximado de 1 Hz. Registrar
PTS decodificados, geometría y hashes antes de inferir. El último ya fue preparado
en SPEC012; reutilizarlo sin cambiar píxeles si coincide su PTS. Las salidas son
conteos por imagen; no se sumarán como asistentes ni individuos distintos.

Modelos congelados, variantes elegidas antes de ver respuestas nuevas:

- SR: DroneCrowd v1, código y hash fijados en SPEC012, ya descargado.
- CLIP-EBC: NWPU CLIP ViT-B/16 Word, paquete `NWPU_CLIP_ViT_B_16_Word_mae.tgz`
  de release v1.0.0, código `b52a5944bd75beed34fd3d0da516aa833ae63354`.
- MPCount: `sta_deterministic.pth`, enlace oficial Drive
  `1EBQzAcwZOhsrd00XDDnUTQ-wYH36y6oV`; fijar commit y hash al descargar.

Incidencia previa a predicciones: el paquete CLIP rotulado ViT-B/16 actualizado
contiene un estado de 24 bloques visuales de ancho 1024, kernel 14 y texto 768,
incompatible con B/16. Se conserva el hallazgo y se sustituye la variante de esta
prueba por NWPU CLIP ResNet50 Word, `best_mae.pth`, del paquete oficial
`NWPU_CLIP_ResNet50_Word.tgz`. No se elige por resultados ni se carga parcialmente.
Hash local del checkpoint RN50:
`da40b664a0a5a57bc33235e7334403e5000271e0741134e301ea706d754bf7df`.
MPCount fijado a commit `6eb06772bcf7dfb771c43a14d67146fce767f103`, checkpoint
`4a0ba849ace0ef2bc87364ec8551de4a3e9dca74fad27764c9b948ebf7928eb0`.
Para MPCount usar patch_size512 en vez del3584 predeterminado: la inferencia
oficial permite reducirlo, conservando unión de parches disjuntos. Declarar la
dependencia del contexto; no equiparar su resultado con una pasada sin cortes.

No entrenar, elegir pesos por cercanía a referencias parciales ni ampliar imágenes.
Es una prueba de ejecución y comparación cualitativa, no una medición validada de
error. No existen etiquetas exhaustivas para los tres cuadros. Puede añadirse
comparación de los ocho controles anteriores, separada del tramo prioritario.

## Auditoría y adaptación

Conservar URL, fecha, tamaño y SHA-256 de archivos descargados. Si falta digest
publicado, declararlo: un hash local identifica el artefacto pero no autentica su
publicación. Inspeccionar archivos comprimidos antes de extraer miembros seguros;
carga restringida `weights_only=True`, estados completos/estrictos y finitud.
Nada de pickle arbitrario ni inicializaciones adicionales por red.

SR mantiene su integración y mosaicos de SPEC012. CLIP y MPCount usarán unidades,
normalización y postprocesado oficiales verificados, sin copiar el divisor de
STNNet. Documentar geometría efectiva, resize/padding exigidos por cada método y
contextos diferentes. Streaming lote uno; acumular medias de solapes, no sumarlas
como nuevas personas. El evaluador CLIP original concatena ventanas: reemplazar
sólo esa gestión por ejecución secuencial equivalente y verificarla en pequeño.
El constructor CLIP debe evitar su descarga de nueve backbones y recomputar sus
features de texto con los pesos ya cargados; no usar texto aleatorio cacheado.

## Recursos y aceptación

Preflight: reservar 20 GiB disco y 2 GiB RAM disponible; estimar inicialmente
6 GiB adicionales de disco para fuentes, paquete/peso CLIP, MPCount y muestras.
Temporales/cachés/staging en `~/home.codex/tmp`; entornos propios sólo si faltan
dependencias, nunca modificar venv compartido. Fuentes de terceros con licencia.

GPU RTX 3060 compartida, lote uno y ejecuciones secuenciales. Scope de 2 GiB RAM,
512 MiB swap y 120 s por ejecución; no relajar límites ante fallo. Para checkpoints
grandes preferir mmap/asignación sin copias, conservando estado estricto. Si no
cabe una variante, registrar el fallo antes de decidir una configuración distinta.
Preparar la muestra nativa en lotes que respeten el presupuesto del muestreador;
no cambiar sus límites. Tres PPM ocupan aproximadamente 75 MB, fuera de Git.

Pruebas significativas de agregación/unidades, verificación sintética y revisión
independiente antes de GPU. Informar conteos, tiempos, memoria y fallos. Comparación
visual privada de los mismos cuadros con nombres y límites claros; no publicar
videos/fotos/pesos. Documentar resultados y cerrar cambios con commits atómicos
publicados en el fork autorizado. No ejecutar videos completos en esta etapa.

## Estado

Plan previo a descarga de pesos nuevos e inferencia del tramo. Coordinador Codex;
auditoría CLIP por Astra y adaptación MPCount por Luna; revisión final independiente.

Completado posteriormente: tres cuadros nativos, nueve mapas GPU y comparación
privada. Todos los modelos dentro de cuotas y con estados completos verificados;
cuatro pruebas de agregación y oráculos CPU aprobados. La alternativa RN50 se
eligió por incompatibilidad del paquete, no por resultados. No hay GT exhaustivo
ni ganador validado. [Resultados](../../docs/comparacion-tres-modelos-video-2026-10-25.md).
