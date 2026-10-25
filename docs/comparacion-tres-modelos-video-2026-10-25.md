# SR, CLIP-EBC y MPCount sobre el mismo tramo de video

Los tres modelos se descargaron y ejecutaron en GPU sobre los mismos tres cuadros
4K del video 0265. MPCount ofrece una respuesta espacial mucho más extensa en la
multitud que SR o CLIP-EBC; esto lo vuelve un candidato interesante para validar,
sin acreditar todavía su cantidad de personas. No se entrenó ni se procesaron
videos completos. La selección prioriza el pedido posterior del autor: unos
segundos del video, antes de extender la comparación a fotos u otros clips.

## Comparación disponible

[Abrir los tres cuadros y mapas en el visor privado](https://192.168.1.99:8870/doc/dronecrowd/revision/tres-modelos-video/index.html).

| PTS del mismo cuadro | SR, DroneCrowd | CLIP-EBC, NWPU ResNet50 | MPCount, STA determinista |
| --- | ---: | ---: | ---: |
| 18,585233 s | 410,45 | 370,76 | 10.426,48 |
| 19,586233 s | 384,77 | 239,01 | 10.533,72 |
| 20,570550 s | 302,87 | 152,15 | 10.365,90 |

Son integrales de densidad, no detecciones de personas verificadas. **No sumar
los tres cuadros.** Muestran esencialmente el mismo lugar en distintos instantes;
también cambia el encuadre. No se interpreta la variación como entrada o salida
de personas. Falta una referencia exhaustiva y una máscara validada de suelo que
excluya terrazas y delimite la cobertura para estimar ocupación a una hora.

Las vistas conservan una escala de color común entre los tres modelos y cuadros,
en masa positiva por píxel de entrada, saturada al percentil 99,5 global. Los
números conservan toda la integral. La imagen del visor está reducida para verla;
los modelos recibieron píxeles nativos, sin upscaling, de 3840×2160 completos.

La señal de MPCount sigue mejor la zona ocupada por multitud en estas imágenes,
pero también muestra discontinuidades rectangulares compatibles con los parches
y su máscara PAC, además de activaciones en carteles y estructuras. No se
demuestra invariancia al tamaño de parche ni que cada
activación corresponda a una persona. No se selecciona el mayor número como verdad.

Como control adicional sin nueva inferencia, se integraron cuatro regiones
previamente seleccionadas del último cuadro. No son las predicciones de recortes
con contexto de ensayos anteriores: provienen de los mapas completos actuales.

| Región anterior, coordenadas xyxy | SR | CLIP-EBC | MPCount |
| --- | ---: | ---: | ---: |
| Personas A, [1104,1552,1168,1648] | 0,11 | 0,37 | 22,93 |
| Personas B, [1072,2000,1200,2128] | 1,73 | 4,01 | 4,98 |
| Multitud densa, [2496,1792,2624,1920] | 0,18 | 0,0044 | 12,37 |
| Cúpula, [0,1024,256,1280] | 0,65 | 0,0021 | 0,00 |

La respuesta en el recorte denso y la ausencia de masa MPCount en esa cúpula
motivan revisar ese candidato, sin validar 10.000 personas ni ausencia general
de falsos positivos. La multitud densa aún no tiene referencia fiable.

## Variantes, fuentes y pesos

[SPEC013](../sdd/013-comparacion-tres-modelos/SPEC.md) se escribió antes de inferir.
El tramo estaba fijado en 18,57055, 19,57055 y 20,57055 segundos solicitados; la
tabla registra los PTS efectivamente decodificados. Dos cuadros se extrajeron con
el muestreador existente y el tercero se reutilizó sin modificar sus píxeles.
Manifiesto de entrada: `f9737bb74cce72913959475808957f059ea62e09ca61e6f31651c01ef0e47745`.

| Método | Código oficial fijado | Checkpoint usado |
| --- | --- | --- |
| [SR, CVPR 2026](https://github.com/wbshu/SR) | `6c5f66c823dd86dcb1329f1b007ba369e622036b` | `dronecrowd.tar`, 86.123.541 bytes |
| [CLIP-EBC, ICME 2025](https://github.com/Yiming-M/CLIP-EBC) | `b52a5944bd75beed34fd3d0da516aa833ae63354` | NWPU ResNet50 Word, `best_mae.pth`, 542.307.202 bytes |
| [MPCount, CVPR 2024](https://github.com/Shimmer93/MPCount) | `6eb06772bcf7dfb771c43a14d67146fce767f103` | STA determinista, 133.040.867 bytes |

SHA-256 de pesos:

- SR: `22ff7ff7bfcfa6cb45b38ea0df6dd74f388acc070b41bbb2d64a7308062097eb`.
- CLIP-EBC: `da40b664a0a5a57bc33235e7334403e5000271e0741134e301ea706d754bf7df`.
- MPCount: `4a0ba849ace0ef2bc87364ec8551de4a3e9dca74fad27764c9b948ebf7928eb0`.

SR coincide con el digest publicado. CLIP y MPCount no publicaron un digest en
los metadatos consultados: los hashes locales identifican las descargas de sus
enlaces oficiales, sin aportar esa autenticación independiente. Se conservan
URLs, tamaños, hashes y decisiones en `outputs/comparacion-tres-modelos/procedencia.json`.

### Incidencia del paquete CLIP actualizado

El paquete `NWPU_CLIP_ViT_B_16_Word_mae.tgz`, 1.136.121.799 bytes, contiene un único
archivo de 1.789.334.433 bytes con 483 tensores. No es compatible con B/16: tiene
24 bloques visuales de ancho 1024, kernel 14, embedding posicional [257,1024] y
texto de ancho 768, estructura compatible con la familia L/14. Esto no certifica
su configuración de entrenamiento. No se lo ejecutó ni se usó carga parcial.

La incidencia se detectó antes de predicciones y se eligió la alternativa oficial
RN50, no por sus resultados. Su paquete de 812.132.262 bytes contiene **dos pesos**,
`best_mae.pth` y `best_rmse.pth`, de aproximadamente 542 MB cada uno. Se fijó
`best_mae` sin evaluar ambos para seleccionar. Esto ilustra por qué el tamaño del
paquete no equivale al tamaño de un modelo. El paquete inconsistente se conserva
localmente para auditoría; no se publica su binario.

## Adaptación y unidades

- SR: misma arquitectura y receta de SPEC012, 21.510.465 parámetros; núcleo
  256×256, halo 160, salida stride8 y suma directa. 135 tiles por cuadro.
- CLIP-EBC RN50: 135.465.313 parámetros y 500 tensores. Entrada ImageNet, bins
  NWPU de truncamiento4 y anclajes [0,1,2,3,4.21931]. Ventanas224/stride224,
  180 por cuadro; promedio de celdas solapadas y suma directa, sin factor100.
- MPCount: 33.233.241 parámetros y 147 tensores. RGB normalizado con media0,5 y
  desviación0,5; salida de densidad en resolución de entrada tras máscara PAC
  oficial. Suma/1000; 40 parches disjuntos de hasta512×512 por cuadro. El CLI
  oficial permite reducir patch_size; se usó512 en vez del3584 predeterminado
  para mantener presupuesto. Cambia contexto y puede generar artefactos de borde.

Todos comparten el mismo encuadre y área efectiva, sin padding ni recorte de borde
en estos cuadros. Sus contextos y postprocesados son propios: es una comparación
de sistemas adaptados, no un control que aísle sólo arquitectura. No se copiarán
factores de escala entre modelos ni se calibrarán para producir un total deseado.

El importador CLIP carga únicamente fuentes oficiales fijadas por hash, evitando
el inicializador que descarga nueve backbones. Construye en meta, carga con mmap,
`weights_only=True`, `strict=True` y `assign=True`, y restaura atributos no buffer.
Recalcula texto después de cargar el checkpoint; el caché del constructor antes
de pesos habría sido incorrecto. MPCount conserva nombres, PAC y upsamplers
deterministas, sustituyendo sólo la construcción VGG sin descarga de ImageNet.
Los estados de ambos se comprobaron por claves, formas, dtype y finitud.

Fuentes CLIP reproducibles mediante `scripts/prepare_clip_sources.py` y
`ops/clip-ebc-sources.json`. Licencias de código en `third_party/clip-ebc-LICENSE`
(MIT) y `third_party/mpcount-LICENSE` (Apache-2.0); no se presumen iguales términos
para datos y pesos. No se cambió ni instaló nada en el entorno compartido.

## Recursos y comprobaciones

| Método | Segundos por cuadro, en orden | Total interno de ejecución | RSS máximo | CUDA máxima asignada |
| --- | --- | ---: | ---: | ---: |
| SR | 5,87 / 8,35 / 8,40 | 25,11 s | 1,009 GiB | 336,58 MiB |
| CLIP-EBC | 5,08 / 5,08 / 5,07 | 18,66 s | 1,412 GiB | 705,99 MiB |
| MPCount | 2,80 / 2,73 / 2,73 | 10,39 s | 1,247 GiB | 362,22 MiB |

RTX3060 compartida, ejecuciones secuenciales, lote1; scope2GiB RAM, swap512MiB,
timeout120s, preflight aprobado y ninguna ampliación de cuotas. Latencias con
CUDA sincronizada, incluyen normalización y transferencias por tile; excluyen
carga/importaciones, preparación de muestras y render. El total interno agrega
carga/warmup y escritura, pero excluye imports previos a main y lanzamiento del
scope. La diferencia con mediciones SR anteriores muestra variabilidad de tiempo;
no se extrapola un orden de velocidad universal desde estos tres cuadros.

Python3.13.13, PyTorch2.14.1+cu130, NumPy2.5.4. Entorno privado temporal
`~/home.codex/tmp/dronecrowd-models-env`, con `.pth` que referencia paquetes del
venv compartido sólo en lectura. Añadidos allí einops0.8.0, ftfy6.2.0 y wcwidth0.2.14;
Pillow13.0.0 del entorno CPU para extraer vistas. Cachés y staging en el temporal
autorizado. Los modelos y salidas permanecen fuera de Git.

Cuatro pruebas nuevas verificaron offsets, cobertura y promedio de solapes, factor
MPCount1000 y rechazo de geometría/NaN/timeout. Oráculo CLIP con pesos reales:
salida CPU idéntica bit a bit al alternar caché textual correcto y cómputo dinámico;
streaming idéntico al evaluador oficial batcheado con predictor sintético.
MPCount adaptado y fuente fijada dieron 147 claves iguales y salidas idénticas
de densidad/PAC en32×48; la factory VGG fue sustituida por features compatibles
porque torchvision no está instalado. Estas pruebas acreditan adaptación, no calidad.

Implementación CLIP/runner/vistas: coordinador Codex; MPCount: GPT-6 Luna;
auditoría de fuentes: GPT-6 Astra; revisión independiente antes de GPU: Codex,
modelo efectivo no expuesto. Se fijaron además hashes de muestras/checkpoints en
el runner como solicitó el revisor. La revisión posterior verificó nueve mapas,
sumas, cuatro regiones y tres láminas; advirtió activaciones en estructuras y la
variabilidad de tiempos. La inferencia fue local: no se subieron fotos o videos
a repositorios ni a servicios de los proveedores de estos modelos.

Informes privados bajo `outputs/comparacion-tres-modelos/`:

| Archivo | SHA-256 |
| --- | --- |
| `sr/report.json` | `4ff9330fdcee2a012c2451dac488ecf69b7491042f8ed37ec64e4ee484aae762` |
| `clip-ebc/report.json` | `959adec16e27271827a01bd6a83aa0d6751d5d79086b4385625ea964a9a9899b` |
| `mpcount/report.json` | `d86f6942cc22c72aa192ac298a24d821e85b11e1e89d91c3996dcaf5ed75e2ac` |
| `procedencia.json` | `6f26f7592b9df169e65b251818f073bbea7f87122ca88f23ac8e7afcac32eb18` |
| `controles-en-cuadro.json` | `77dfcb43c1723407ff681882d0dfe129e2205467583b9847bb84d9641db04333` |
| `clip-oracle-cpu.json` | `2686d08be992d6c4d52169ce0f6099e392bdfa1771477d7028fd1d611eb37b64` |

Para repetir, preparar el entorno privado descrito, exportar TMPDIR/TMP/TEMP al
temporal autorizado y comprobar preflight. Cada comando de inferencia debe ir
dentro de un scope con los límites anteriores y un directorio de salida nuevo.
Ejemplo de comando interno para CLIP-EBC:

```sh
DRONECROWD_PYTHON="$TMPDIR/dronecrowd-models-env/bin/python" \
  bash ops/python-ml.sh scripts/evaluate_crowd_candidates.py \
  --model clip-ebc \
  --input outputs/comparacion-tres-modelos/entrada/input.json \
  --checkpoint outputs/modelos/CLIP-EBC/resnet50/best_mae.pth \
  --checkpoint-sha256 da40b664a0a5a57bc33235e7334403e5000271e0741134e301ea706d754bf7df \
  --clip-source "$TMPDIR/dronecrowd-models-audit/clip/source/Yiming-M-CLIP-EBC-b52a594" \
  --output-dir outputs/comparacion-tres-modelos/replica-clip
```

El mismo runner acepta `--model sr` o `--model mpcount` con sus rutas y hashes
indicados, sin `--clip-source`. Las fuentes CLIP pueden reconstruirse en otra
ruta temporal mediante `prepare_clip_sources.py --output-dir RUTA`; ajustar el
argumento correspondiente. La muestra privada no se distribuye por Git.

Próximo paso científico: revisar recortes de estos mapas, completar referencias
exhaustivas y negativos y estudiar sensibilidad al tamaño/contexto de parches,
especialmente MPCount. La cobertura espacial y temporal de una estimación de la
marcha sigue siendo otro trabajo. No se retoma entrenamiento de STNNet ni se
declara un ganador por tener un total mayor.
