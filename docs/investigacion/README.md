# Investigación y selección de modelos

[Nota sobre upscaling, escala y personas dispersas](resolucion-y-densidad.md).
[Primera prueba ejecutada de SR y tiempos de video](../prueba-sr-2026-10-25.md).
[Los tres candidatos ejecutados sobre el mismo tramo](../comparacion-tres-modelos-video-2026-10-25.md).

Consulta inicial:25/10/2026. Hay alternativas posteriores a STNNet que merecen
comparación. Ninguna está validada todavía para nuestros videos de Buenos Aires.
Esta selección es dirigida, no una revisión sistemática ni un ranking universal.
Se conserva STNNet como referencia reproducible, sin suponer que su antigüedad
determina el resultado.

## Candidatos y preguntas contrastables

| Trabajo | Tarea y motivo para estudiarlo | Disponibilidad comprobada | Estado local |
| --- | --- | --- | --- |
| [SR, CVPR2026](https://github.com/wbshu/SR) | Conteo por cuadro en video; incluye DroneCrowd y regularización temporal durante entrenamiento | Código MIT; release v1 `dronecrowd.tar`, hash comprobado | Ejecutado: ocho controles y tres cuadros nativos; calidad local sin validar |
| [CLIP-EBC, ICME2025](https://arxiv.org/abs/2403.09281) | Conteo supervisado de multitudes por clasificación de bloques y densidad; otra familia de representación | [Código MIT y pesos/logs](https://github.com/Yiming-M/CLIP-EBC); repo archivado22/10/2026 | NWPU ResNet50 ejecutado en tres cuadros; anomalía del paquete ViT actual documentada |
| [MPCount, CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Peng_Single_Domain_Generalization_for_Crowd_Counting_CVPR_2024_paper.html) | Transferencia entre dominios y ambigüedad entre personas y fondo | [Código Apache-2.0 y enlaces a pesos](https://github.com/Shimmer93/MPCount) | STA determinista ejecutado en tres cuadros; calidad local pendiente |
| [MovingDroneCrowd, ICCV2025](https://github.com/fyw1999/MovingDroneCrowd) | Individuos únicos a través del video, con cámara móvil | Enlaces de pesos; ficha CC-BY-NC-4.0; SDNet disponible reentrenado | Otra tarea: requiere especificación y evaluación temporal |
| [P2RLoss, CVPR2025](https://arxiv.org/abs/2505.21943) | Cómo supervisar fondo y puntos con pocos datos completos | [Código](https://github.com/Elin24/P2RLoss); licencia y checkpoints sin verificar | Referencia metodológica, no ejecución inmediata |

La revisión ampliada de [Astra](astra-bibliografia.md) conserva las fuentes
primarias y sus límites: DM-Count, P2PNet, CAN, NWPU-Crowd, DecideNet, URM, T2ICount
y DCA-MoE, además de los anteriores. Las métricas de papers con datasets o tareas
distintos no se comparan directamente con nuestra MAE histórica20,30.

Para SR se comprobó el [listado oficial de assets v1](https://github.com/wbshu/SR/releases/expanded_assets/v1):
`dronecrowd.tar`, 86,1 MB (82,1 MiB), SHA-256 publicado
`22ff7ff7bfcfa6cb45b38ea0df6dd74f388acc070b41bbb2d64a7308062097eb`.
El hash ya se recalculó y el binario se cargó de forma restringida: 118 tensores,
metadata `epoch=10`; arquitectura con 21.510.465 parámetros. Ver prueba enlazada.
La licencia del código no se extiende automáticamente a pesos o datasets.

## Orden de trabajo

1. Cerrar el diagnóstico y control de la receta STNNet antes de atribuir su
   deterioro a las nuevas fotos. [Revisión del entrenamiento](../revision-conjunta-entrenamiento-2026-10-25.md).
   El control150 ya se ejecutó y conserva MAE20,287, sin demostrar mejora:
   [resultados](../controles-ajuste-dominio-2026-10-25.md).
2. Revisar implementación de SR y CLIP-EBC: commit exacto, variante, pesos,
   licencia, preprocesado, normalización del conteo y carga restringida.
   Un README ejecutable no demuestra compatibilidad con nuestro entorno.
3. Especificar una prueba pequeña con cuotas vigentes, entorno propio si hace
   falta y archivos fijados por hash. Medir primero ejecución y recursos; la
   RTX3060 de12GB no acredita que se cumpla el límite de RAM del proceso.
4. Comparar con STNNet sobre un conjunto local anotado de forma exhaustiva,
   separado por fuente/evento y con negativos urbanos. Congelar métricas y
   tolerancias antes de inferir. La muestra provisional actual no alcanza para
   decidir superioridad ni precisión en una manifestación completa.
5. Considerar combinación sólo si aparecen errores complementarios y una regla
   fijada en validación mejora frente al mejor modelo individual. El acuerdo
   entre modelos no valida una cantidad de personas.

La inferencia con pesos congelados puede ejecutarse antes de terminar etiquetas;
las anotaciones son necesarias para medir error. Las propuestas automáticas
ayudan a revisar y no se convierten por sí solas en referencia verdadera.

No se eligió aún una arquitectura híbrida, no se descargaron datasets nuevos y
no se necesita regenerar todos los HDF5. La primera prueba SR tiene SPEC012 y
resultados propios; SPEC013 incorpora CLIP-EBC y MPCount en el mismo tramo nativo.
No se ha validado la precisión local de ninguno.

## Reproducibilidad y revisión

El [marco SDD](../../sdd/000-investigacion-reproducible/SPEC.md) exige evidencia
trazable, resultados adversos preservados y discusión por personas con competencia
técnica y profesional y honestidad intelectual. Las discrepancias se documentan
y se resuelven con comprobaciones explícitas. La revisión por agentes ayuda a
detectar errores; no sustituye revisión científica externa ni etiquetas humanas.

Astra (`gpt-6-astra`) verificó publicaciones y repositorios oficiales. El
coordinador contrastó repositorios, releases y la referencia ICME de CLIP-EBC.
La primera consulta adicional a Fable falló por sesión OAuth vencida. Después de
que el autor renovara el acceso, se completaron dos respuestas con
`is_error=false` y modelo real `claude-fable-5-1`, sin herramientas ni fuentes
externas nuevas. Astra contestó sus objeciones y cerró el intercambio. El
[acta bibliográfica](discusion-bibliografica.md) conserva rectificaciones y
pendientes; los dictámenes originales no se reescriben como si no hubieran tenido
errores. La revisión anterior del entrenamiento permanece separada. No se
modificó la autenticación original ni se enviaron fotos/videos a Fable.

Código y documentos se publican en el fork autorizado; imágenes, videos,
etiquetas con rutas privadas, pesos y salidas permanecen fuera de Git. Cuando
no se puedan distribuir imágenes, publicar la receta y las limitaciones de
acceso sin afirmar que todos pueden reproducirlas únicamente clonando el repo.
