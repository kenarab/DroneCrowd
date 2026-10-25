# Preparación de validación extendida para el dispatcher GPU

Encargo del autor, 25/10/2026: extender la prueba y preparar tareas que un
dispatcher compartido ejecute cuando esté disponible. El alcance de este cambio
es el cliente del proyecto: no instalar ni implementar otro dispatcher, modificar
servicios compartidos o ejecutar la GPU por fuera de la cola.

## Preguntas y selección previa

La estabilidad de MPCount entre tres cuadros no valida su precisión. Examinar
respuesta y variación espacial durante toda la duración disponible, además de
fondos, carteles, cambios de encuadre y artefactos de parche. Mantener SR y
CLIP-EBC RN50 como comparadores con los mismos pesos y reglas de SPEC013.
No entrenar, elegir un checkpoint por conteos deseados ni sumar cuadros.

Preparar tres videos locales a resolución nativa y tiempos enteros válidos desde
cero, con muestreo de un cuadro por segundo:

| Fuente | Duración | Tiempos solicitados | Cuadros | Lotes de hasta dos |
| --- | ---: | --- | ---: | ---: |
| 0265 | 22,856167 s | 0–22 s | 23 | 12 |
| 0269 | 9,392717 s | 0–9 s | 10 | 5 |
| trim | 10,985 s | 0–10 s | 11 | 6 |

Total: 44 cuadros únicos, 23 lotes de preparación y 69 trabajos GPU (un modelo
por lote). Esto cubre los instantes muestreados de toda la duración, no cada uno
de sus 2592 cuadros. El último tramo entre muestras no se observa continuamente.
Registrar PTS efectivos, hashes de originales/RGB y cualquier borde excluido.

Los dos videos grandes quedan en 3840×2160. El trim mantiene píxeles nativos,
excluyendo explícitamente 12 filas inferiores: 960×528 para los tres modelos,
sin interpolar. No combinar clips como vistas simultáneas; sus metadatos indican
horas distintas y no prueban sincronización ni cobertura completa de la marcha.

## Cliente y contratos

Primero preparar muestras y manifiestos usando sólo CPU; mantener el presupuesto
del muestreador existente de 120 MiB planificados por lote. Reutilizar únicamente
entradas completas verificadas por hash y geometría. Reanudar preparación sin
sobrescribir resultados ni aceptar datos parcialmente escritos como completos.

Generar un plan privado con identificadores idempotentes, proyecto, revisión de
código y hashes, modelo/checkpoint, artefacto de entrada y estimaciones de recursos.
Identidad de trabajo derivada de contenido, no de hora o contador mutable. Separar
planificado, preparado, enviado y ejecutado: un JSON local no constituye encolado.
No incluir datasets, credenciales ni shell arbitrario en solicitudes al coordinador.
Las rutas de datos y resolución de artefactos quedan del lado del proyecto.

El perfil propuesto `dronecrowd-native-count-v1` aún debe registrarse/aceptarse en
el dispatcher real. Su invocación tipada conservará hashes de código/checkpoint
y manifiesto, rutas de entrada/salida del proyecto y variantes permitidas. No
crear una API supuesta, una cola paralela o una concesión GPU ficticia para avanzar.
Si la interfaz aparece, integrar y enviar los trabajos autorizados, conservando
recibos reales. No afirmar envío ni ejecución sin esa evidencia.

Generalizar el contador de SPEC013 para aceptar otro manifiesto cuyo SHA-256 se
declare explícitamente; mantener el hash anterior como valor predeterminado.
Máximo tres cuadros por proceso y geometría acotada, múltiplos de16; estos lotes
usan como máximo dos. Mantener anclajes, contextos, promedio de solapes y unidades
de cada método. Registrar el estado inicial y después de cada cuadro, para que
interrupciones no oculten resultados parciales. No reintentar un fallo ampliando
límites ni sobreescribir automáticamente su directorio.

## Recursos y comprobaciones

Sin ejecución GPU durante la preparación. Por trabajo GPU: lote1, un hilo CPU,
máximo2GiB RAM y512MiB swap, 120s; estimación conservadora1GiB VRAM y1,5GiB RAM
basada en SPEC013. La admisión real debe comprobar recursos frescos y sus márgenes;
los límites no reservan físicamente la RTX3060 ni garantizan exclusividad.

Preflight CPU y disco antes de materializar: reservar20GiB libres/2GiB RAM host,
estimar4GiB adicionales para RGB, previews, futuros mapas y resúmenes. Ejecución
CPU acotada por scope2GiB/120s; si no termina, preservar progreso y reanudar de
forma explícita. Todos los temporales/cachés en `~/home.codex/tmp`; venv compartido
sólo lectura. No instalar dependencias nuevas ni reconstruir HDF5.

Pruebas CPU de muestreo, partición, IDs, cambios de contenido, errores de límites
y reanudación. Revisión independiente antes de materializar y antes de enviar
GPU. Verificar44 RGB completos,69 trabajos materializados y ninguna ejecución
GPU al cerrar esta fase si no existe interfaz. Documentar la diferencia entre
estimación y tiempo real; sin GT no habrá MAE local ni cifras de asistencia.

Datos, planes privados y resultados quedan fuera de Git. Código, SPEC, metodología
y límites se publicarán en commits atómicos; conservar cambios ajenos.

## Estado inicial

La consulta de servicios y código accesibles no encontró un dispatcher GPU
operativo ni una CLI/API de envío. Hay diseño documental; no se acredita ausencia
en otras ubicaciones. Se solicitó al autor la interfaz si está instalándose en
otro lugar. No se enviaron trabajos ni se inició una campaña por este documento.
