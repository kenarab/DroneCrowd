# Piloto de ajuste con anotaciones revisables y replay original

## Encargo y límites de evidencia

Durante la ventana autónoma de hasta 12 horas, el autor propone continuar
entrenando pesos con nuevas marcas en las fotos y datos originales regenerables.
Se prioriza preparar y comprobar un piloto pequeño de adaptación. No es un
entrenamiento completo ni una promesa de precisión en la marcha.

Etiquetas nuevas: puntos de cabezas visibles con coordenadas, candidatos dudosos,
fuente/hash, recorte y tipo de supervisión. Guardar JSON y exportar MAT compatible
`image_info.location = [x,y,id_local]`, `number = N`; IDs locales no representan
trayectorias. La anotación de agentes se declara provisional y revisable,
separada de los MAT originales. No convertirla en GT humano certificado.

Primeras fuentes de entrenamiento propuestas: la foto `Copy of IMG_8418.JPG`
y recortes del video0265. Reservar la foto `IMG_8415.jpg` y video0269 como fuentes
de evaluación. Un mismo archivo/hash o video no puede contribuir a ambos grupos.
La separación sigue siendo dentro del mismo material aportado, sin demostrar
generalización a otras marchas. Las regiones densas sin anotación exhaustiva
pueden aportar puntos positivos parciales, nunca ceros para personas sin marcar.

## Preparación y pruebas antes de ajustar

1. Ampliar etiquetas claras con recortes pequeños; revisión independiente antes
   de consultar nuevas predicciones o entrenar. Conservar dudosos y zonas ignoradas.
   Preparar una herramienta HTML local para revisar puntos, sin servidor, API,
   subida, telemetría ni dependencias de red.
2. Generar blancos nuevos por puntos con Gaussianas normalizadas, suma una
   persona por punto, válidos incluso para 0,1,2,3 puntos y colisiones. Para
   etiquetas parciales, supervisar sólo vecindades confiables con pérdida
   unilateral `mean(relu(target*100-prediction)**2)`: nunca penaliza masa
   adicional de personas sin marcar. No usar MSE bilateral para puntos parciales.
   Dudosos se ignoran en un radio mínimo de cuatro sigmas; no se usan como
   personas confirmadas ni como fondo. Sigma nuevo: dos píxeles del mapa a
   stride2; radio ignorado mínimo ocho píxeles de mapa / 16 píxeles de entrada.
   Excluir también 32 píxeles de borde del ROI para blancos exhaustivos pequeños.
3. Replay sólo de `train_data` original: subconjunto fijo de secuencias separado
   de test/val, MAT y RGB con hashes. Generar la densidad histórica auditada por
   imagen en memoria y recortar, con caché RAM pequeña; no recrear HDF5 masivos.
4. Confirmar CPU: masa, bordes, colisiones, máscaras, particiones y que el paso
   de optimización cambia exclusivamente los parámetros seleccionados. La
   rutina histórica `mytrain.py` condiciona `optimizer.step` a `loss > den_loss`,
   lo que omite la actualización en modo den puro; usar ruta nueva explícita,
   conservando el archivo histórico como procedencia.
5. Ensayo GPU breve con datos originales antes de usar etiquetas provisionales;
   confirmar pérdida/gradientes finitos y pesos realmente modificados, sin
   interpretar ese ensayo como adaptación a la marcha.

## Experimento exploratorio de adaptación

Si las anotaciones alcanzan para una supervisión explícita y revisada, comparar
como máximo dos pilotos, ambos desde el checkpoint den original verificado:
cabeza den1 solamente y, si hace falta, conv4 más den1. Resto de parámetros
congelado. Lote 1, recortes/contextos pequeños (máximo 512×512), seed fija,
Adam nuevo con LR inicial 1e-5 y gradiente limitado a norma 1. Mezcla fija de un
ejemplo nuevo por cada tres ejemplos originales. Máximo 200 pasos por piloto,
timeout externo de 120 s por trabajo; ningún barrido automático de hiperparámetros.
Documentar las diferencias respecto del entrenamiento original.
129 parámetros entrenables en la primera opción; 147.713 en conv4 más den1.
Cada pérdida se normaliza por los píxeles supervisados válidos de ese ejemplo;
el mismo peso por ejemplo/fuente hace explícita la mezcla 1:3. Informar pérdidas
y pasos por origen, sin confundir número de muestras con área supervisada.

Objetivos: reducir la respuesta en fondos reservados y mejorar señales sobre
personas conservando el control original. Evaluar primero la cohorte histórica
congelada y controles no usados para entrenamiento, antes de más video. La
cohorte test histórica no se usa en gradientes ni para construir blancos.
Las métricas sobre referencias nuevas de agentes son exploratorias; registrar
por separado las métricas contra anotaciones originales humanas del dataset.
No seleccionar un resultado sólo porque aumentó el total de personas predicho.
Criterio de retención fijado: MAE del control histórico repetido de 30 imágenes
no debe superar 1,20× el baseline den1 de 20,3018173 (24,3621808 personas).
Detener ante pérdidas/gradientes no finitos, cambios de parámetros congelados o
colapso a conteos cero. Esa cohorte ya fue observada y sirve de control repetido,
no de evaluación final ciega. Las fuentes nuevas reservadas son escenas
relacionadas, sin independencia fuerte respecto del mismo acto.

## Recursos y entrega

Disco local verificado: 410 GiB libres; SSD externo no necesario. Preflight Nexo
antes de nuevas cargas: temporal 1 GiB, retención estimada 0,5 GiB, RAM inicial
1,25 GiB. El smoke midió 1,68 GiB RSS; actualizar la estimación a 1,75 GiB antes
del piloto mixto, sin ampliar cuotas. Margen y reservas 20 GiB disco / 2 GiB RAM
disponible. Mantener scope High 1,5 GiB,
Max 2 GiB, swap 0,5 GiB; GPU RTX3060 secuencial, sin cambios a drivers/venv.
No ampliar cuotas si falla: reducir el experimento y registrar el fallo.
Temporales/cachés en `~/home.codex/tmp`, salidas persistentes privadas en
`outputs/ajuste-dominio/`, máximo 512 MiB adicionales. Modelos originales intactos.

Entrega: etiquetas/puntos revisables y procedencia, herramienta offline,
generación de blancos en memoria, entrenador acotado comprobado, resultados
antes/después o bloqueo científico explícito si no hay etiquetas suficientes,
revisión independiente y commits atómicos. No publicar ni instalar dependencias
en el venv compartido. DM-Count queda como alternativa investigada, no ejecutada.

## Estado

Preparadas siete regiones: cuatro de entrenamiento (dos de puntos parciales y
dos negativos), tres reservadas. Se excluyen los positivos densos irresolubles
de video y el grupo ambiguo de la fuente. Anotaciones congeladas privadas:
`outputs/ajuste-dominio/anotaciones-v2.json`, SHA-256
`b7b37952b31d2efd3c3c3e6079587be88096f9b5c368bc56c46ccd2e0f6cfa1b`.
HTML offline y siete MAT en `outputs/ajuste-dominio/revision-v2/`.
El recuento nominal de puntos no equivale a masa o centros supervisados: bordes
y zonas dudosas se excluyen y se registran en cada paso.

Replay: ocho secuencias equidistantes entre las 82 de entrenamiento, una imagen
midpoint disponible y dos recortes nativos 512×512 por imagen. Particiones
disjuntas comprobadas. Preparador implementado por Codex GPT-6 Luna, selección
corregida por revisión del coordinador y revisada independientemente por el agente
existente. Sin HDF5 ni escrituras en el dataset.

Smoke GPU original-only de cinco pasos completo: cambia únicamente weight/bias
de den1 (129 parámetros); tres controles originales con masa positiva mantienen
conteos positivos. CUDA pico 246.253.056 bytes, RSS 1.765.348 KiB, 2,72 s internos.
Las pérdidas no se interpretan como precisión. Código ejecutado preservado con
el hash del informe en `outputs/ajuste-dominio/executed-code/`.
El detector de colapso excluye blancos negativos y exige tres recortes originales
con masa >=1; falla si todos sus conteos clamped son cero. Es un guardia interno,
separado del control histórico de retención de 30 imágenes.
Tres pruebas CPU del entrenador verifican actualización real, congelados,
contexto fuera del ROI y detección de colapso sin confundir negativos.
Dos pilotos mixtos completos, ambos de 200 pasos desde el original (150 replay,
50 nuevos). Head129 y conv4+head147713 cambian sólo los tensores permitidos.
MAE histórico posterior35,02254 y112,87392, ambos mayores que24,36218 permitido;
también aumentan las respuestas en los tres controles reservados de la marcha.
Ambos se rechazan. No ejecutar más barridos con estas etiquetas/pérdidas.
Original y checkpoints exploratorios se conservan separados.
Revisor independiente existente aprobó preparador, referencias y entrenador;
herramienta Codex, modelo no identificado. Preflight actualizado a 1,75 GiB
admitido sin cambios de cuotas. El HTML exportado pasa comprobación de sintaxis
JavaScript; todavía no se hizo una prueba interactiva en navegador.

Entrega y límites en `docs/piloto-ajuste-dominio-2026-10-25.md`; evidencias privadas
en `outputs/ajuste-dominio/` (~262MiB, límite512MiB). Las doce pruebas nuevas pasan.
SSD externo no requerido y ningún HDF5 masivo reconstruido.
Revisión independiente final aprobó conclusiones, código de informe y figuras:
herramienta Codex, modelo no identificado. Verificados 18 hashes de fuentes del
informe, 16 archivos originales usados por replay y el checkpoint den original
intacto. Piloto completado; ninguna mejora de precisión en marcha demostrada.

Auditoría posterior solicitada por el autor (SPEC006): Astra y Fable coinciden
en que el diseño quedó insuficientemente controlado. La aprobación de ejecución
previa no acredita validez de la receta; falta control original-only150/200 y
diagnóstico de objetivos/gradientes. Se conserva el rechazo de ambos pesos, sin
inferir inviabilidad de la adaptación. Ver `docs/revision-conjunta-entrenamiento-2026-10-25.md`.

Implementados `domain_targets.py`, `domain_annotations.py` y el revisor HTML
offline. Nueve pruebas: seis de masa/máscaras/gradientes (venv ML, CPU) y tres
de MAT/particiones (Python base). Revisión independiente del contrato científico:
agente existente `revision_retencion_hdf5`; formato/MAT/HTML: Codex GPT-6 Luna
con contexto acotado, autorizado por el autor para reducir coste. Se corrigió
su observación sobre el radio de borrado al cambiar zoom. Los puntos visuales
propuestos por Luna requieren segunda revisión; no se aceptan por el solo hecho
de pasar la validación de formato.
