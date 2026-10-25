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
   etiquetas parciales, supervisar sólo vecindades confiables y enmascarar el
   resto; dudosos no se usan como personas confirmadas ni como fondo.
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

Objetivos: reducir la respuesta en fondos reservados y mejorar señales sobre
personas conservando el control original. Evaluar primero la cohorte histórica
congelada y controles no usados para entrenamiento, antes de más video. La
cohorte test histórica no se usa en gradientes ni para construir blancos.
Las métricas sobre referencias nuevas de agentes son exploratorias; registrar
por separado las métricas contra anotaciones originales humanas del dataset.
No seleccionar un resultado sólo porque aumentó el total de personas predicho.

## Recursos y entrega

Disco local verificado: 410 GiB libres; SSD externo no necesario. Preflight Nexo
antes de nuevas cargas: temporal 1 GiB, retención estimada 0,5 GiB, RAM 1,25 GiB
con margen y reservas 20 GiB disco / 2 GiB RAM. Mantener scope High 1,5 GiB,
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

Preparación y ampliación de referencias en curso. Ningún ajuste de pesos ejecutado
todavía; el checkpoint original sigue siendo la única versión usada hasta aquí.
