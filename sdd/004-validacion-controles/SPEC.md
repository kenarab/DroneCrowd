# Controles anotados y calibración de STNNet

## Encargo

El 24/10/2026 el autor autoriza continuar autónomamente hasta 12 horas con las
pruebas necesarias. Confirma interés en conteo GPU. Ventana iniciada con el goal
de esta sesión; no obliga a consumirla si se alcanza una conclusión verificable.
Mantener revisión independiente por el agente ya autorizado, commits locales
atómicos y originales intactos. Sin publicar, entrenar, modificar el venv ni
invocar nuevos servicios/LLM. Inferencia secuencial en RTX 3060, lote 1.

## Orden de decisión

1. Control positivo con GT histórico: elegir, antes de inferir, el fotograma
   central de cada una de las 30 secuencias de `test_data` (orden lexicográfico,
   índice `len//2`). Evaluar primero seis secuencias espaciadas (índices 0,5,10,
   15,20,25); si la carga es válida, completar las 30. No es el benchmark oficial
   completo de 4500 imágenes ni una muestra para ajustar pesos.
2. Comparar cuatro cuadrantes sin reescala (receta histórica, 960×540 por crop)
   con mosaico/halo ya probado, sobre las mismas imágenes. Verificar las tres
   cabezas por separado; usar den1 como criterio principal fijado previamente.
   MAE/RMSE/sesgo sólo respecto del GT original, nunca respecto de conteos inventados.
3. Controles de marcha: elegir y congelar recortes por contenido visual antes de
   consultar sus predicciones: edificios sin personas, personas distinguibles y
   multitud densa. Coordenadas, fuente, PTS y hash trazables. Marcar puntos de
   personas visibles donde sea defendible, con casos dudosos separados.
   La anotación visual del asistente es provisional, no un GT humano validado;
   no presentarla como precisión certificada. Si no pueden contarse, declararlo.
   Inspección inicial: los recortes densos del video son borrosos para referencias
   exhaustivas. Las fotos JPG ya aportadas contienen peatones más distinguibles;
   se permite añadir controles positivos/negativos de esas fotos, identificados
   por separado y sin asumir que equivalen al mismo instante de los videos.
4. Medir sensibilidad a resolución/contexto y falsas respuestas con un número
   acotado de comparaciones derivadas de los controles. Elegir el siguiente paso
   según evidencia; no expandir automáticamente a videos completos.
   Primera comparación de ROI fijada: ocho controles (cinco de video, tres de
   foto), contexto exterior de 128 píxeles, escala nativa y Lanczos 2× en dos
   trabajos. Integrar cada ROI sobre su mapa con coordenadas alineadas, sin
   multiplicadores de área en el conteo. Los intervalos son de candidatos
   visuales provisionales; la masa de Gaussianas puede cruzar sus bordes y
   ninguna diferencia pequeña certifica un error de conteo de centros.
5. Sólo si es útil, evaluar la variante local de localización del ZIP existente,
   tras inspección estática/carga restringida y validación estricta de sus pesos.
   No usar variantes de tracking ni introducir dependencias de CUDA compiladas.
   Activación de esta rama: el modelo den cuenta aproximadamente los controles
   históricos y peatones claros, pero asigna masa alta a una fachada y apenas
   responde a las regiones densas. Comparación fijada: mismo piloto histórico
   de seis escenas y los ocho ROIs congelados a 1×/2×. `loc_model_best.pth.tar`
   contiene metadata float64 antigua inspeccionada estáticamente: mismos globals
   del modelo den, sin permitir nuevas clases, excepción sólo para su SHA-256
   `88168a85079b4931b86ea1145d501ad5e73d3d27e903fedf4c049d62514a8d5d`.
   Carga explícita `use_loc=True`, tracking desactivado, estado completo estricto;
   comparar su cabeza de densidad sin sumar cabezas ni descartar parámetros.

## Anotaciones, recursos y aceptación

El MAT de fotograma contiene filas `(x,y,id)`: contar filas válidas, conservar
el identificador como metadata y no sumarlo como personas. Registrar hash del
JPG/MAT, geometría y puntos. No regenerar mapas masivos. Videos/fotos/dataset
se leen; salidas en `outputs/validacion-controles/`, excluidas de Git.

Presupuesto nuevo de artefactos: 512 MiB; temporales 1 GiB, estimación RAM
1,25 GiB por proceso con margen de preflight Nexo y reservas 20 GiB disco/
2 GiB RAM. Mantener scope High 1,5 GiB / Max 2 GiB / swap 0,5 GiB y timeout
externo 120 s por trabajo acotado; dividir cohortes en lotes pequeños si hace
falta. GPU: consultar ocupación, no paralelizar, estimación pico inicial menor
de 0,75 GiB VRAM para crops históricos; no se declara una reserva GPU aislada.
Si una cuota falla, registrar y reducir el trabajo sin ampliar límites.
Todos los temporales/cachés en `~/home.codex/tmp`.

Entrega: manifiestos, métricas con procedencia y denominadores claros, figuras
locales de controles y puntos, código reproducible, revisión independiente y
una decisión sobre seguir con STNNet o cambiar de enfoque. Resultados negativos
y límites deben quedar conservados. Registrar cualquier desviación antes de
ejecutar el trabajo afectado.

## Estado

Selección histórica y controles visuales en preparación; GPU aún sin ejecutar
en esta fase.
