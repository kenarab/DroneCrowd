# Control con anotaciones originales de DroneCrowd

La adaptación y los pesos de densidad producen conteos cercanos a las anotaciones
en una muestra del dominio original. No aparece un error grueso de unidades o del
factor 100 que explique los resultados de la marcha. Esto no valida precisión
en Buenos Aires: perspectiva, densidad, oclusión y escala son distintas.

Se congeló antes de inferir un fotograma de cada una de las 30 secuencias test:
imagen central disponible, índice `len//2` en orden lexicográfico. En esta copia
es el frame 152. Hay 43–252 personas anotadas por imagen y 4.105 en total.
Se cuentan filas `(x,y,id)` de los MAT originales, verificando `number` y límites;
el ID no se suma como personas. No se regeneraron HDF5.

| Método / salida | Imágenes | MAE | RMSE | Sesgo medio |
|---|---:|---:|---:|---:|
|Cuatro crops históricos, den1 (principal)|30|20,30|23,82|−3,22|
|Mosaico con contexto, den1|30|19,45|22,99|−1,05|
|Cuatro crops, den2|30|23,00|27,75|−3,99|
|Cuatro crops, den3|30|22,14|29,35|−6,62|

MAE es el error absoluto medio en personas por imagen; RMSE penaliza más los
errores grandes; sesgo es predicción menos anotación. Las tres cabezas se miden
por separado, con den1 como criterio principal fijado de antemano. No se ajustó
una constante a partir del cociente de conteos.

No es una reproducción del score publicado ni del benchmark completo de 4.500
imágenes. La selección es determinista por secuencia y no se presenta como una
muestra aleatoria con incertidumbre estadística estimada. El piloto de seis
secuencias y las 24 restantes son disjuntos; sus filas se combinaron antes de
calcular el resultado de 30, sin promediar medias con pesos incorrectos.

## Evidencia local y protocolo

- [Puntos originales y mapas en cuatro ejemplos](../outputs/validacion-controles/reporte-historico/controles-historicos.png).
- [Gráficos de métricas](../outputs/validacion-controles/reporte-historico/metricas.png).
- [CSV con las 60 comparaciones](../outputs/validacion-controles/reporte-historico/counts.csv).
- [Resumen y hashes](../outputs/validacion-controles/reporte-historico/summary.json).

JPG/MAT originales leídos sin modificaciones; RGB mediante Pillow, como el loader
histórico, sin reescala. Cuatro cuadrantes nativos de 960×540, lote 1, normalización
ImageNet y clamp de la suma por cuadrante antes de dividir por 100. El mosaico usa
núcleo 320×128 y halo 96, conserva sólo núcleos y aplica las mismas cuatro sumas
globales, sin duplicar solapes. Los mapas guardados son float32 crudos, den1.

Las imágenes históricas elegidas responden sobre los grupos anotados en los
ejemplos inspeccionados. El ejemplo nocturno también muestra respuesta difusa
al fondo. Estas observaciones explican por qué un conteo global cercano al GT
no garantiza que cada persona esté localizada correctamente.

Cohorte: `outputs/validacion-controles/historico-validado/cohort.json`, SHA-256
`f94dd4a09e014b57cae0b365fd4dbbccea8598599157f18e40a718521d9e2f8e`.
Ejecuciones: `cuadrantes-piloto/`, `cuadrantes-restantes/`, `mosaico-historico/`.
Comandos: `pilot-command.json` y `cohort-commands.json`, bajo el mismo directorio.
Cada reporte registra hashes, versiones, límites kernel y tiempos. Una copia del
código exacto ejecutado se conserva en `executed-code/`; posteriormente sólo se
añadió un argumento de dispositivo para probar la receta con un fixture CPU.

## Comprobaciones y recursos

17 regresiones de video/mosaicos pasadas; un test de esquema MAT y dos tests de
métricas/ensamblado histórico, incluidos rechazo de anotaciones inválidas y
separación de cabezas. El revisor independiente Codex autorizado
`/root/revision_retencion_hdf5` comprobó los 30 MAT, 60 mapas y la combinación
de métricas; su modelo efectivo no está expuesto.

RTX 3060, PyTorch 2.14.1+cu130; sin instalaciones ni cambios al venv. GPU máxima
asignada: aproximadamente 425,5 MiB con crops históricos y 161,1 MiB con mosaico.
Pico RSS menor de 1,04 GiB. Trabajos de 6,09 / 10,25 / 29,88 s dentro del
evaluador, excluyendo arranque y preparación. Límites efectivos de 2 GiB RAM,
swap 0,5 GiB, timeout externo 120 s por trabajo y ejecución secuencial.

La preparación inicial produjo una falsa alarma al comparar el tiempo de acceso
del archivo, que cambia al leerlo. Se corrigió la guarda para comparar identidad,
tamaño y mtime; el fallo y su primer RGB parcial se conservaron por separado.
Las fuentes no se modificaron.

La continuación evalúa referencias visuales provisionales de los archivos
aportados y la cabeza de densidad del checkpoint de localización. Es una
comparación separada, con los mismos controles congelados y sin atribuirle aún
una mejora de precisión en la marcha. [SPEC](../sdd/004-validacion-controles/SPEC.md).
