# Controles para el ajuste de dominio

## Propósito

Separar los efectos de la receta de entrenamiento de la incorporación de
etiquetas nuevas. El primer control ejecuta 150 pasos sobre las 16 muestras
originales del replay, usando el mismo orden cíclico, inicialización, tasa,
optimizador, tamaño de lote y clipping del piloto mixto. Los pesos de entrada
son siempre los del checkpoint original verificado. Adam se inicializa de cero
en cada corrida; por eso el estado del reloj y de sus momentos difiere entre
control y piloto y la comparación no aísla una suma lineal de gradientes.

## Diagnóstico antes de entrenar

Antes de actualizar pesos se registrará, por cada ejemplo de las 16 muestras
originales y 4 nuevas de entrenamiento, el gradiente inicial de los 129
parámetros `den_output_layer1` (orden estable), su norma antes y después de
clipping global a 1, los signos y componentes del bias, y norma/dirección.
Se informarán productos escalares y cosenos por fuente, además de agregados
ponderados por las frecuencias de la receta mixta: 150 pasos originales, 26
parciales positivos y 24 negativos. Estos agregados describen gradientes en el
punto inicial; no predicen directamente los pasos de Adam.

La comparación espacial de kernels se limita a anotaciones originales
completas. Si el alcance o tiempo no permite verificarla, se registra como
pendiente; las etiquetas parciales nuevas no sirven para inferir masa completa.

## Criterios previos de interpretación

- El umbral histórico de MAE es 24.3621808 y la referencia original reportada
  es MAE 20.3018173. Se reportan como contexto, sin usar el conjunto test de 30
  imágenes para elegir checkpoints, hiperparámetros o variantes.
- Retención histórica aceptable: MAE `<= 24.3621808`.
- En las 16 muestras train, comparar suma de conteos predichos posterior y
  original. El cociente `posterior/original` debe quedar en `[0.5, 1.5]` para
  detectar colapso o inflación grave. Este gate es una protección de estabilidad
  del entrenamiento y no evidencia de generalización.
- Presupuesto de ejecución: máximo 2 GiB RAM y 120 segundos. No regenerar ni
  almacenar HDF5; los mapas originales se reconstruyen en memoria.

## Límites

El control no usa ejemplos nuevos ni sus etiquetas. Las salidas son diagnósticas,
no una evaluación de precisión. Cualquier selección por calidad requiere una
decisión definida antes de consultar resultados de test.

## Revisión previa

Implementación: agente `gpt-6-luna`. Un revisor independiente con Codex (modelo
no identificado) detectó un contrato Path/dict incorrecto en el diagnóstico; se
corrigió y se añadió una prueba CPU que recorre lectura PPM, backward y clipping.
También se fijó el hash original y la preservación de valores del gate antes de
rechazar. Revisión final sin bloqueantes antes de ejecutar GPU. Cuatro pruebas
del control y tres del entrenador aprobadas. Ejecución y resultados pendientes.
