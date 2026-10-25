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
completas; las etiquetas parciales nuevas no sirven para inferir masa completa.
El diagnóstico CPU descrito abajo fue revisado y ejecutado sobre ocho fuentes
y dieciséis recortes; véase el estado y sus límites al final.

Antes de más entrenamiento se comparan CPU y sin predictores ocho fuentes
originales completas, una por `source_group`, y sus dieciséis recortes del
replay. En ambos mapas se usa la misma rejilla booleana de ocupación y las
coordenadas históricas truncadas `int(point/2)`. La única variable del kernel
es el sigma: histórico adaptativo `0.1 * suma de distancias a los tres vecinos
más cercanos` frente a sigma fijo 2. Se conserva el soporte `round(4*sigma)`,
la normalización discreta de cada kernel antes de recortarlo al frame y los
bordes históricos sin renormalizar masa perdida. El mapa adaptativo se compara
con la reconstrucción histórica. Se informan colisiones de puntos que ocupan
la misma celda, sigma, pico aislado, masa, pico, entropía y coseno espacial por
frame y crop. Antes de comparar kernels, cada uno de los ocho mapas adaptativos
debe coincidir bit a bit (`numpy.array_equal`) con
`scripts/audit_hdf5.py:reconstruct_density`; el reporte guarda el SHA-256 del
oráculo utilizado. El campo `raw_points_inside_box` cuenta filas MAT crudas en
el recorte, incluidas colisiones en la misma celda; no es un conteo de celdas
ocupadas. Es una comparación de implementación espacial; no mide calidad,
precisión ni comportamiento del modelo y no selecciona variantes.

`scripts/diagnose_target_kernels.py` recibe el replay y su SHA-256 y escribe un
`report.json` nuevo. Sólo usa SciPy/NumPy/Pillow y MAT v5 vía SciPy; no carga
PyTorch, HDF5 ni etiquetas nuevas. Valida los hashes de imagen y MAT, y que
`full_points_xy` del replay coincida exactamente con `image_info.location[:, :2]`
y el campo `number`. Cada frame conserva sus dos recortes originales. Las
métricas muestran sumas retenidas tras el clipping, picos, entropía de masa
normalizada y cosenos espaciales; no renormalizan bordes ni convierten puntos
parciales en mapas completos.

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

Implementación inicial: agente `gpt-6-luna`. Un revisor independiente con Codex
(modelo no identificado) detectó un contrato Path/dict en el diagnóstico de
gradientes; se corrigió y se añadió una prueba CPU de lectura PPM/backward.
También se fijó el hash original y la preservación del gate antes de rechazar.
La comparación de kernels añade pruebas CPU para bordes sin renormalización,
colisiones de coordenadas truncadas, cambio de sigma y equivalencia bit a bit
con el oráculo. El SHA-256 revisado de `scripts/audit_hdf5.py` es
`51c5b80cb82480957c0b1b02a4572f3ea0f1d6a4bdc7e55581ceb7add7831241`.
Revisión independiente final aprobada, con cuatro pruebas de kernels y cuatro
del postprocesado automático comprobadas por el revisor.

## Estado de ejecución

Completados gradientes iniciales sin actualizar el estado, control original-only150,
retención histórica final y comparación CPU de ocho fuentes/16recortes. MAE20,2872
frente a20,3018 original: conserva el control, sin mejora demostrada. Razón de suma
train16=0,98779, sólo dos tensores de la cabeza cambiaron. Los ocho mapas adaptativos
coincidieron bit a bit con el auditor; no se escribieron HDF5. Resultados, hashes y
recursos en [informe de controles](../../docs/controles-ajuste-dominio-2026-10-25.md).
No se ejecutaron original-only200, adaptación conv4 adicional ni ajuste con v3/v4.
