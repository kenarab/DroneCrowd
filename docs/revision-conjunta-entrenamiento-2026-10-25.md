# Revisión conjunta del entrenamiento: Astra y Fable

El piloto no estaba suficientemente controlado para evaluar la propuesta de
ajustar el modelo. Los dos pesos nuevos empeoraron realmente; ese resultado no
demuestra que el ajuste con buenas anotaciones no pueda funcionar. Faltó un
control equivalente con datos originales solos y se cambiaron varias decisiones
al mismo tiempo. El coordinador debió aislarlas antes de interpretar resultados
o ampliar la capa entrenable.

La revisión se realizó con **GPT-6 Astra (`gpt-6-astra`)** y **Claude Fable5.1
(`claude-fable-5-1`)**. Este informe distingue acuerdos, rectificaciones e hipótesis
pendientes. No hubo nuevos entrenamientos ni inferencias GPU durante la discusión.

## Material y discusión efectivamente realizados

Ambos recibieron el [expediente completo](expediente-revision-entrenamiento-2026-10-25.md):
receta, código, datos, etiquetas efectivas, pérdidas, checkpoints/hashes, resultados,
pruebas y controles ausentes. La primera lectura fue independiente, sin consultar
el dictamen del otro. Luego se intercambiaron objeciones y réplicas, mediadas por
el coordinador. Los textos originales se conservan, incluyendo sus errores.

| Ronda | Astra | Fable |
| --- | --- | --- |
| Dictamen independiente | [Ronda1](revisiones-entrenamiento/astra-ronda1.md) | [Ronda1](revisiones-entrenamiento/fable-ronda1.md) |
| Contraste y objeciones | [Ronda2](revisiones-entrenamiento/astra-ronda2.md) | [Ronda2](revisiones-entrenamiento/fable-ronda2.md) |
| Réplica final | [Ronda3](revisiones-entrenamiento/astra-ronda3.md) | [Ronda3](revisiones-entrenamiento/fable-ronda3.md) |

Entre las dos primeras rondas ambos recibieron el mismo
[suplemento medido](revisiones-entrenamiento/suplemento-ronda2.md): checkpoints
inspeccionados en CPU con carga restringida y diferencias de mapas/JSON existentes.
Las últimas réplicas se prepararon en paralelo: Astra ronda3 todavía menciona
objeciones que Fable rectifica en su propia ronda3. Aquí se consideran rectificadas
las concesiones explícitas, sin presentar aquel desacuerdo temporal como vigente.

Astra fue un agente separado sin historial heredado. Fable se ejecutó mediante
`claude --model fable`, CLI2.1.308, con modelo real confirmado en los tres JSON.
Primera ronda: sólo Read/Glob/Grep. Rondas siguientes: documentos completos
incluidos, sin herramientas. Fueron consultas nuevas con sus dictámenes previos,
sin persistencia de sesión. No se habilitaron escritura, shell, agentes o MCP.
Solicitudes y respuestas originales: `outputs/ajuste-dominio/revision-diseno/`.

## Resultados que siguen siendo válidos

| Modelo | MAE histórico,30 imágenes | Sesgo medio | Fachada reservada | Peatones reservados |
| --- | ---: | ---: | ---: | ---: |
| Original | 20,30 | −3,22 | 28,12 | 10,25 |
| Cabeza | 35,02 | +33,95 | 35,25 | 11,99 |
| Conv4+cabeza | 112,87 | +112,87 | 54,14 | 16,74 |

Ambos pesos incumplen el umbral previo de MAE24,36218 y empeoran incluso negativos
usados para entrenar. Ambos revisores respaldan rechazarlos. El original queda
como referencia intacta, sin validarlo como contador de la marcha. Los conteos ROI
son integrales de densidad, no detecciones individuales. Las referencias nuevas
son provisionales: tres cabezas claras y un candidato dudoso en peatones; ninguna
persona discernible en fachada. Las30 imágenes históricas tienen anotaciones
originales, pero ya se habían observado: no son un test final ciego.

## Problemas concretos de diseño

**Faltó el control fundamental.** El smoke original de cinco pasos verificó
actualización y conservación de congelados. No verificó qué hace la receta después
de150/200 pasos sin fotos nuevas. Por eso no se separa deterioro de la ruta base
y adaptación. El entrenador además limita `--original-only` a cinco pasos: antes
del control propuesto habría que cambiar ese límite mediante un cambio revisado.

**La pérdida podía mejorar mientras empeoraba el conteo.** Los ROIs positivos ya
recibían más masa que sus candidatos visibles. Sin embargo, la pérdida parcial
pedía alcanzar picos estrechos y nunca penalizaba exceso. En repeticiones del mismo
ROI la pérdida baja mientras suben los conteos y empeoran negativos. Eso cuestiona
la correspondencia del objetivo con la finalidad práctica, sin identificar aún
toda la causa del deterioro.

**Existe un mecanismo matemático preocupante.** Den1 recibe features no negativas
(ReLU multiplicada por sigmoid). La derivada de la pérdida parcial respecto a cada
salida es no positiva; también lo son sus derivadas respecto a todos los pesos
y bias de la cabeza final. Con features congeladas, un paso de SGD simple sobre
ese parcial sólo podría elevar o mantener salidas de otras imágenes. Ambos aceptan
esta demostración de Astra. Adam con historia y fuentes mezcladas no hereda esa
conclusión por actualización; tampoco se determina el gradiente de conv4.

**Frecuencia1:3 no garantizaba protección.** Se promediaban pérdidas sobre703/1805
píxeles positivos frente a65.536 de replay y9216 en negativos. La frecuencia por
ejemplo sí era1:3, pero la influencia no se deduce de ella. Un denominador común
tampoco garantizaría equilibrio: importan residuos, features, direcciones, clipping
y momentos del optimizador.

**Poca supervisión efectiva y varios cambios simultáneos.** Diecinueve marcas
nominales de dos recortes de una foto se redujeron a11 centros supervisados, sin
positivos nuevos exhaustivos ni revisión humana externa. Además cambiaron LR
(diez veces respecto al código histórico), beta1, estado Adam, recortes, reducción,
salidas supervisadas y kernel nuevo fijo frente al histórico adaptativo. Son
factores que requerían controles, no causas probadas por ser distintos.

## Rectificaciones producidas por el intercambio

| Afirmación inicial | Evidencia/objeción | Estado final |
| --- | --- | --- |
| Bias como posible explicación dominante | Aporte aislado+2,38 frente a+37,17 personas medias en cabeza;−1,80 en conv4 mientras aumenta el conteo | No explica por sí solo la inflación |
| Incremento uniforme y sin relación con GT | Variación espacial apreciable; correlaciones delta–GT0,264/0,098 | Extendida y dependiente de escena; no constante ni independencia probada |
| Máximo delta conv4≈185 | JSON:155,43 | Error de Fable retirado |
| Picos sigma2 inalcanzables | Sin medición de capacidad representacional | Hipótesis espacial, no imposibilidad demostrada |
| Sigma histórico o cotas superiores desde marcas parciales | Faltan vecinos y exhaustividad | Recomendaciones retiradas para estas etiquetas |
| Clipping empeora razón de normas | Parcial4→1 frente a replay0,1 baja razón40→10 | Sentido aritmético corregido por Fable |
| Denominador común igualaría frecuencia e influencia | Residuos/features siguen determinando el vector | Garantía retirada |
| Suma de normas como≈60% de actualización | Ignora direcciones, cancelaciones y Adam | Estimación no válida |
| Ausencia de bugs | Sólo se inspeccionaron ciertas rutas | Ninguno identificó un bug causal en la ruta revisada |

El incremento de casi todos los píxeles examinados y la prueba de signo respaldan
investigar presión ascendente mal compensada. No miden cuánto corresponde a
etiquetas, kernel, ponderación, optimizador o geometría. El bias aislado tampoco
puede reemplazar el análisis de pesos/features.

Astra aclaró que “pérdidas contrarias” describía tendencias (parciales bajan,
negativos suben), sin proponer dos mecanismos. Fable mantuvo la hipótesis de una
elevación común. Las observaciones pueden coexistir: la atribución causal sigue
abierta. Un gradiente inicial favorable o desfavorable puede debilitar premisas
cuantitativas, pero no demuestra por sí solo la causalidad de200 pasos Adam.
Ninguna revisión convierte sigma2 o unilateralidad en métodos universalmente
inválidos; falta justificación para mantenerlos como receta de este ajuste.

## Protocolo mínimo propuesto, aún no ejecutado

1. **Diagnóstico sin actualizar.** En el mismo checkpoint original, medir vectores
   por fuente, signos, normas antes/después de clipping y productos internos.
   Distinguir diagnóstico inicial y trayectoria Adam. Comparar blancos/respuesta
   espacial sobre anotaciones completas, sin estimar vecinos con marcas parciales.
2. **Criterios previos.** Detectar inflación además de colapso, separar validación
   y test final y fijar de antemano evaluaciones intermedias/parada. Consultar
   repetidamente la cohorte test no la convierte en evaluación final ciega.
3. **Original-only150.** Misma ruta y orden de las150 exposiciones originales del
   mixto. Al quitar50 ejemplos nuevos cambia el reloj de Adam; declararlo. Un
   control de200 iguala actualizaciones, pero agrega exposiciones originales:
   complementa la comparación, no la sustituye.
4. **Una modificación por control.** Si original-only deteriora, revisar primero
   la ruta base. Si conserva, aislar objetivo y ponderación antes de abrir capas.
   El margen del ROI merece revisión, sin asumir que contexto visual equivale a
   anotación exhaustiva exterior.

No quedó validado un kernel nuevo, normalización óptima o pérdida superior. Para
entrenar con cotas superiores se requieren regiones exhaustivas y una convención
de frontera defendible. Estos controles acotados no requieren regenerar HDF5
masivos ni conectar el SSD.

## Corrección del informe anterior y trazabilidad

Hashes, unidades, actualizaciones, congelados y rechazo de pesos estaban
comprobados. Eso permitió detectar el problema, pero no justificaba presentar el
piloto como prueba suficiente de la adaptación ni priorizar más anotaciones antes
de comprobar la receta base. Se conserva el [informe previo](piloto-ajuste-dominio-2026-10-25.md)
y sus datos, corrigiendo ese alcance. Sus medias por supervisión no son curvas;
`first/last` agregados pueden pertenecer a ROIs distintos y no deben interpretarse
como trayectoria. Los dictámenes comparan también repeticiones del mismo ROI.

Las tres llamadas Fable terminaron con `is_error=false` y el modelo solicitado;
Astra completó sus tres dictámenes. El aviso “Selected model is at capacity”
referido por el usuario no apareció en esas respuestas. Sin conocer su interfaz
de origen no se lo atribuye a Codex o Claude, ni a GPU/disco local.

Se conservan SPEC006, expediente, seis dictámenes y suplemento, con commits
locales. No se publicaron datos ni cambios. Claude recibió código y resultados
tabulares, sin videos/fotos originales. Sus temporales y la copia privada aislada
de autenticación quedaron en `~/home.codex/tmp`; esa copia se retira al terminar,
sin modificar la autenticación original.
