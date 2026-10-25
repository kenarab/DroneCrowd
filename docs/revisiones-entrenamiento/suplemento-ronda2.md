# Evidencia común adicional para la segunda ronda

Este suplemento se preparó después de recibir ambos dictámenes independientes.
No se ejecutó entrenamiento ni inferencia nueva. Se inspeccionaron checkpoints
en CPU con hash/carga restringida y se calcularon diferencias de mapas y JSON ya
guardados. Las cifras siguientes son mediciones descriptivas posteriores al piloto;
no son controles causales ni un conjunto nuevo de validación.

## Cambios exactos en la cabeza final

Fuente: `outputs/ajuste-dominio/revision-diseno/checkpoint-deltas.json`.

| Modelo | Bias final | Delta de bias | Aporte uniforme crudo a1920×1080, antes de clamp | Pesos cabeza que aumentan / disminuyen |
| --- | ---: | ---: | ---: | ---: |
| Original | 0,00099304819 | 0 | 0 | 0 / 0 |
| Cabeza | 0,00145153329 | +0,00045848510 | +2,37679 personas | 115 / 13 |
| Conv4+cabeza | 0,00064648391 | −0,00034656428 | −1,79659 personas | 98 / 30 |

La columna uniforme es5184×delta_bias. Los cambios totales medidos incluyen los
pesos y, en el segundo piloto, las características modificadas por conv4. El
clamp puede alterar la relación entre masa cruda y conteo final. El bias aislado
no basta para explicar la inflación observada; en conv4 incluso desciende.

## Diferencias por imagen histórica respecto al original

Fuente: `outputs/ajuste-dominio/revision-diseno/deltas-descriptivos.json`.
Baseline combinado por índice de `cuadrantes-piloto` y `cuadrantes-restantes`,
comparado con cada retención posterior, sobre las mismas30 imágenes.

| Modelo | Delta medio | Desvío poblacional | Mínimo | Máximo | Correlación de Pearson delta vs GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cabeza | +37,16732 | 5,98082 | +26,85046 | +47,78254 | 0,26371 |
| Conv4+cabeza | +116,09150 | 24,41592 | +78,14049 | +155,42810 | 0,09782 |

Una regresión descriptiva delta=a+b×GT da a=33,77766,b=0,02477 en cabeza y
a=110,95850,b=0,03751 en conv4+cabeza. No se calculó significancia, independencia
estadística ni capacidad explicativa causal. La correlación baja no equivale a
ausencia demostrada de relación. En este conjunto, el máximo delta conv4 es155,43;
el aproximado185 del primer dictamen de Fable no coincide con estos JSON.

## Distribución espacial del incremento

Se restaron mapas crudos posteriores menos baseline, conservando idéntico ROI,
sin reescalar ni clamp por píxel. Media/desvío se expresan en unidades del modelo
por celda stride2; la integral de la diferencia dividida por100 coincide con el
delta de conteo de la tabla del piloto.

| ROI y variante | Media delta | Desvío delta | Desvío/media | Fracción de píxeles con delta positivo |
| --- | ---: | ---: | ---: | ---: |
| Dispersas, cabeza | 0,01329787 | 0,00766931 | 0,57673 | 100% |
| Fachada, cabeza | 0,01087937 | 0,00645634 | 0,59345 | 100% |
| Suelo, cabeza | 0,00477236 | 0,00116237 | 0,24356 | 100% |
| Dispersas, conv4+cabeza | 0,05374159 | 0,03625546 | 0,67463 | 100% |
| Fachada, conv4+cabeza | 0,03970557 | 0,02893414 | 0,72872 | 99,99695% |
| Suelo, conv4+cabeza | 0,01421512 | 0,00461079 | 0,32436 | 100% |

Hay un incremento muy extendido, pero no una constante espacial. Un delta de
conteo aproximadamente aditivo entre imágenes no demuestra uniformidad dentro
de cada mapa. No se midió aún la descomposición de pesos por canal y sus features,
los gradientes en un estado común ni el efecto efectivo de cada paso Adam.

## Preguntas para confrontar los dictámenes

1. ¿Qué afirmaciones del otro revisor quedan demostradas, son hipótesis útiles
   o exceden estas mediciones? Citar la afirmación concreta y responderla.
2. Astra deriva el signo de todos los gradientes de la cabeza en ejemplos
   parciales desde features no negativas. ¿Lo acepta Fable? ¿Qué permite concluir
   sobre SGD frente a Adam mezclado y sobre conv4?
3. Fable sugiere intervalos de conteo y aplicar la regla histórica de sigma a las
   marcas nuevas. ¿Bastan referencias provisionales/parciales para imponer cotas
   superiores o estimar vecinos? ¿Qué revisión o datos adicionales necesitarían?
4. ¿Cómo justificar la normalización y mezcla sin confundir frecuencia de ejemplos,
   pesos por píxel, norma escalar sumada y dirección/actualización Adam?
5. ¿Qué control mínimo priorizan:150 originales para exposiciones equivalentes,
   200 para pasos equivalentes, gradientes sin actualizar u otro? Distinguir qué
   pregunta responde cada uno y conservar restricciones de recursos.

No se cambió ningún peso, etiqueta o hiperparámetro durante la discusión.
