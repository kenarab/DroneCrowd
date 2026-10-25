# Fable — réplica final

Modelo verificado: `claude-fable-5-1`; CLI sin herramientas, con dictámenes y suplemento explícitos.

Entrego la réplica directamente, sin tocar herramientas ni archivos, como se pidió.

# Fable, ronda 3

Modelo `claude-fable-5-1`, consulta nueva sin herramientas. Todo lo que sigue se deriva del material entregado.

## 1. Revisión literal de mis afirmaciones de ronda 2

**Recorte a norma 1 "empeora la razón relativa frente al replay".** Falso tal como lo escribí. Los parciales pasan de 3 a 4 hasta 1; el replay, por debajo de 1, no cambia. La razón parcial/replay baja de 4 a 500 veces hasta 1,4 a 140 veces. El recorte reduce el desbalance, no lo empeora. Lo que sostengo es más débil: no lo elimina, y al escalar siempre al mismo tipo de fuente cambia su aporte a m y v de forma no proporcional. Esto último no está medido.

**Denominador común "para que frecuencia y gradiente coincidan".** Error. Un denominador común iguala el coeficiente por píxel entre fuentes. El vector de gradiente sigue dependiendo del residuo, del número de píxeles con residuo y de las features activas. Con suma, 703 residuos positivos contra 65.536 bilaterales podrían invertir la dominancia. Ninguna normalización hace que la frecuencia 26/24/150 describa la influencia. Astra tiene razón: la frecuencia es sólo una de cuatro magnitudes y las otras hay que medirlas.

**"La medición en estado común servirá para cuantificar, no para refutar".** Lo retiro. La prueba de signo acota la dirección de los parciales en la cabeza, no la magnitud del replay. Si la suma de gradientes de replay sobre sus 150 exposiciones, en el checkpoint original, supera y se opone coordenada a coordenada a la suma de los 26 parciales, la hipótesis de dominancia queda refutada. Además el bias de conv4 bajó, lo que ya muestra que las actualizaciones no heredan el signo. Mi dato de normas 0,007 a 0,70 proviene de estados distintos y tras recorte; no sirve para fijar esa cota.

**"Sin bug de implementación".** Debe decir: no se encontró bug en la ruta inspeccionada. Quedan fuera la carga de recortes replay, la auditoría HDF5 completa, la ejecución o no con `--allow-provisional-labels` y cualquier no determinismo de GPU. Dos lecturas coincidentes reducen la probabilidad, no la anulan.

**Cuatro objetos que confundí.** La norma es un escalar global tras la reducción y el recorte. El vector tiene dirección y signo por coordenada, y es lo que entra en el momento. La contribución al momento depende del orden y de la frecuencia, con β1 y β2 pesando la historia. La actualización Adam es lr·m̂/√v̂ por coordenada, acotada en magnitud y con signo heredado de m, no del gradiente del paso. Mi "~60 %" sumaba normas y lo presentaba como fracción de actualización; eso no es una medición válida.

## 2. Objeciones de Astra que acepto

- Reformular la inflación como incremento generalizado dependiente de escena, sin ganancia proporcional al GT demostrada. Ya corregido el 185 y el bias.
- "Picos 15 veces más agudos" y "blancos inalcanzables" son hipótesis sin medición de capacidad. La cabeza 1×1 depende de sus features. La prueba propuesta es medir pico y anchura de la respuesta del den original sobre cabezas originales, comparados con el 3,98 del kernel.
- "El objetivo no se está ajustando" fue incorrecto: la pérdida parcial baja sobre los mismos ROI. La formulación correcta es que esa mejora coexiste con el deterioro de conteo y de negativos entrenados.
- Nada de cotas superiores desde `reference_interval`, ni regla histórica de sigma sobre marcas incompletas. Ya retirado.
- El borde de 32 px no se elimina mecánicamente. Mantengo que dentro de un ROI con positivos parciales excluye centros confirmados. La convención mínima que propongo discutir es renderizar el blanco en el contexto de 128 px y recortar al ROI, lo que resuelve la truncación de kernels sin declarar exhaustivo el exterior.

## 3. Desacuerdos que mantengo

**Un mecanismo o dos.** Sigo leyendo la subida de pérdida en negativos y la bajada en parciales como una sola elevación de salida, no como fenómenos contrarios. Es hipótesis, no hecho. Se zanja con el producto interno entre el gradiente parcial y el negativo en estado común: si son antiparalelos y la suma neta apunta hacia arriba, basta un mecanismo.

**Guardia.** Su observación sobre dos colapsos es correcta pero no explica este fallo. Un criterio de inflación sobre replay logueado era el control que faltaba.

## 4. Protocolo mínimo compartido

1. Gradientes por fuente en el checkpoint original, CPU, sin actualizar: vector, norma pre y post recorte, productos internos, y sumas acumuladas con las exposiciones reales 26/24/150. Responde dominancia y oposición posible del replay.
2. Descomposición del delta de cabeza: Σ Δw_j·media(h_j) por ROI frente a Δb. Falta sólo el término de pesos.
3. Comparación de blancos sigma 2 contra regla histórica sobre anotaciones originales completas, y respuesta real del den original.
4. Original-only 150 pasos, mismo orden y exposiciones del replay, retención cada 50, criterios de inflación y retención fijados antes, levantando el tope de 5 pasos y declarando el cambio de reloj Adam. Original-only 200 como complemento posterior.
5. Si 150 falla, revisar la receta base; si conserva, aislar objetivo y ponderación antes de abrir capas.

Sin entrenamiento con fotos nuevas, sin cotas superiores desde etiquetas parciales, sin cambios en pesos ni particiones.

**No resuelto:** atribución entre etiquetas, kernel, ponderación, optimizador y geometría; si existe incompatibilidad espacial real; convención de frontera para parciales.
