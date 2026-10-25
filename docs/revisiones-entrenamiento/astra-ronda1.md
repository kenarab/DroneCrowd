# Dictamen independiente de Astra — primera ronda

Modelo: `gpt-6-astra`, confirmado por la configuración de lanzamiento del coordinador; `fork_turns=none`, sin sobreescritura de `reasoning_effort`. Herramientas: lectura local de código y JSON con `exec_command`; sin GPU, entrenamiento, instalaciones ni modificaciones durante la revisión. Este archivo se guarda posteriormente con `apply_patch`, por encargo explícito del coordinador, sin cambiar el contenido de fondo del dictamen. No se leyó el dictamen de otro revisor.

1. **Conclusiones respaldadas y excesivas.**

Los dos pesos experimentales deben rechazarse: incumplen la retención predefinida y aumentan la masa de densidad en todos los controles nuevos informados, incluidos negativos usados para entrenar. Esto demuestra el fracaso de esas ejecuciones, con esas particiones y objetivos.

No demuestra inviabilidad del ajuste de dominio, falta de capacidad del modelo ni que entrenar más necesariamente ayude. Tampoco distingue todavía una receta base dañina de una interacción dañina con los positivos nuevos.

El resultado original sobre treinta imágenes constituye un control útil de escala y funcionamiento; no valida todo el benchmark ni localización. Las referencias propuestas por agentes y revisadas por agentes no equivalen a anotaciones independientes certificadas.

La interpretación “el entrenamiento estuvo bien diseñado porque ejecutó la SPEC” sería excesiva. Su instrumentación permite rechazar los pesos, pero su diseño no permite explicar suficientemente el rechazo.

2. **Bug verificable versus diseño insuficiente.**

No encontré en la ruta nueva inspeccionada un bug de implementación que explique por sí solo el deterioro. Sí hay insuficiencias concretas de diseño:

- Se cambian simultáneamente optimizador, reducción de pérdida, recortes, etiquetas y alcance entrenable sin un control original-only de duración equivalente. El propio entrenador limita ese control a cinco pasos (`scripts/train_domain_pilot.py:145`).
- “Peso igual por ejemplo” no significa contribución comparable: las máscaras contienen 703/1805 píxeles positivos frente a 65.536 de replay. La media se calcula sobre cada máscara (`scripts/domain_targets.py:78`). Puede ser una decisión válida, pero requería medir sus efectos antes de interpretar la mezcla 1:3 como protección suficiente.
- El objetivo parcial obliga a alcanzar una forma espacial estrecha, aunque el conteo del ROI ya sea excesivo. No tiene una restricción superior. Esto alinea débilmente la optimización con la finalidad práctica de corregir sobreconteo.
- El guardia sólo rechaza que los tres conteos sean cero: `not any(row["count"] > 0...)`, línea129. Incluso dos colapsos individuales pasarían. Es coherente con la descripción “colapso total”, pero no es un control general de estabilidad.

El bug histórico de `if loss > den_loss` existe en `STNNet/mytrain.py:271`; la ruta nueva llama siempre a `optimizer.step()` después de controles de finitud (`train_domain_pilot.py:224`). No vincularía ese bug histórico con estos resultados nuevos ni con el origen de los pesos publicados.

3. **Hipótesis principal y mecanismo adicional comprobable.**

La explicación que priorizaría es una presión ascendente global originada en objetivos positivos estrechos y unilaterales, con corrección insuficiente por replay/negativos. Es una hipótesis causal; no una atribución ya medida.

Hay una conclusión matemática más fuerte que la del expediente: **en esta arquitectura, los positivos parciales producen gradientes no positivos en todos los pesos y en el bias de la cabeza final**, no sólo respecto de sus salidas.

En efecto, den1 recibe features después de ReLU (`STNNet/stnmodel.py:143`) y de multiplicación por un sigmoid positivo (`:363–364`). Por tanto, esas features \(h_j\) son no negativas. Para cada píxel, \(p=\sum_j w_jh_j+b\), y la pérdida parcial implica \(\partial L/\partial p\leq0\). Entonces:

\[
\frac{\partial L}{\partial w_j}
=\sum_i\frac{\partial L}{\partial p_i}h_{ij}\leq0,
\qquad
\frac{\partial L}{\partial b}\leq0.
\]

En el piloto de cabeza, con features congeladas, un paso de descenso simple sobre un parcial sólo puede aumentar o mantener las salidas de otras imágenes. Adam con memoria y mezcla de fuentes impide trasladar esto automáticamente al signo de cada actualización real. Para conv4 tampoco se deduce el signo de sus gradientes.

Los logs refuerzan la hipótesis sin demostrarla:

- `mixto-head/training.json`, pasos1/193: la pérdida de **la misma** foto-dispersas baja de0,41642 a0,40362, mientras su conteo de contexto sube de46,9785 a55,4943.
- Pasos5/197: foto-grupo baja de0,70188 a0,68597 mientras su contexto sube de98,6099 a109,3148.
- En conv4+cabeza, esos conteos llegan a81,7069 y145,1396; las pérdidas parciales también bajan.
- En negativos de entrenamiento ocurre lo contrario: la pérdida de edificio-b en cabeza aumenta de0,00007758 a0,00012519; la de calle vacía, de0,00012512 a0,00023898.

Estos son contextos completos, no los conteos ROI de la tabla. Las comparaciones se hacen entre repeticiones del mismo ejemplo y muestran que una mejora en el objetivo parcial puede coexistir con el deterioro práctico buscado evitar.

Las normas registradas también son sugerentes: los primeros parciales de cabeza tienen3,156 y4,112, recortadas a1; los primeros negativos,0,0174 y0,0269. No son mediciones en un estado común ni demuestran dominancia neta en Adam.

**Magnitud:** el factor de integración de5184 explica por qué un desplazamiento medio pequeño puede sumar decenas de personas. Pero no tenemos aún una descomposición del incremento entre bias, pesos y features. No afirmaría que el bias sea la causa principal.

La hipótesis perdería fuerza si original-only reprodujera un deterioro similar, o si mediciones en estado común mostraran suficiente oposición del replay y el análisis de pasos Adam no revelara deriva ascendente asociada a parciales.

4. **Control mínimo antes de otro ajuste con fotos.**

Primero haría diagnósticos sin entrenamiento:

- En el mismo checkpoint original, medir por ejemplo gradiente del bias y de pesos, normas antes/después de clipping y productos internos entre fuentes. Registrar también residuos y masa en máscaras.
- Contrastar los blancos histórico y fijo sobre los mismos puntos completos, con la misma geometría. Ver picos, anchuras, masa y pérdida del modelo original; no limitarse a comprobar que cada kernel suma1.
- Si se cargan los checkpoints finales, calcular el delta del bias y su contribución algebraica. Para el piloto de cabeza, separar además el término debido a cambios en pesos usando features congeladas.

Después, **original-only de150 pasos**, conservando orden y exposiciones originales del mixto, con evaluación idéntica y criterio predefinido. Este control elimina los50 pasos nuevos, por lo que cambia también la evolución temporal de Adam: declararlo. Un original-only de200 responde otra pregunta, sobre igual cantidad total de actualizaciones; no confundir ambos emparejamientos.

Si original-only falla, corregir primero esa ruta. Si conserva, estudiar la mezcla y el objetivo positivo antes de abrir más capas. No hace falta empezar por varias semillas ni por más entrenamientos grandes para detectar el problema fundamental.

5. **¿Mantener unilateralidad y sigma2?**

No los mantendría como receta establecida para corregir conteos.

Una pérdida unilateral puede servir como supervisión auxiliar para etiquetas parciales cuando existe un mecanismo independiente y suficientemente fuerte que limite densidad excesiva. Aquí no se comprobó ese equilibrio. Reemplazarla simplemente por MSE bilateral sobre anotación incompleta tampoco sería justificable: podría penalizar personas no anotadas.

Priorizaría pequeñas regiones exhaustivas revisadas por una persona, positivos y negativos, con separación de validación. La información de conteo y la incertidumbre de frontera deben corresponder al área realmente supervisada.

Sigma2 no es intrínsecamente un bug. Necesita justificación espacial y compatibilidad con la representación aprendida. Conservar masa por punto verifica una propiedad contable, no que el blanco sea apropiado. Exigir picos estrechos a una cabeza casi congelada puede favorecer elevar una respuesta extensa en vez de localizar mejor.

6. **Correcciones del informe y comprobaciones válidas.**

El expediente actual corrige muchas sobreinterpretaciones y distingue razonablemente hechos de hipótesis. Añadiría:

- La propiedad de signo de los gradientes de toda la cabeza final, con la salvedad de Adam.
- Comparaciones temporales de pérdidas sobre los mismos ROIs, ya disponibles.
- La distinción entre conteos de contexto del log y conteos ROI de evaluación.
- Que el problema observado incluye empeoramiento de negativos de entrenamiento; atribuirlo sólo a “poca generalización” omite evidencia.
- Que la propuesta original-only requiere modificar la restricción actual de cinco pasos.

Son valiosos los hashes, congelamiento verificado, actualización efectiva, división por fuente, exclusión de ambiguos y rechazo de pesos. Son controles de ejecución y trazabilidad; no sustituyen controles de identificación causal ni una validación del objetivo de entrenamiento.
