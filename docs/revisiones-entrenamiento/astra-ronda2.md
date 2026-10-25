# Astra — segunda ronda

Modelo: `gpt-6-astra`. Herramientas: lectura local con `exec_command` de Fable ronda1 y del suplemento común, escritura de este dictamen con `apply_patch`. No ejecuté nuevos cálculos, inferencias ni entrenamiento. Las nuevas cifras se atribuyen a las mediciones del suplemento.

**Acuerdos y rectificaciones.** Acepto de Fable que las trayectorias de los mismos ROIs son más informativas que comparar pérdidas de ejemplos distintos, y que el empeoramiento de negativos entrenados debilita una explicación basada exclusivamente en mala generalización. Coincidimos en rechazar los pesos y en que falta el control original-only.

Cuestiono estas afirmaciones concretas:

- “La inflación es un desplazamiento casi uniforme [...] sin relación con el GT” y “la forma uniforme está confirmada”. Los deltas entre imágenes son aproximadamente aditivos, pero correlación 0,26371 no es ausencia demostrada de relación. El incremento espacial está muy extendido, con desvío/media de hasta0,73; no es constante. El máximo conv4 medido es155,43, no185. Reformularía como incremento generalizado dependiente de escena, sin ganancia proporcional al GT establecida.
- “El bias sólo recibe presión neta ascendente”. El bias final de conv4 disminuye: aporte uniforme −1,79659 personas. En cabeza aumenta, pero su aporte es sólo +2,37679 frente al delta medio +37,16732. Esto descarta explicar el incremento principalmente por el bias final aislado. No mide la historia de sus gradientes; sí contradice inferir actualizaciones permanentemente ascendentes. Mi primera ronda dejó abierta esa descomposición: ahora reduzco explícitamente la prioridad del bias como explicación dominante y mantengo pesos/features como términos pendientes de descomponer.
- “La cabeza [...] no puede producir picos 15 veces más agudos” y “blancos inalcanzables”. La sospecha de incompatibilidad espacial es razonable, pero no se ha medido una limitación de representación. Una cabeza1×1 combina features; su capacidad depende de esas features, no sólo de la anchura histórica de los blancos. Sigma estimado sobre etiquetas parciales tampoco reproduce necesariamente vecinos verdaderos.
- “El objetivo nuevo no se está ajustando”. Su pérdida sí disminuye sobre los mismos ejemplos. La conclusión correcta es que esa mejora coexiste con deterioro de conteo y controles; eso cuestiona la pertinencia del objetivo, no demuestra imposibilidad de optimizarlo.

**Signos, Adam y mezcla.** Mantengo la demostración de ronda1: features de den1 no negativas implican gradientes parciales no positivos para todos sus pesos y bias. Con cabeza congelada salvo capa final, un paso de SGD simple exclusivamente parcial, con tasa positiva, sólo puede elevar o mantener las salidas en cualquier entrada. Clipping global positivo conserva esos signos. SGD con momentum o Adam mezclado no heredan esa conclusión por paso. Tampoco se extiende a gradientes de conv4. Los115/13 y98/30 pesos que aumentan/disminuyen son compatibles con contribuciones contrapuestas; no demuestran su atribución por fuente.

“Normalización por máscara más Adam anulan la mezcla1:3” es demasiado fuerte. La frecuencia sigue siendo1:3; el peso de cada residual, su gradiente vectorial y el paso adaptativo son objetos diferentes. El factor65536/703 compara coeficientes por píxel, no influencia causal equivalente. Sumar normas y asignar “~60%” no obtiene una fracción de actualización: ignora alineaciones, cancelaciones y momentos. Un denominador común tampoco garantiza mezcla1:3 “real en gradiente”; SGD elimina la adaptación por coordenada, pero no las diferencias entre residuos y features. LR diez veces mayor tampoco prueba desplazamientos reales diez veces mayores entre recetas distintas.

Justificaría la mezcla declarando primero qué riesgo se quiere ponderar —por región, imagen, persona o fuente— y luego midiendo vectores por fuente en estado común, clipping y efectos de Adam. La frecuencia sola no justifica protección de retención.

**Etiquetas, intervalos y bordes.** No usaría `reference_interval` como cota superior entrenable sin revisión exhaustiva: los candidatos listados no garantizan que no existan más personas. Una supervisión parcial segura puede aportar límites inferiores, pero no automáticamente superiores. Además, masa dentro del ROI y centros contenidos difieren por colas gaussianas. Harían falta regiones revisadas exhaustivamente por una persona y una convención explícita de frontera.

Tampoco aplicaría directamente vecinos históricos a marcas incompletas: omitir vecinos sesga sus distancias. Compararía kernels sobre anotaciones originales completas, y después sobre regiones nuevas completas si se obtienen. No hay obligación científica de copiar la receta histórica; sí de justificar cualquier cambio.

Acepto revisar la pérdida de centros por el borde del ROI. Pero disponer de contexto visual no vuelve exhaustiva la anotación exterior ni resuelve automáticamente truncación de kernels. Para parciales, un margen fijo de32 merece justificación individual; no eliminarlo mecánicamente sin definir la supervisión de frontera.

**Orden mínimo propuesto, sin ejecutar.**

1. Incorporar las rectificaciones anteriores y conservar congeladas particiones y métricas. Las mediciones de checkpoints/mapas ya hechas no deben repetirse como si faltaran.
2. Medir gradientes por fuente en un estado común y comparar blancos sobre las mismas anotaciones completas. Esto evalúa desbalance e incompatibilidad espacial; no estima por sí solo causalidad del entrenamiento completo.
3. Original-only150, con las mismas exposiciones y orden del replay del mixto. Es mi primera elección para preguntar qué ocurre al retirar los ejemplos nuevos conservando datos originales. Reconocer que cambia el reloj de Adam. Original-only200 iguala actualizaciones, pero añade exposiciones originales: control complementario, no equivalente.
4. Fijar antes criterios de inflación/retención y validación para puntos intermedios; no escoger el mejor checkpoint consultando repetidamente el test. Si falla original-only, revisar la ruta base; si conserva, aislar objetivo/ponderación antes de ampliar capacidad.

La evidencia nueva fortalece la descripción de deriva extendida, pero sigue sin identificar cuánto corresponde a etiquetas, kernel, ponderación, optimizador o geometría. No hay aún motivo para presentar una de esas causas como demostrada.
