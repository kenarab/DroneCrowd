# Fable — segunda ronda

Modelo verificado: `claude-fable-5-1`; consulta nueva por CLI con dictámenes anteriores y expediente incluidos, sin herramientas ni persistencia.

**Segunda ronda de Fable** (`claude-fable-5-1`, sin herramientas ni ejecuciones; todo se deriva del material entregado).

## 1. Correcciones a mi dictamen anterior

- **Máximo conv4.** Escribí +78 a +185. Los JSON dan +78,14 a +155,43. Error mío de lectura; lo retiro.
- **Bias.** Predije que un +37 puramente por bias exigiría Δb ≈ 7e‑3 y que Adam no llega. Medido: Δb = +4,6e‑4 en cabeza (2,4 personas de 37) y −3,5e‑4 en conv4+cabeza. La inferencia se confirma: el bias no explica la inflación. En conv4 incluso baja, lo que muestra que el signo del gradiente parcial no determina el signo de la actualización Adam mezclada.
- **"Desplazamiento casi uniforme".** Fue impreciso. Lo que está medido es: aditivo entre imágenes (correlación delta‑GT 0,26 y 0,10, pendiente 0,025 y 0,038) y ubicuo dentro de cada mapa (100 % de píxeles positivos), pero con desvío/media 0,24–0,73. La formulación correcta es "elevación ubicua modulada por las features", no "constante". Esto es exactamente lo que predice Δp = Σ Δw_j h_j + Δb con 115 de 128 pesos subiendo sobre h ≥ 0. Evidencia descriptiva de 30 imágenes y tres ROI, sin independencia ni significancia.
- **Sigma histórico ≈ 8.** Era una estimación sin base aplicable: la regla necesita cuatro centros ocupados y dispersas tiene dos supervisados. Lo reformulo abajo.

## 2. Afirmaciones de Astra

**Acepto como prueba matemática** la derivación del signo: den1 recibe ReLU × sigmoid, luego h_j ≥ 0; con ∂L/∂p ≤ 0 se sigue ∂L/∂w_j ≤ 0 y ∂L/∂b ≤ 0 para cada paso parcial. Es más fuerte que lo que yo escribí (sólo sobre salidas). Consecuencia exacta para SGD puro en el piloto cabeza: un paso parcial produce Δw ≥ 0 y Δb ≥ 0, y por tanto Δp ≥ 0 en todo píxel de toda imagen. Para Adam mezclado sólo se deduce que los parciales aportan componentes de signo fijo al primer momento; el signo final depende de m/√v por coordenada. Para conv4 no se deduce nada: el gradiente pasa por pesos de signo arbitrario. El 115/13 medido y el 100 % de píxeles positivos son consistentes con dominancia de los parciales, no la demuestran.

**Acepto** que MSE bilateral sobre anotación incompleta no es justificable, que el guardia sólo detecta colapso total y que el `if loss > den_loss` histórico no se vincula a estos resultados.

**Cuestiono tres puntos:**

1. Astra presenta el aumento de pérdida en negativos como "lo contrario" de la baja en parciales. No son fenómenos opuestos sino el mismo: una elevación global baja levemente `max(100t−p,0)²` donde hay cabezas y sube `p²` donde no. Separarlos sugiere dos mecanismos cuando la evidencia apunta a uno.
2. Astra dice que la hipótesis perdería fuerza "si mediciones en estado común mostraran suficiente oposición del replay". Por la propia prueba de signo, el replay en la cabeza sólo puede oponerse donde su residuo es negativo, y su norma registrada es 0,007–0,70 frente a 1 tras recorte. La oposición posible está acotada por construcción; la medición en estado común servirá para cuantificar, no para refutar.
3. La observación sobre el guardia (dos colapsos individuales pasarían) es correcta pero irrelevante para el fallo observado, que fue inflación. Un criterio de inflación sobre los replay ya logueados era más urgente.

## 3. Intervalos y regla de sigma (pregunta 3)

Retiro la pérdida de intervalo como objetivo de entrenamiento con las referencias actuales. Son propuestas de agentes, parciales, y esta discusión no autoriza derivar cotas superiores de ellas. Quedan como diagnóstico descriptivo. Para imponer cotas haría falta anotación exhaustiva humana de regiones pequeñas, con incertidumbre de frontera declarada, como propone Astra.

La regla histórica de sigma no se puede aplicar a las marcas nuevas: faltan vecinos. Lo que sí se puede hacer en CPU y sin etiquetas nuevas: medir en el den original, sobre cabezas del `train_data` original con separaciones de 10–40 px, el pico y anchura de la respuesta real, y compararlos con el pico 3,98 del kernel sigma 2. Eso aisla la compatibilidad del blanco con la representación aprendida sin tocar personas.

## 4. Normalización y mezcla (pregunta 4)

Son cuatro cantidades distintas y el expediente las confunde en "1:3":

- **Frecuencia de ejemplos:** 26 parciales, 24 negativos, 150 replay.
- **Peso por píxel:** media sobre máscara → un píxel parcial pesa ≈ 93 veces (65536/703) uno de replay. Esto es aritmética, no hipótesis.
- **Norma sumada:** parciales 3–4 pre‑recorte, replay 0,007–0,70, negativos 0,02–0,09. El recorte a norma 1 iguala sólo a los grandes, así que empeora la razón relativa frente al replay.
- **Actualización Adam:** por coordenada ≈ lr·m/√v; con componentes de signo fijo dominando m, la dirección se fija y la magnitud se acota a ~lr por paso, lo que explica deltas pequeños y coherentes en 128 pesos.

Justificar una mezcla exige medir las cuatro en el checkpoint original antes de entrenar, y fijar la normalización con denominador común (suma, o media sobre área fija) para que frecuencia y gradiente coincidan.

## 5. Control mínimo, en orden

1. **CPU, un paso sin actualizar, checkpoint original:** gradiente de bias y pesos por fuente, normas pre/post recorte, productos internos entre fuentes. Responde: ¿cuánto domina cada fuente y cuánto puede oponerse el replay? Coste mínimo.
2. **CPU, descomposición del delta de cabeza:** Σ Δw_j · media(h_j) por ROI frente a Δb. Responde: ¿pesos o bias? Ya está medido parcialmente; falta el término de pesos.
3. **Original‑only 150 pasos**, misma ruta, retención cada 50, levantando el tope de 5 pasos. Responde: ¿la receta base conserva con exposiciones iguales? Prefiero 150 porque la pregunta es sobre la receta; la historia Adam ya difiere al quitar 50 pasos y debe declararse. 200 sólo si 150 conserva, para igualar actualizaciones.
4. Después, una modificación a la vez: normalización, kernel, borde.

Todo dentro de los límites actuales de RAM, lote 1 y timeout.

## Consensos y desacuerdos

Consenso: rechazo de pesos correcto, sin bug de implementación, diseño insuficiente, bias descartado como causa principal, diagnósticos sin entrenar antes de original‑only, unilateral y sigma 2 no mantenibles como receta. Desacuerdo residual: Astra separa negativos y parciales como fenómenos distintos; yo los leo como un único mecanismo de elevación ubicua. Falta la medición en estado común para zanjarlo.
