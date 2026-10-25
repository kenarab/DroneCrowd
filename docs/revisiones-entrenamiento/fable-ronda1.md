# Dictamen independiente de Fable, ronda 1

Modelo verificado: `claude-fable-5-1`; CLI `claude --model fable`, versión2.1.308.
Herramientas habilitadas: Read/Glob/Grep. Sin GPU, entrenamiento ni ediciones.
Respuesta original preservada en `outputs/ajuste-dominio/revision-diseno/fable-ronda1.json`.

## Dictamen Fable, primera ronda independiente (sección 12)

Leí el expediente, `scripts/train_domain_pilot.py`, `scripts/domain_targets.py`, `scripts/audit_hdf5.py`, `STNNet/stnmodel.py`, `STNNet/mytrain.py`, los dos `training.json`, las evaluaciones de retención por imagen, `summary.json`, `supervision-efectiva.json`, `anotaciones-v2.json` y las pruebas. No leí el dictamen de Astra ni revisiones previas. No ejecuté nada.

### 1. Conclusiones respaldadas y excesivas

**Respaldado.** Los dos pilotos empeoran el control histórico y los siete ROI; sólo cambian los tensores permitidos (hashes antes/después en `training.json`); masas, máscaras y mezcla 26/24/150 coinciden con el código. El rechazo de los pesos es correcto.

**Insuficiente, no excesivo.** El expediente trata el deterioro como un resultado a explicar con hipótesis abiertas. Los logs ya contienen evidencia que ordena esas hipótesis y que el informe no explotó:

- **La inflación es un desplazamiento casi uniforme por imagen, no una ganancia sobre personas.** Comparando `cuadrantes-piloto` y `cuadrantes-restantes` con `mixto-head-retencion`, el delta por imagen va de +26,9 (GT 60) a +47,8 (GT 252), media +37,2, sin relación con el GT (43 a 252). Una ganancia multiplicativa daría deltas de 10 a 130. En conv4 el delta va de +78 a +185. Esto es un hecho derivable de los JSON, no una hipótesis.
- **El modelo empeora sobre su propio replay mientras lo entrena.** `img049152-crop01` (blanco 9e-5) sube monótonamente 4,19 → 7,04 en cabeza; `img001152-crop00` (blanco 8,85) sube 14,49 → 19,99 (`collapse_guard`); `img014152-crop00` (blanco 58,2) sube 66,0 → 71,4. Todos los recortes replay suben, incluidos los que ya estaban por encima del blanco.
- **Los negativos de entrenamiento también suben:** edificio-b 7,87 → 10,17 y calle 8,91 → 12,12 en cabeza; 9,82 → 16,36 y 12,18 → 20,81 en conv4. Su pérdida crece paso a paso.
- **La pérdida parcial casi no baja:** dispersas 0,4164 → 0,4036 (paso 1 → 193) y grupo 0,7019 → 0,6860 en cabeza, mientras el conteo del contexto 512² sube 46,98 → 55,49 y 98,61 → 109,31. En conv4 baja 14 % y 10 % con conteos 46,98 → 81,71 y 104,47 → 145,14. El objetivo nuevo no se está ajustando; el modelo deriva.

**Excesivo en el informe:** `summary.json` → `loss_by_supervision.partial_positive.first/last` compara dispersas con grupo, dos ROI distintos; sugiere una pérdida que sube cuando en realidad baja levemente por ROI.

### 2. ¿Bug o diseño insuficiente?

No encontré un bug de implementación: geometría de recortes, unidades ×100, máscara `target>0`, clamp unilateral y recuento 26/24 coinciden con la SPEC. El problema es de diseño, en cuatro puntos verificables:

**(a) Escala del blanco incompatible con el modelo.** `domain_targets.py:19-40` usa sigma 2 a stride 2: pico 100/(2π·4) ≈ 3,98 unidades/píxel. La regla histórica (`audit_hdf5.py:201`, sigma = 0,1 × suma de distancias a 3 vecinos) aplicada a los dos centros supervisados de dispersas (distancias a medio tamaño ≈ 12,6/27/39 px) daría sigma ≈ 8 y pico ≈ 0,26. Estimación mía, no medida. La cabeza 1×1 sobre features suaves no puede producir picos 15 veces más agudos que los que aprendió; la forma más barata de reducir `max(100t−p,0)²` es subir todo.

**(b) Normalización por máscara más Adam anulan la mezcla 1:3.** La SPEC (`sdd/005-ajuste-dominio/SPEC.md:61-63`) afirma que normalizar por píxeles válidos "hace explícita la mezcla 1:3". Es al revés: cada píxel parcial pesa 65536/703 ≈ 93 veces un píxel replay. Los logs lo confirman: norma de gradiente 3,08–4,33 en parciales (recortada a 1), 0,007–0,70 en replay, 0,017–0,089 en negativos. En norma total aproximada, 26 pasos parciales ≈ 26, 150 replay ≈ 16, 24 negativos ≈ 0,6; los parciales aportan ~60 %. Con Adam la asimetría se amplifica: el segundo momento queda dominado por los pasos parciales y el momento de primer orden apunta siempre hacia arriba; los pasos replay, bilaterales y pequeños, no mueven la dirección. Estimación para el bias: gradiente parcial ≈ −2·media(residuo⁺) del orden de 0,3–0,4 tras recorte, siempre negativo (empuja arriba); replay ≈ ±0,02 con signo mixto; negativos ≈ +0,02. El bias sólo recibe presión neta ascendente.

**(c) Los parciales empujaban en la dirección contraria al objetivo desde el paso 1.** La tabla de la sección 7 muestra que el original ya sobreestima ambos ROI positivos de entrenamiento: dispersas 9,58 frente a 7–8, grupo 21,06 frente a 12–19. Una pérdida que sólo puede subir no podía acercar esos conteos a la referencia. El diseño confunde agudeza espacial con adaptación de conteo. Esto se lee en el propio expediente sin cálculo nuevo.

**(d) El borde de 32 px se aplica dentro del ROI aunque existe contexto.** `domain_targets.py:53-56` excluye 16 filas/columnas del mapa del ROI; `train_domain_pilot.py:82-92` ya embebe el ROI en un contexto de 128 px. Para positivos parciales, donde el fondo no se supervisa, no veo justificación: una cabeza confirmada en el borde sigue siendo confirmada y el contexto cubre el campo receptivo. Resultado: 19 marcas → 11 centros; dispersas 7 → 2. Eso elimina el 44 % del área del ROI y la mayoría de la supervisión positiva.

Dos observaciones menores: `train_domain_pilot.py:145-146` limita original-only a 5 pasos, así que el control propuesto en la sección 10 exige cambiar el entrenador; y el reporte no registra si corrió con `--allow-provisional-labels` (`train_domain_pilot.py:166`), aunque todas las etiquetas son `agent_provisional`.

### 3. Hipótesis que mejor explica dirección y magnitud

**Mecanismo matemático (apoyado por los logs):** la pérdida parcial unilateral, normalizada sobre ~700–1800 píxeles y con blancos inalcanzables en agudeza, produce un gradiente coherente y grande; Adam convierte eso en un desplazamiento por coordenada de ≈ lr por paso en dirección constante; el replay bilateral, ~10–100 veces más débil, no lo frena. El resultado es una elevación casi uniforme de den1. Predicción: delta por imagen ~constante; confirmado (+27 a +48). Predicción: negativos y replay suben aunque estén en gradientes; confirmado.

**Inferencia causal, con límite:** la identidad 5184·Δb de la sección 9 implica que un +37 uniforme exige Δb ≈ 7e-3 unidades si fuese sólo bias. Adam a 1e-5 en 200 pasos mueve cada coordenada típicamente ≤ 2e-3. Luego el desplazamiento uniforme viene también de los 128 pesos moviéndose coherentemente sobre la respuesta media de fondo de las features, no sólo del bias. Esto se verifica leyendo el checkpoint en CPU, sin GPU.

**Conv4:** 147 584 parámetros adicionales con el mismo empuje coherente dan más capacidad para elevar la respuesta de fondo; consistente con +78 a +185 y con que las imágenes 8 y 9 (GT 102 y 75) reciban los mayores incrementos en ambos pilotos. Esto último sugiere dependencia de la escena, no del número de personas.

**Refutaciones posibles:**
- Si un original-only con mismo Adam/LR produce un desplazamiento uniforme comparable, la causa sería la ruta base y mi hipótesis pierde peso. Espero lo contrario: con gradientes bilaterales pequeños y signos mixtos, el replay solo debería mover los conteos hacia sus blancos.
- Si la diferencia de mapas `.npy` (baseline-rois vs mixto-head-rois) muestra incrementos concentrados en los 11 centros y no en el fondo, la hipótesis de deriva uniforme se cae.
- Si el delta del checkpoint muestra Δb ≈ 7e-3, el bias explicaría todo y los pesos poco.

Hipótesis que no veo respaldadas: error de unidades (masas y conteos replay son coherentes en los logs), incompatibilidad Python/Torch (baseline 20,3 razonable y sin anomalías), "olvido catastrófico" como categoría separada (es el mismo mecanismo de dominación de gradiente).

### 4. Control mínimo antes de volver a ajustar

Orden por coste, todo antes de tocar fotos nuevas:

1. **CPU, sin entrenar:** cargar original y piloto cabeza; reportar Δbias, ‖Δweight‖, fracción de pesos con el mismo signo de cambio y 5184·Δb frente a +37. Restar los `.npy` de ROI baseline y cabeza; reportar media y desvío espacial del incremento dentro y fuera de los discos de los centros supervisados.
2. **CPU, un paso sin actualizar:** gradiente del bias y norma por fuente en el checkpoint original (uno por ROI nuevo y dos replay), antes y después del recorte. Confirma o refuta las magnitudes que estimé arriba.
3. **Aplicar la regla histórica de sigma a los puntos nuevos** y comparar pico y soporte con el blanco sigma 2 sobre la misma anotación.
4. **Original-only 200 pasos** con retención evaluada cada 50 pasos, levantando el tope de 5. Si deteriora, corregir receta base antes de cualquier otra cosa.
5. Sólo después, una modificación a la vez.

### 5. ¿Mantener pérdida unilateral y kernel?

El kernel sigma 2 fijo, no. Debe derivarse de la misma regla que el modelo aprendió o de un ajuste explícito del pico a la respuesta del original sobre cabezas originales de separación similar.

La pérdida unilateral por píxel, sólo bajo condiciones:
- normalización común (suma, o media sobre un denominador fijo) para que la mezcla 1:3 sea real en gradiente; alternativamente SGD sin normalización adaptativa, donde las magnitudes sí importan;
- supervisión descendente de peso comparable: negativos más grandes o una pérdida bilateral de intervalo sobre el conteo del ROI usando los `reference_interval` ya existentes, que penalice tanto sobreconteo como subconteo;
- borde aplicado al contexto, no al ROI, para parciales;
- retención cada N pasos con parada temprana e inflación como criterio, no sólo colapso a cero.

Sin esas condiciones, prefiero una pérdida de conteo por ROI con intervalo, que es lo único que las referencias provisionales realmente informan.

### 6. Correcciones al informe previo

- Reemplazar la tabla de "pérdida media" por trayectorias por ROI; el campo `first/last` de `summary.json` mezcla ROI.
- Añadir la tabla de deltas por imagen frente al original: muestra uniformidad y es el dato más informativo del piloto.
- Declarar que el original ya sobreestimaba los dos ROI positivos de entrenamiento y que la pérdida parcial no podía corregirlo.
- Corregir la afirmación de la SPEC sobre la mezcla 1:3: por ejemplo sí, por píxel y por gradiente no. Añadir la relación de normas de gradiente por fuente.
- Reformular la hipótesis del bias: la forma uniforme está confirmada; la atribución bias vs pesos está pendiente del delta del checkpoint.
- Dejar explícito que `mytrain.py:133,135,212` usa MSE suma con coeficiente 2/3000 por píxel, LR 1e-6 y beta1 0,5: con Adam el piloto mueve cada parámetro ~10 veces más por paso que la receta histórica. No lo tomo como causa demostrada, es un confusor para el original-only.

Bien comprobado y que no cuestiono: hashes, congelamiento, masa de blancos nuevos, semántica de máscara parcial, separación de fuentes, cálculo de MAE/RMSE/sesgo (`evaluate_calibration.py:16-22`) y conteo por ROI (`evaluate_roi_controls.py:19-32`).

Modelo y herramienta: Claude Fable 5.1 (`claude-fable-5-1`) vía Claude Code, sólo con Read/Glob/Grep sobre el repositorio.
