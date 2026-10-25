# Ampliar la revisión conjunta y mostrar predicciones por región

Encargo del autor,25/10/2026, antes de preparar la nueva selección: los fragmentos
son pocos, hay personas omitidas en foto-dispersas y las nueve marcas dudosas sí
son personas. Ampliar selección para revisión conjunta y mostrar qué predicen los
modelos sobre cada recorte. Conservar los originales y la versión2 congelada.

## Datos y revisión

Crear versión3 con las nueve candidatas reclasificadas por instrucción humana,
registrando esa procedencia. Esto no aprueba la exhaustividad ni los centros de
todas las etiquetas. Inspeccionar omisiones y distinguir adiciones del agente de
confirmaciones del autor. No marcar todo el manifiesto como user_reviewed.

Seleccionar12regiones adicionales, antes de obtener sus predicciones, cubriendo
personas dispersas con sombras, densidad intermedia/alta, oclusiones y superficies
urbanas. Fuentes a resolución nativa, recortes de256/512px; no aplicar restauración
generativa. Mostrar dimensiones originales y contexto. La nueva selección carece
de referencia exhaustiva: mostrar «pendiente», nunca cero personas.

Regla inicial para personas: anclar el centro de cabeza visible; si la presencia
se sostiene por cuerpo/sombra pero no se ubica la cabeza, registrar por separado
presencia y localización incierta. La sombra puede apoyar la identificación de
una persona, no debe contarse como otra persona ni utilizarse su centro como cabeza.
No inferir personas totalmente ocultas ni convertir signos visuales ambiguos en GT.

## Predicciones y aceptación

Reutilizar las predicciones existentes de las7regiones sólo después de verificar
hash de controles, fuente, geometría y checkpoint. Mostrar original y los dos
ajustes como rechazados, no como modelos validados. Su valor es integral de densidad
en el ROI con128px de contexto, no cantidad de detecciones. Una referencia parcial
no permite MAE ni cota superior completa.

Para12regiones nuevas, inferencia congelada del original, lote1 y escala nativa,
cuotas2GiB RAM/120s y preflight previo. No entrenar ni seleccionar regiones a partir
de la predicción. Si un ensayo falla, registrar el fallo sin relajar recursos.
Los ajustes rechazados no requieren inferencias nuevas para ampliar la revisión.

Vista privada versionada `revision/ampliada-v3/` con originales, puntos numerados,
predicciones por región, fuente/recorte/hash y estado provisional. Sin sobrescribir
la vista anterior ni modificar manifiestos utilizados por los pilotos. La portada
de revisión enlazará la versión nueva una vez comprobada. No publicar imágenes en
GitHub ni habilitar ejecución de HTML arbitrario en el explorador.

Revisión independiente de implementación y coherencia de datos, comprobación de
enlaces HTTPS autenticados y commits atómicos. La revisión conjunta con el autor
puede continuar sobre regiones por nombre y punto; no implica aprobación automática.

## Estado

Vista19regiones implementada y comprobada, sin diferencias de geometría al unir
predicciones. Inferencia de12regiones nuevas con el original:GPU3,56s, sin GT
inventado. Cinco pruebas del render aprobadas; el revisor independiente Codex
(modelo no identificado) verificó hashes, joins y artefactos privados. HTTPS
autenticado200. ImplementaciónLuna y ajustes del coordinador. Correcciones
posteriores de presencia y vista automática se mantienen en SPEC010/v4, sin
reescribir esta selección ni las versiones utilizadas por los pilotos.
