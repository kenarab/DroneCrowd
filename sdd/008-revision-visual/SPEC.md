# Acceso a las anotaciones para revisión humana

Estado: formalización de la vista preparada el25/10/2026; no se presenta esta
SPEC retrospectiva como un diseño anterior al render. El marco000 establece
especificaciones previas para los siguientes experimentos.

## Necesidad y alcance

El autor quiere comprobar las etiquetas propuestas y no encontraba DroneCrowd
en el explorador existente. Preparar una vista de sólo lectura con dos láminas
y siete recortes originales, ampliables sin herramientas especiales. Distinguir
marcas nominales, dudas y negativos provisionales. No confundir visualización
con aprobación humana ni edición de un manifiesto congelado.

## Contrato

`render_annotation_review.py` verifica manifiesto y fuentes por hash, crea láminas
privadas y un índice de procedencia. `prepare_annotation_browser.py` verifica esas
láminas y produce exclusivamente una carpeta `revision/` nueva con índice Markdown,
dos PNG y siete recortes, más hashes. No sobrescribir ni copiar fotos completas,
videos o pesos. `revision/` y `outputs/` están excluidos de Git.

La entrada declarativa `ops/file-browser-entry.json` permite docs, sdd y revision.
No registra ni publica nada por ejecutarse. El autor autorizó después una excepción
puntual para incorporar esa entrada al servicio de archivos y reiniciarlo. Se
conservan su autenticación y sus controles existentes; no habilitar JavaScript.
El editor offline de puntos queda separado, sin prometer edición en el navegador.

## Aceptación y evidencia

- Manifiesto v2 fijado por SHA-256;7regiones,22marcas y9dudas verificadas.
- Coordenadas locales alineadas; original a izquierda y marcas a derecha.
- Índice privado con hashes de fuentes y salidas; ninguna etiqueta pasa a
  `user_reviewed` por generar o mostrar las láminas.
- Entrada y enlaces comprobados con el renderizador del servicio; tras el alta,
  GET HTTPS autenticados200 de página y dos láminas. Pruebas existentes del
  explorador:12 aprobadas. El autor aún debe confirmar aceptación visual del enlace.
- Implementación de láminas: `gpt-6-luna`, ajustes/revisión del coordinador.
  Revisión adicional: agente Codex distinto, modelo no identificado, sin bloqueantes.
  Generador de carpeta y marco000 revisados por ese mismo revisor independiente.

La página y los archivos privados quedan fuera de GitHub. Se publica únicamente
el código, configuración declarativa y documentación para reproducir la vista.
Instrucciones en [revisión visual](../../docs/revision-visual-anotaciones-2026-10-25.md).
