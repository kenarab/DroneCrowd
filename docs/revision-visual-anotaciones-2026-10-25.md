# Revisión visual de las anotaciones provisionales

Se prepararon dos láminas locales para que el autor pueda revisar los ejemplos.
Hay siete regiones pequeñas, no siete fotos completas anotadas. Los22 puntos
nominales y9 dudosos fueron propuestos por agentes; aún no tienen aprobación
humana externa. Los negativos son también provisionales.

La lámina de personas muestra el recorte sin marcas a la izquierda y los puntos
numerados a la derecha: verde para cabezas marcadas, naranja con prefijo D para
candidatos dudosos. La otra lámina muestra los cuatro negativos. El objetivo es
comprobar exhaustividad, centrado y dudas, no aprobar por apariencia el conteo del
modelo. El manifiesto v2 no se modifica por mostrar estas imágenes.

Archivos privados, excluidos de Git:

- `outputs/ajuste-dominio/revision-visual-usuario/personas.png`
- `outputs/ajuste-dominio/revision-visual-usuario/negativos.png`
- `outputs/ajuste-dominio/revision-visual-usuario/index.json`
- Editor offline existente: `outputs/ajuste-dominio/revision-v2/revisar-puntos.html`.

El editor permite exportar correcciones. Cualquier revisión debe generar una
versión nueva, conservar la anterior y registrar quién revisó cada región. Las
marcas dudosas no deben convertirse automáticamente en cabezas o fondo.

Para reproducir las láminas desde la raíz del repositorio, con las fuentes privadas
disponibles y un entorno que contenga Pillow y Matplotlib:

```sh
export TMPDIR=/home/alephbet/home.codex/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR" PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg MPLCONFIGDIR="$TMPDIR/dronecrowd-matplotlib"
export XDG_CACHE_HOME="$TMPDIR/dronecrowd-xdg-cache" OPENBLAS_NUM_THREADS=1
python3 scripts/render_annotation_review.py \
  --annotations outputs/ajuste-dominio/anotaciones-v2.json \
  --annotations-sha256 b7b37952b31d2efd3c3c3e6079587be88096f9b5c368bc56c46ccd2e0f6cfa1b \
  --output-dir outputs/ajuste-dominio/revision-visual-usuario
```

La salida debe ser nueva; el programa rechaza sobrescritura. Se verifican hashes
de manifiesto e imágenes, límites de tamaño y posición de los recortes. `index.json`
registra hashes de script, fuentes y láminas. El render se ejecutó en CPU y ambas
láminas se inspeccionaron visualmente. Implementación: agente `gpt-6-luna`; revisión
y ajustes visuales: coordinador Codex. Un segundo revisor independiente con Codex
(modelo no identificado) verificó los hashes, las siete regiones y ambas láminas;
aprobado sin bloqueantes.

Para preparar una carpeta curada compatible con un explorador autenticado:

```sh
python3 scripts/prepare_annotation_browser.py \
  --annotations outputs/ajuste-dominio/anotaciones-v2.json \
  --annotations-sha256 b7b37952b31d2efd3c3c3e6079587be88096f9b5c368bc56c46ccd2e0f6cfa1b \
  --sheets outputs/ajuste-dominio/revision-visual-usuario
```

Genera `revision/index.md`, dos láminas y siete recortes originales, con hashes.
La carpeta completa queda fuera de Git. No copia videos, pesos ni fotos completas.
El render estático no requiere un visualizador especial ni habilitar JavaScript;
el editor offline no está habilitado dentro del explorador. Preparar la carpeta
no registra ni publica automáticamente su contenido en ningún servicio.
