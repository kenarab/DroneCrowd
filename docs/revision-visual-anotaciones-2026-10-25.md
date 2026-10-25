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

## Ampliación y propuestas automáticas

La v2 descrita arriba se conserva. El autor confirmó las9candidatas como personas;
la v3 suma31marcas nominales anteriores/corregidas más una propuesta del agente
en la zona de sombra de foto-dispersas:32en total. No se aprobó exhaustividad ni
el centrado de todos los puntos. Se seleccionaron12recortes adicionales antes
de predecir, sin referencia todavía. Son19regiones para revisión, de las mismas
fuentes; no constituyen nuevos eventos ni un test ciego.

La v4 añade P1, persona indicada por el autor entre N3/N4, como presencia sin
centro de cabeza resuelto. No se inventa un punto para entrenar. Se conserva
la discrepancia del autor sobre test-peaton-fuente: el recorte fuente muestra
tres personas juntas en el suelo de la plaza y otra marca de borde, aún
provisionales. Cada PNG automático lleva el nombre dentro de la imagen.

Fuentes: foto amplia5280×3956, foto de plaza4000×3000, videos3840×2160.
Los recortes son nativos; ampliar la vista no recupera detalle ausente.
El área objetivo comprende calle/vereda/plaza y excluye terrazas, mediante una
máscara explícita aún pendiente. No se aplicó esa máscara silenciosamente.

`render_extended_review.py` genera `revision/ampliada-v3/`: referencias y conteos
del original y dos ajustes rechazados sobre los7recortes anteriores, y sólo el
original sobre los12nuevos. Verifica identidad de ROI, hashes, escala1 y densidad.
Los ajustes siguen rechazados; no se elige el que más se acerque a marcas parciales.

`render_automatic_review.py` genera `revision/automatico-v4/` a partir de esos mapas
originales. Reproduce el postprocesado `calc_denpt` histórico sin nuevo umbral:
candidatos en magenta, referencias parciales en verde, P1 en celeste y mapa de
densidad. La integral regional y la cantidad de máximos son cifras diferentes.
Foto-dispersas:9,575 de integral y10candidatos; fachada reservada:28,121 y47.
Los máximos no son etiquetas verdaderas ni probabilidades calibradas.

La inferencia original nueva se ejecutó en GPU:12regiones,3,56s, RSS1,027GiB y
VRAM161MiB. Extraer máximos sobre mapas ya guardados es CPU; no requiere volver a
inferir ni completar primero etiquetas. Ambas vistas son HTML estático sin JS,
con PNG ampliables y hashes; acceso HTTPS autenticado comprobado. Implementación
Luna y ajustes del coordinador, revisión independiente Codex sin modelo identificado.
Pruebas:5del render ampliado y4del automático, incluyendo paridad con la función
histórica extraída sin ejecutar el programa completo. Las fuentes personales siguen
fuera de Git. Entorno CPU: Python3.13.13, NumPy2.5.4, SciPy1.18.1, Pillow13.0.0;
las láminas iniciales usaron Matplotlib3.11.2.

[SPEC009](../sdd/009-ampliacion-anotaciones/SPEC.md) y
[SPEC010](../sdd/010-predicciones-automaticas/SPEC.md) documentan selección,
correcciones, postprocesado y aceptación. Los índices privados conservan las
evaluaciones y sus hashes; se mantuvieron las portadas anteriores del explorador.
