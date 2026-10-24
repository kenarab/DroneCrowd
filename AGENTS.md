# Trabajo con agentes en DroneCrowd

## Encargo y alcance

El autor trasladó su copia personal a este checkout el 24/10/2026 y autorizó
infraestructura de agentes, pruebas y commits locales atómicos. Primera tarea:
evaluar una prueba breve de conteo en otro video, conservando sus adaptaciones.
Responsable científico: DroneCrowd. Nexo coordina recursos y retención; no incorpora
el código del producto. Ver docs/proyecto.md y sdd/001-video-breve/SPEC.md.

- Trabajar en este repositorio; no modificar hermanos ni el venv compartido.
- Usar palabras descriptivas para conceptos y pasos; evitar claves de letras.
- Un solo coordinador por checkout. No lanzar agentes nuevos sin encargo explícito;
  una revisión independiente puede usar un agente existente disponible.
- Conservar los cambios previos del autor y registrar procedencia. Los pines en
  STNNet/requirements.txt son históricos, no una instalación validada actualmente.
- Preferir STNNet para la primera evaluación. STANet requiere otro trabajo de
  compatibilidad y no forma parte de esta primera prueba.

## Temporales y entornos

Antes de ejecutar herramientas:
```sh
export TMPDIR=/home/alephbet/home.agents/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
export npm_config_cache="$TMPDIR/npm-cache"
export PIP_CACHE_DIR="$TMPDIR/pip-cache" UV_CACHE_DIR="$TMPDIR/uv-cache"
export PYTHONDONTWRITEBYTECODE=1
```

No usar ni acceder a /tmp o /root; no cambiar HOME ni CODEX_HOME.
ops/python-ml.sh prepara además cachés propias de CUDA y PyTorch en ese temporal.
Usar el venv compartido ~/home.agents/venvs/ml sólo para ejecutar/inspeccionar:
no instalar, bajar versiones ni quitar paquetes allí. Si falta una dependencia,
preparar un entorno propio y documentar su vínculo con los paquetes compartidos.
No instalar paquetes globales ni cambiar drivers o CUDA del sistema.

## Recursos, datos y aceptación

- Diagnóstico inicial: importaciones y prueba sintética pequeña, sin entrenamiento.
- Antes de cargas grandes, aplicar preflight y cuotas Nexo: reserva mínima20GiB
  disco,2GiB RAM disponible y estimaciones de pico. No relajar límites automáticamente.
- GPU compartida: lote1, resolución pequeña y tiempo acotado; sin campañas,
  entrenamiento completo ni ejecución automática por abrir documentación.
- No invocar LLM adicionales, servicios pagados o modelos por API con este encargo.
- Dataset, videos, wheels, pesos, salidas y entornos quedan fuera de Git.
- dataset es un enlace local a datos del autor, no un permiso para borrarlos.
  No retirar ni trasladar ~/data/DroneCrowd sin decisión explícita del autor.
- Una muestra HDF5 o una extensión de archivo no prueban que todo sea regenerable.
  Conservar originales, anotaciones y resultados propios antes de decidir retención.
- Verificar procedencia y hash de pesos. Cargar checkpoints con el modo restringido
  de PyTorch; no habilitar deserialización arbitraria para sortear errores.
- Una prueba con tensores aleatorios sólo verifica ejecución, no calidad ni conteo.
  En video sin anotaciones no afirmar precisión cuantitativa.
- Mantener el video personal local; no subirlo ni servirlo automáticamente.

## Commits y comprobaciones

Cerrar cada cambio funcional validado con un commit local de propósito único.
Añadir rutas explícitas, revisar diff staged y comprobar lo relevante.
Identidad Codex: alejandro baranek <alejandrobaranek+codex@gmail.com>.
Separar recuperación de cambios del autor, infraestructura y adaptación del modelo.
La autorización de commits locales no publica cambios: push, tags y despliegue
requieren instrucción aplicable a este fork. Registrar límites y próximos pasos.
Revisor distinto del implementador; anotar herramienta/modelo cuando se conozca.
