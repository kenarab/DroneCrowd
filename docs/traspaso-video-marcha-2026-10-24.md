# Traspaso Nexo → DroneCrowd: videos de la marcha

Fecha24/10/2026. El autor solicita preparar la SPEC y continuará en una sesión
abierta por él en **/home/alephbet/home.agents/git/DroneCrowd**.

Leer primero AGENTS.md, [SPEC vigente](../sdd/001-video-breve/SPEC.md),
[plan](../sdd/001-video-breve/PLAN.md) y [entorno](entorno.md).
Esta nota describe el estado al entregar; no inicia otra sesión ni una ejecución.

## Estado y entradas

- Videos en subida a **/home/alephbet/home.agents/data/marchauniversitaria**.
  Al preparar el traspaso, la ruta aún no existía. Confirmar que terminó la subida;
  estabilidad de tamaño no basta como única evidencia.
- El autor identifica a Juan Roleri y fecha de entrega2/10/2024. La búsqueda
  anterior encontró otro video de abril; no usarlo como original identificado.
- Código local: master en8c52c73 antes de este traspaso, sin cambios ajenos.
- Pines históricos preservados en d0442e4 y5f73ed0; infraestructura b06f1c0.
- ml ya permite ejecutar el núcleo STNNet en CPU/GPU. No se instalaron paquetes.
- Sin checkpoints descargados, sin adaptador de video y sin inferencia real.
- Dataset histórico intacto; dataset es un enlace, no una carpeta de resultados.
- Disco disponible observado9,4GiB, inferior a la reserva20GiB para cargas grandes.
  Coordinar recuperación con Nexo; no borrar datos para sortear la reserva.

## Arranque seguro

Desde el checkout de DroneCrowd, antes de herramientas:
```sh
export TMPDIR=/home/alephbet/home.agents/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
export npm_config_cache="$TMPDIR/npm-cache"
export PIP_CACHE_DIR="$TMPDIR/pip-cache" UV_CACHE_DIR="$TMPDIR/uv-cache"
export PYTHONDONTWRITEBYTECODE=1
git status --short
bash ops/python-ml.sh scripts/check_environment.py
```

Ese diagnóstico sólo lee metadatos. Las comprobaciones de runtime/GPU y la suite
se ejecutan bajo scope/cuotas; no asumir que lanzar Python directamente limita RAM.
La nueva sesión puede necesitar permiso de acceso al dispositivo o bus de usuario,
según su sandbox; comprobar aislamiento antes de concluir que fallan drivers.
No cambiar HOME/CODEX_HOME ni usar temporales globales.

Desde el checkout de Nexo, el preflight existente acepta estimaciones explícitas:
```sh
python3 -m nexo.capacity preflight --scratch-gib 1 --retained-gib 0.125 --worker-ram-gib 1 --parallel 1
```
Son estimaciones iniciales a revisar; el comando es consultivo y no reserva recursos
ni autoriza el lanzamiento. Aplicar además límites efectivos de memoria.

## Decisiones ya tomadas y trabajo siguiente

Primero inferencia de conteo con STNNet y pesos entrenados; localización puede
evaluarse si existe la variante correspondiente. Seguimiento, STANet y entrenamiento
no forman parte de la primera prueba. La SPEC fija máximo30 fotogramas/lote1/120s,
salidas128MiB y un proceso acotado antes de ampliar nada.

pretrained_backbone=False evita torchvision/VGG16 cuando el checkpoint es completo.
Las dependencias de tracking son optativas en ese modo. No confundir pesos
aleatorios de las pruebas con un modelo entrenado. Elegir un checkpoint compatible,
cargar por nombres/formas y registrar el factor de densidad del entrenamiento.

El README de STNNet contiene el enlace público de modelos; sólo se comprobó
la página models.zip. Falta comprobar tamaño, contenido, procedencia y hash.
Si decodificación requiere paquetes nuevos, preparar entorno propio sin tocar ml.

El video original permanece local y de sólo lectura. Extraer únicamente la muestra
elegida, revisar escala/crops y guardar outputs excluidos de Git. No evaluar conteos
como precisión ni sumar fotogramas como personas únicas.

Cerrar cada incremento con commit local atómico, pruebas y documentación.
Push/publicación siguen sin autorización específica. La siguiente decisión humana
es continuar/ajustar/retirar el ejercicio y qué datos conservar: no ejecutar borrados
como efecto de una prueba aprobada o fallida.

## Evidencia previa

Dos pruebas sintéticas de conteo/localización pasaron en CPU y RTX3060 en2,377s.
Scope verificado High1,5GiB/Max2GiB/swap0,5GiB dentro del pool8GiB.
Revisión independiente previa: Codex /root/revision_archivos, modelo efectivo
desconocido, sin bloqueantes de arquitectura/infraestructura. No validó pesos,
video ni calidad científica. No repetir benchmarks sólo para confirmar esta nota.

Revisión independiente de este traspaso: Codex /root/revision_archivos, modelo
efectivo desconocido; sin bloqueantes. Comprobó ruta aún ausente, enlaces,
condiciones de subida y recursos y separación de diagnóstico/inferencia. Sólo
lectura: no ejecutó videos, ML/GPU, instalaciones o borrados. Validación del
coordinador:12 enlaces locales y git diff --check correctos.
