# Alta preparada para el dispatcher GPU

25/10/2026. El autor pidió preparar el alta. El proyecto entrega una propuesta
completa para registrar **DroneCrowd y tres perfiles iniciales**. No se aplicó al
registro compartido ni se enviaron solicitudes. La activación del coordinador y
la admisión real siguen siendo responsabilidad de infraestructura.

[SPEC015](../sdd/015-alta-dispatcher/SPEC.md) define esta entrega. La evaluación
completa tiene [44 cuadros y69 trabajos preparados](validacion-extendida-en-cola-2026-10-25.md).
La primera tanda sólo prueba el circuito con los mismos dos cuadros4K del inicio
de video0265: SR, CLIP-EBC RN50 y MPCount. Presupuesto: tres intentos totales,
máximo120s cada uno. Sin reentrenamiento ni reintentos automáticos.

## Paquete local para infraestructura

Después del commit, generar desde la raíz del proyecto:

```sh
export TMPDIR=/home/alephbet/home.codex/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR" PYTHONDONTWRITEBYTECODE=1
.venv/gpu-candidates/bin/python scripts/prepare_gpu_registration.py
```

El comando sólo valida en CPU y escribe
`outputs/alta-dispatcher-20261025/`. Requiere checkout rastreado limpio y verifica
que el entrypoint y lock estén en el commit. No sobrescribe un paquete existente.

| Archivo | Aplicación prevista por infraestructura |
| --- | --- |
| `project-registration.json` | Añadir esta entrada al array `projects` del registro, preservando las existentes |
| `resource-registration-change.json` | Retirar la entrada suplementaria anterior `DroneCrowd` al promoverla al registro principal |
| `gpu-profiles.json` | Añadir sus tres entradas al objeto `profiles` de la política, preservando controles y perfiles existentes |
| `requests.json` | Proyecto, perfil e ID idempotente para tres futuras solicitudes |
| `handoff.json` | Revisión exacta, plan congelado, presupuestos y estado preparado/no aplicado |
| `sha256.json` | Integridad de los cinco archivos anteriores |

El ID del proyecto es `dronecrowd`; la raíz registrada es `DroneCrowd`. Los
perfiles son `dronecrowd-smoke-sr-v1`, `dronecrowd-smoke-clip-ebc-v1` y
`dronecrowd-smoke-mpcount-v1`. Cada uno fija argv sin shell, un ID del plan y
`max_runs=1`. No aplicar el fragmento como reemplazo de una política completa.

Ya existe una entrada de recursos con ID `DroneCrowd`, sin suscripción principal.
La promoción a `dronecrowd` debe retirar esa entrada suplementaria en el mismo
cambio de configuración: ambas normalizan a la misma slice y dejarlas juntas
produce una colisión. Se conserva la unidad `nexo-projects-dronecrowd.slice`.
Infraestructura debe comprobar sus cuotas efectivas y las del pool antes de
enviar; si faltan, aplicar su procedimiento vigente y verificarlas en el kernel.

El paquete incluye rutas locales y queda fuera de Git; no contiene videos,
cuadros, pesos o credenciales. Código, lock y metodología están versionados.
Una vez registrados los perfiles, conservar HEAD y archivos rastreados hasta
terminar esa tanda: cualquier cambio exige revisar/actualizar la aprobación.

La interfaz real admite `submit --project dronecrowd --profile PERFIL
--request-id ID` y `status`; el ejecutable pertenece al dispatcher instalado.
Los IDs de `requests.json` son los del plan, no contadores nuevos por reintento.
Máximo32 solicitudes pendientes por proyecto: los69 trabajos no se envían juntos.
`waiting_profile` sólo acredita recepción; `running` y un resultado verificable
son evidencias diferentes. Esta preparación no produjo recibos de envío.

## Entorno y recursos

Entorno privado `.venv/gpu-candidates`, Python3.13.13; hereda Torch2.14.1 y
NumPy2.5.4 del entorno ML compartido mediante `.pth`, sin modificarlo. Se añadieron
offline desde caché einops0.8.0, ftfy6.2.0 y wcwidth0.2.14. Regex2026.9.29 ya
estaba disponible. El lock registra las versiones de metadata; el build de Torch
usado antes declara `+cu130`. Un cambio del entorno compartido puede invalidar
el lock: fallar y reconstruir/verificar el entorno antes de admitir inferencia.
Este lock describe versiones verificadas; no es un archivo con hashes de wheels.

Por perfil: RAM estimada1,5GiB, VRAM estimada1024MiB, scratch0,25GiB y salidas
0,25GiB. Infraestructura comprueba sus márgenes y reservas de recursos al admitir.
El wrapper exige estar en la unidad del dispatcher y reduce esa misma unidad a
High1,5GiB, Max2GiB y swap512MiB, verificando el kernel antes de cargar Torch.
No crea otra unidad; cancelarla termina también la inferencia. El límite interno
del hijo es hasta115s dentro del presupuesto total120s, con un hilo CPU.
El timeout externo del dispatcher contempla hasta10s adicionales para terminar
un proceso que no respondió a la señal de cancelación.

Se verifican plan, código, fuentes CLIP, input, checkpoint y versiones antes de
ejecutar. Un resultado completo se comprueba y no se repite; uno parcial se
conserva y requiere un intento nuevo revisado. Informes, densidades y logs quedan
en `outputs/`; staging/cachés sólo en `~/home.codex/tmp`. El hijo tiene límite de
64MiB por archivo, suficiente para una densidad4K float32 y para acotar logs.
Un timeout puede dejar estado inicial/parcial: no interpretarlo como completo.
Los errores previos al lanzamiento también conservan un recibo local único en
`outputs/dispatcher-validation-failures/`, porque el coordinador descarta stderr.

## Comprobaciones y límites

Pasaron11 pruebas CPU del wrapper,7 del contador y9 del preparador. Este último
usa el Python CPU con Pillow; el entorno de inferencia no necesita ese paquete.
La aceptación incluye verificación de las entradas
iniciales y del entorno real, y validación del paquete con el parser del dispatcher
usando configuración temporal. También se comprueba la promoción contra los
parsers de recursos y el plan de cuotas. Revisión independiente registrada en SPEC015.
No se ejecuta GPU para validar el alta. Las primeras ejecuciones reales deberán
comprobar scope, cancelación, tiempos y resultados antes de ampliar la tanda.

Los conteos son estimaciones por instante y encuadre. Sin referencia exhaustiva
no se informará precisión cuantitativa ni se sumarán cuadros como asistentes.
