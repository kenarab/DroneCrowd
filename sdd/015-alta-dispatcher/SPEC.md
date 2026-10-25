# Alta preparada de DroneCrowd en el dispatcher GPU

Encargo del autor, 25/10/2026: «Prepará el alta del proyecto». Entregar la
entrada del registro y tres perfiles de prueba con argumentos fijos, listos para
revisión y aplicación por el responsable de la infraestructura. No modificar
registros compartidos, reiniciar servicios ni enviar solicitudes en esta fase.

## Contrato

Identidad `dronecrowd`, raíz `DroneCrowd`, etiqueta DroneCrowd. Preparar un entorno
privado `.venv/gpu-candidates`, sin modificar el entorno ML compartido. Registrar
versiones reales de Python y dependencias; no instalar otra copia de Torch.
Promover la entrada suplementaria de recursos existente `DroneCrowd` al registro
principal; proponer su retirada simultánea para no duplicar la slice normalizada.

Conservar el plan privado congelado de SPEC014 y sus hashes. Primera tanda:
SR, CLIP-EBC RN50 y MPCount sobre los dos cuadros del lote `0265-batch-000`.
Cada perfil fija revisión Git completa, entrypoint rastreado, lock rastreado,
plan/hash y un único ID de trabajo. Presupuesto máximo: un intento por perfil,
120 segundos cada uno, sin entrenamiento. Los otros 66 trabajos quedan preparados.

El wrapper valida entorno, código, entrada y pesos antes de cargar modelos;
exige un scope del dispatcher y reduce/verifica sus propios límites a RAM High
1,5 GiB, Max 2 GiB y swap 512 MiB. Un hilo, lote uno, VRAM estimada 1 GiB.
No crea una cola alternativa ni una unidad que sobreviva a la cancelación.
Salidas y logs acotados locales; conservar parciales y no reintentar fallos
automáticamente. Temporales exclusivamente en `~/home.codex/tmp`.

Generar el paquete concreto después del commit para evitar una referencia Git
circular. Rechazar checkout rastreado sucio. Incluir propuestas de registro y
perfiles, solicitudes idempotentes y hashes del paquete. No reemplazar políticas
enteras ni incluir datos, credenciales o fuentes de otros proyectos locales.

## Aceptación

Pruebas CPU de la validación del wrapper y construcción de perfiles. Comprobar
el paquete contra el parser real con registro/política temporales, sin escritura
en la configuración compartida y sin invocar GPU. Incluir registro de recursos y
plan de cuotas en esa comprobación. Revisión independiente antes
de dar por lista la entrega. Commits atómicos y publicación autorizada en el fork.

Registrar claramente: paquete preparado, alta todavía no aplicada, solicitudes
todavía no enviadas. El alta no demuestra activación del coordinador ni conteos.

## Cierre de preparación

Entorno privado creado offline y versiones verificadas. Pasaron11 pruebas del
wrapper y7 del contador con el entorno de inferencia, y9 del preparador con el
Python CPU original que incluye Pillow. El plan congelado93405… permanece intacto;
código, fuentes CLIP y archivos de los tres trabajos iniciales verificados.

Revisión independiente: herramienta Codex, modelo efectivo no identificado;
implementación del wrapper GPT-6 Luna, generador e integración Codex. El revisor
reprodujo con configuración temporal la cadena real registro→recursos→plan de
cuotas→política GPU: tres perfiles válidos, promoción exacta sin colisión y misma
slice High2/Max4/swap1GiB, sin aplicar cuotas o registros. Revisión favorable sin
bloqueantes para generar la entrega preparada/no aplicada.

Se corrigieron antes del cierre el campo de hash del plan, el registro persistente
de errores previos a inferencia y la comprobación de resultados completos contra
código, todos los cuadros/PTS y métricas finitas. Cancelación compatible por
inspección con la unidad heredada; su verificación efectiva queda para la primera
tanda real. No hubo GPU ni solicitudes en esta preparación. El paquete concreto
se genera después de este commit para fijar su revisión completa.
