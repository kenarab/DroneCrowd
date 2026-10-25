# Auditoría del diseño y discusión entre revisores

El autor cuestiona el diseño del entrenamiento tras el deterioro observado y
pide un informe completo para revisión por Astra y Fable, con discusión entre
ellos. Preparar un expediente verificable; no defender la receta anterior ni
confundir pruebas de implementación con validez del experimento.

## Alcance

- Congelar hechos, fuentes, hashes, código, datos, pérdidas, diferencias con el
  entrenamiento histórico, resultados completos y controles ausentes.
- Separar defectos comprobables de hipótesis causales y de resultados todavía
  no medidos. Mantener las etiquetas nuevas como provisionales.
- Primera lectura independiente, luego intercambio explícito de objeciones,
  respuestas y desacuerdos. Registrar identificador real de cada modelo.
- Astra está habilitado como `gpt-6-astra`. El autor indicó ejecutar Fable mediante
  `claude`; la versión local2.1.308 ofrece el alias `fable`. Se usa ese alias,
  sin sustitución ni fallback a otro modelo.
- Los revisores podrán consultar archivos locales y figuras privadas, sin subir
  datos, instalar dependencias ni lanzar entrenamiento o nuevos agentes.
- No cambiar pérdidas, etiquetas, pesos o hiperparámetros durante la revisión.
  El control original-only200 y los gradientes por fuente quedan como propuestas
  a justificar antes de una nueva ejecución. Se permite inspección de checkpoints
  en CPU, restringida y con hash, para completar hechos del expediente.

## Entregables y aceptación

Informe principal y anexo de discusión en `docs/`, evidencia privada bajo
`outputs/ajuste-dominio/revision-diseno/`. Matriz de hechos/hipótesis/controles,
explicación de qué permite concluir el piloto y protocolo mínimo para discriminar
fallas de receta base y fallas de la adaptación. Conservar resultados anteriores
y explicar las correcciones sin reescribirlos como éxitos.

Revisión independiente distinta del implementador; commits locales atómicos.
Temporales exclusivos `~/home.codex/tmp`; ninguna expansión de cuotas ni nuevas
cargas GPU por abrir este expediente. La imposibilidad de acceder a un modelo
debe quedar registrada, sin presentar una discusión incompleta como realizada.

## Estado

Expediente completo en `docs/expediente-revision-entrenamiento-2026-10-25.md`.
Astra (`gpt-6-astra`, agente sin historial heredado) y Fable completaron la primera
ronda independiente, guardada en `docs/revisiones-entrenamiento/`. La respuesta
JSON de Claude confirma modelo `claude-fable-5-1`, sin fallback. Fable usó solamente
Read/Glob/Grep, sin escritura, shell, subagentes, MCP ni persistencia de sesión.
Su configuración y copia privada de autenticación se aislaron en
`~/home.codex/tmp`, sin modificar originales. El JSON se conserva fuera de Git.

Después de ambas primeras lecturas, se preparó evidencia común adicional:
delta de bias y pesos por carga restringida CPU, diferencias por imagen histórica
y estadísticas espaciales de mapas ya guardados. Ninguna carga GPU ni entrenamiento.
Estos hechos se registran en el suplemento de ronda2 y se entregan a ambos para
corregir hipótesis o afirmaciones excesivas de sus dictámenes independientes.
Completadas tres rondas por modelo: independencia, objeciones y réplica final.
Fable confirmó `is_error=false` y `claude-fable-5-1` en las tres respuestas;
Astra conservó sus tres dictámenes en archivos separados. Las últimas réplicas
fueron simultáneas; el acta distingue rectificaciones explícitas de desacuerdos
anteriores. Astra revisó la fidelidad del acta final y la aprobó sin bloqueantes.

Entrega: `docs/revision-conjunta-entrenamiento-2026-10-25.md`, expediente completo,
seis dictámenes y suplemento medido. Se corrige el alcance del informe del piloto:
instrumentación comprobada no valida el diseño; los dos pesos siguen rechazados,
sin concluir inviabilidad de adaptar el modelo. Se propone diagnóstico común de
gradientes/blancos y control original-only150, con criterios previos;200 es otro
emparejamiento. Ninguno se ejecutó durante esta revisión. Pesos, etiquetas,
entrenador y particiones permanecen intactos.
