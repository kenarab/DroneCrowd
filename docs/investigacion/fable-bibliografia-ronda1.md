# Dictamen bibliográfico de Fable: primera lectura

Modelo real: `claude-fable-5-1`, confirmado en JSON. CLI `claude --model fable`,
25/10/2026. Revisión sin herramientas del paquete de fuentes de Astra y marco SDD;
no verificación web independiente ni ejecución de modelos. Texto original conservado,
incluyendo afirmaciones que se contrastan en la discusión posterior.

Entrego el dictamen directamente, sin herramientas, sobre el paquete y el marco tal como fueron pegados.

## 1. Hechos del paquete e inferencias propias

**Hechos que tomo del paquete** (verificados por Astra o el coordinador, no por mí): STNNet MAE 20,30, RMSE 23,82 y sesgo −3,22 sobre 30 imágenes ya vistas. Dos pilotos rechazados. Siete ROI, 22 marcas, 9 dudosas, 11 centros efectivos. SR y CLIP-EBC con código MIT y artefactos listados. P2PNet con licencia académica. Pesos de MovingDroneCrowd bajo CC-BY-NC-4.0. Repositorio de CLIP-EBC archivado.

**Inferencias mías**, que deben leerse como hipótesis:

- **El sesgo negativo agregado y el fallo en edificios son compatibles solo si hay cancelación.** Subconteo en zonas densas y masa falsa en fachadas se compensan en la integral. Eso vuelve al MAE global casi inútil como criterio de selección hasta que exista una tabla de error por estrato y una medida de masa en negativos.
- **La referencia STNNet no es una medida de test.** Las 30 imágenes fueron vistas en entrenamiento. Cualquier candidato preentrenado en DroneCrowd puede haber visto las mismas imágenes o no, según el split. Comparar sin saberlo mezcla retención con generalización.
- **Los 11 centros efectivos no sirven para entrenar nada**, ni STNNet ni candidatos. El paquete lo dice y lo suscribo. Agrego que tampoco sirven para medir recall de ningún modelo, incluido el de puntos.

## 2. Tres tareas que el marco debe mantener separadas

- **Conteo visible por cuadro**: personas visibles en una región e instante. Es la única tarea que la evidencia local actual permite empezar a evaluar. SR, CLIP-EBC, MPCount, DM-Count, CAN y STNNet pertenecen aquí.
- **Individuos únicos en video**: requiere identidad temporal y anotación de entradas y salidas. MovingDroneCrowd pertenece aquí. No existe anotación temporal local. Queda fuera hasta que haya SPEC propia.
- **Asistencia total**: requiere cobertura espacial completa, cobertura temporal y modelo de duplicación. Ninguna fuente del paquete la aborda. Nada de lo que se ejecute en los próximos ensayos permite enunciar una cifra de asistencia.

## 3. Qué evaluar primero y por qué

El orden propuesto por el coordinador es correcto en lo esencial. Lo reformulo con una razón explícita en cada paso.

1. **Cerrar el control original-only 150 y el diagnóstico de gradientes sin actualizar.** Sin eso no se sabe si el deterioro de los pilotos vino de las fotos nuevas, de la receta o de la máscara. Un candidato nuevo no resuelve esa ambigüedad; la hereda.
2. **Antes de cualquier candidato, producir la tabla de error por estrato para STNNet** sobre las 30 imágenes y sobre las 7 ROI, separando marcas confirmadas, dudosas y zonas no anotadas. Si la cancelación que infiero existe, debe aparecer aquí. Es un análisis de salidas ya existentes, cabe en el plazo y no toca el entorno.
3. **Recién después, inferencia congelada de un candidato a la vez**, lote 1, sin ajuste, sobre los mismos cuadros y el mismo recorte. Medir pico de RAM, VRAM, tiempo y masa en negativos antes de mirar MAE.

## 4. Tres candidatos, en orden, y por qué no más

**Primero: SR con pesos DroneCrowd.** Es el único candidato cuyo dominio de preentrenamiento coincide con el de la referencia. Eso permite una comparación interpretable con STNNet sobre los cuadros originales: misma fuente de datos, dos modelos, una diferencia de método. La regularización temporal es solo de entrenamiento, así que la inferencia es por imagen y encaja con la tarea 1. Riesgo principal: no sé qué split usó ni si los 30 cuadros caen en su entrenamiento. Sin esa respuesta el resultado no se puede clasificar como retención ni como generalización.

**Segundo: CLIP-EBC con pesos NWPU.** Motivo distinto: NWPU incluye negativos explícitos. Es la familia con más probabilidad de haber aprendido a no disparar en fachadas. Riesgos: backbone CLIP con sesgo semántico hacia caras y figuras humanas en carteles, pancartas y estatuas, muy frecuentes en una marcha; repositorio archivado, es decir, sin corrección de incompatibilidades, obliga a fijar commit y entorno propio; carga de ViT-B/16 y dos variantes NWPU que exigen fijar una regla de elección antes de ver resultados. La reserva de 2 GiB de RAM de proceso es plausible que no alcance para cargar el modelo; es una verificación, no una suposición.

**Tercero, condicional: MPCount.** Su protocolo de generalización desde un solo dominio es el que más se parece a nuestro problema real. Pero sus pesos públicos vienen de vistas de vigilancia, no aéreas, y la versión determinista debe fijarse. Solo entra si los dos primeros muestran que el fallo dominante es cambio de dominio y no escala.

**Excluidos y por qué.** P2PNet por licencia académica no permisiva y PyTorch 1.5. MovingDroneCrowd por tarea distinta y pesos NC. DM-Count queda como pérdida de referencia, no como modelo: su aporte se mide cambiando un factor en un control, y hoy no hay anotaciones completas para que esa comparación signifique algo. CAN, DecideNet, URM, T2ICount y DCA-MoE: contexto bibliográfico, sin artefacto reproducible en 12 GiB o sin pertinencia de tarea.

## 5. Condiciones de descarte y reproducibilidad

Un ensayo breve se registra como válido solo si cumple todo esto. Si no, se descarta sin inferir nada sobre el modelo.

- Commit fijado, hash de pesos **recalculado localmente**, no copiado del release.
- Entorno propio con versión de framework declarada. Si el modelo exige una versión incompatible con el driver compartido, se registra el obstáculo y no se fuerza.
- Lote 1, resolución de entrada documentada y una sola política de tiles para todos los modelos del ensayo. Comparar nativo contra tiles es otro experimento.
- Normalización del conteo documentada: factor de reducción del mapa, redondeo, umbral de confianza si es de puntos.
- Pico de RAM y VRAM medido. Si supera la cuota, el modelo se declara **no ejecutable en este entorno**, no inferior.
- Regla de rechazo escrita antes de correr: por ejemplo, descartar si la masa predicha en negativos arquitectónicos supera la de STNNet, o si el sesgo por estrato tiene el mismo signo en todos los estratos que STNNet.

## 6. Conjunto local mínimo definido por cobertura

Rechazo fijar un número de marcas. Propongo definir celdas y exigir al menos una región exhaustiva por celda presente en el video:

- Densidad: dispersa, media, compacta.
- Tamaño aparente de cabeza en píxeles de la resolución original: tres rangos, con el rango inferior marcado como indeterminado si no se resuelve.
- Vista: cenital, oblicua, rasante.
- Negativos: fachadas con ventanas, pavimento, vehículos, árboles, carteles con rostros.

Cada región lleva borde cerrado, convención de frontera, marcas confirmadas, dudosas y subzonas indeterminadas. Las dudosas se reportan como cota: conteo mínimo con confirmadas, máximo con confirmadas más dudosas. Un modelo cuyo conteo cae fuera de ambas cotas falla; dentro no se declara correcto. Una segunda pasada de otra persona sobre las confirmadas es necesaria para hablar de aprobación humana. Esto detecta signo y magnitud del error por celda. No produce intervalos poblacionales ni habilita a hablar de Buenos Aires.

## 7. Cuándo combinar

Combinar solo si la tabla por celda muestra sesgos de signo opuesto en celdas disjuntas, con una regla de selección por región fijada en validación y evaluada en celdas no usadas para elegirla. Si los dos modelos sobrecuentan fachadas, la combinación refuerza el error. El acuerdo entre modelos no es anotación. La discrepancia sí sirve para priorizar revisión humana.

## 8. Críticas al informe de Astra

- **Inconsistencia interna**: la sección de MPCount habla de "cuatro recortes"; el resto del paquete y el marco hablan de siete ROI. Hay que fijar cuál es el dato.
- **SR como "CVPR2026" y la URL de CVF** son afirmaciones que no puedo verificar. Si la publicación es reciente, su página puede cambiar. Registrar fecha de consulta y guardar copia del PDF primario.
- **El hash de SR** se presenta correctamente como metadata. Falta decir que un hash de servidor sin recalcular no distingue archivo corrupto de archivo válido.
- **Licencias de pesos**: Astra separa código de pesos, bien. Pero no menciona que la licencia del dataset de origen puede condicionar pesos derivados. DroneCrowd y NWPU tienen términos propios que no se consultaron. Es un hueco relevante para publicar resultados.
- **Falta un criterio de descarte por recurso.** El informe dice que 12 GiB no garantiza nada, pero no propone qué hacer si un candidato no cabe: se trata como no evaluable, no como peor.
- **El orden de prioridad cambió entre secciones** sin justificar por qué DM-Count y MPCount bajaron. La razón plausible, artefactos visibles y pertinencia de tarea, debería estar escrita.

## 9. Críticas al marco SDD

- Define métricas pero no **qué es una persona visible**: cabeza visible, torso parcial, persona bajo bandera. Sin esa definición el recuento exhaustivo no es reproducible entre anotadores.
- No fija **cómo entran las marcas dudosas en las métricas**. Propongo el esquema de cotas de la sección 6.
- No menciona la **varianza de la propia referencia**: STNNet tiene un solo entrenamiento. Sin al menos dos semillas no se sabe si 20,30 es estable o si los pilotos cayeron dentro de la variación normal.
- Exige la misma política de tiles para comparar, pero no la declara. Debe ser una decisión congelada del marco.
- El umbral 24,36 está bien delimitado como retención. Falta decir explícitamente que ningún umbral local existe todavía y que se escribirá antes de ver candidatos.

## 10. Lo que no se puede afirmar aún

- Que algún modelo cuente mejor que STNNet en marchas locales.
- Que el deterioro de los pilotos se deba a las fotos nuevas y no a la receta.
- Que STNNet tenga MAE 20,30 en datos no vistos.
- Que la cancelación entre subconteo denso y falsos positivos arquitectónicos exista: es mi inferencia y la tabla por estrato debe confirmarla o refutarla.
- Que alguno de los candidatos corra con lote 1 dentro de 2 GiB de RAM.
- Ninguna cifra de individuos únicos ni de asistencia.

## 11. Pregunta para Astra

Para los pesos `dronecrowd.tar` de SR: ¿qué split de DroneCrowd se usó para entrenar y cuál para reportar, con qué resolución de entrada y factor de normalización del conteo, y el repositorio publica un log de evaluación que reproduzca el MAE del artículo sobre ese split? Sin esa respuesta no se puede clasificar una comparación con STNNet sobre los 30 cuadros originales como retención ni como generalización.
