# Investigación reproducible de multitudes en manifestaciones

Estado: especificación marco, 25/10/2026. Autoriza investigación y controles
acotados; no acredita todavía un contador validado para Buenos Aires.

## Pregunta y alcance

Medir cuántas personas son visibles dentro de una región y un instante definidos
en imágenes de manifestaciones argentinas, inicialmente Buenos Aires.
El área local comprende suelo de calles, veredas y plazas, excluyendo terrazas.
Fijar la máscara espacial explícitamente; reconocer una persona fuera del área no
implica incluirla en el conteo de esa manifestación. No hay aún máscara validada.
Registrar área cubierta, instante, perspectiva, resolución original y oclusiones. Una suma
de fotogramas no estima asistentes únicos. La asistencia acumulada requiere una
especificación temporal independiente y evidencia sobre duplicaciones y cobertura.

La investigación debe ser reproducible y abierta a discusión por personas con
competencia técnica y profesional y honestidad intelectual. Las objeciones se
resuelven mediante evidencia, métodos explícitos y experimentos contrastables.
Registrar discrepancias y resultados adversos con el mismo rigor que las mejoras.

## Método de trabajo

1. Escribir la pregunta, hipótesis, particiones, recursos, criterios de aceptación
   y parada en una SPEC antes de ejecutar el experimento correspondiente.
2. Mantener una referencia intacta. Cambiar un factor por control cuando se quiera
   atribuir causalidad. Distinguir reproducción de un artículo, adaptación local
   y propuesta nueva; documentar diferencias con los autores.
3. Revisar código y protocolo con un revisor distinto del implementador, indicando
   herramienta/modelo si se conoce. La revisión por agentes no equivale a una
   revisión científica externa ni a validación humana de las etiquetas.
4. Cerrar cambios validados con commits atómicos. Publicar código, especificaciones,
   referencias y resultados agregados; conservar los datos personales fuera de Git.

## Datos y anotaciones

Separar entrenamiento, validación y evaluación por evento y fuente de video,
evitando que cuadros cercanos crucen particiones. Con un solo evento no se puede
afirmar generalización a Buenos Aires. Congelar manifiestos, hashes y política de
frontera antes de comparar modelos. La cohorte histórica de30 imágenes ya fue
observada: sirve como control de retención, no como evaluación final ciega.

Anotar exhaustivamente regiones pequeñas de varias densidades, escalas y vistas,
junto con negativos de fachadas y superficies. Distinguir marcas confirmadas,
dudosas y zonas no anotadas. Una región parcial no constituye un negativo completo
ni permite medir recall. La aprobación humana debe quedar explícita y versionada;
mostrar una lámina no implica aprobar sus puntos.

Convención inicial: contar una presencia visualmente sustentada dentro del área;
anotar como punto el centro de cabeza visible. Cuerpo o sombra coherentes pueden
apoyar la presencia de una persona cuando su cabeza no se resuelve, pero deben
registrarse separados de una localización de cabeza confirmada. Una sombra no es
otra persona. No inferir individuos totalmente ocultos. Usar regiones semiabiertas
`[x0,x1) × [y0,y1)` para asignar centros a una sola región; los casos de borde sin
centro resoluble se registran como indeterminados. Una futura anotación exhaustiva
debe resolver explícitamente estos casos antes de evaluar detecciones.

Las marcas conocidas pueden dar un mínimo de referencia si su presencia se valida;
sumar dudosas sólo da un máximo si además se verificó exhaustividad y no quedan
zonas indeterminadas. En la colección parcial actual no hay cota superior del total
ni MAE local certificado. Registrar por punto qué confirmó el autor y qué propuso
el agente. Una confirmación parcial no valida todo el recorte.

Situación inicial: siete regiones provisionales,22 marcas nominales y9 candidatas
dudosas. Sólo11 centros nuevos de entrenamiento quedaron supervisados por las
máscaras del piloto. No son un benchmark local exhaustivo.

Tras revisión del autor se reclasificaron las9candidatas y el coordinador propuso
otra marca omitida: versión3 con32marcas nominales, todavía parcial. Sólo las9
candidatas tienen esa confirmación humana expresa. Se agregaron12regiones para
revisión conjunta, sin referencia todavía; se conserva la versión2 de los pilotos.
En v4 se agrega `presence_only P1` entre dos marcas de foto-dispersas: presencia
indicada por el autor, centro de cabeza sin resolver. No se suma como otro punto
de cabeza ni se utiliza para supervisión espacial. Son32marcas y esa presencia
adicional separada. La discrepancia posterior sobre test-peaton-fuente se conserva.

## Comparación y aceptación

Elegir candidatos por pertinencia de tarea, evidencia primaria, código/pesos
accesibles, licencias diferenciadas y coste medido. Un año más reciente o un MAE
publicado en otro dataset no establece superioridad local. Comparar todos sobre
los mismos datos y convenciones; registrar también los casos excluidos.

Medir MAE, RMSE y sesgo por densidad/escala/evento cuando haya anotaciones completas;
masa falsa en negativos; localización cuando exista referencia exhaustiva; tiempo,
RAM y VRAM. Informar distribución de errores y limitaciones del tamaño de muestra.
Los intervalos estadísticos deben respetar la dependencia por evento; con una sola
fuente no se deben presentar como incertidumbre poblacional de manifestaciones.

La combinación de modelos requiere primero demostrar errores complementarios y
mejora en validación separada, junto con su coste. El acuerdo entre modelos no
constituye una anotación verdadera. No ajustar ni seleccionar sobre el test final.
El umbral de MAE histórico24,3621808 pertenece al control de retención vigente;
no es una tolerancia de error aceptable para una convocatoria.

Todavía no existen tolerancias locales aprobadas. Se fijarán en la SPEC del ensayo
antes de consultar candidatos. El protocolo de entrada, escala y tiles se congelará
allí por modelo; documentar diferencias necesarias y sus ablaciones, sin confundir
preprocesado idéntico con preprocesado científicamente equivalente.

## Reproducción y recursos

Cada ejecución registra commit y cambios sin commit, comandos, semilla, versiones,
preprocesado, variante exacta, pesos/hash, manifiestos, pérdidas, métricas y cuotas.

Si un candidato excede las cuotas se registra como no evaluable bajo esa
configuración, sin inferir que cuente peor. Separar determinismo de inferencia,
variación de entrenamientos y sensibilidad al muestreo; evaluar un checkpoint fijo
no requiere reentrenarlo con otras semillas. Verificar por separado los términos
de código, pesos y datos antes de reutilizar o redistribuir artefactos.
Distinguir enlace anunciado, artefacto visible, descarga verificada y ejecución
reproducida. Publicar restricciones de acceso que impidan repetir el experimento.

No es necesario almacenar todos los mapas HDF5: conservar imágenes/anotaciones y
regenerar blancos verificables en memoria. No sustituir detalles ausentes mediante
upscaling y tratarlos como evidencia observada. Para nuevos modelos preparar un
entorno propio si hace falta, sin modificar el entorno compartido ni los drivers.

Mantener las reservas y cuotas vigentes; no ampliarlas automáticamente. Ningún
documento dispara entrenamiento, descarga o publicación de medios privados.

## Entregables vinculados

- [Índice de investigación y selección](../../docs/investigacion/README.md).
- [Revisión conjunta del ajuste fallido](../../docs/revision-conjunta-entrenamiento-2026-10-25.md).
- [Primera prueba de video](../001-video-breve/SPEC.md),
  [resolución](../003-resolucion-original/SPEC.md),
  [controles históricos](../004-validacion-controles/SPEC.md),
  [ajuste exploratorio](../005-ajuste-dominio/SPEC.md) y
  [auditoría del diseño](../006-revision-diseno/SPEC.md).

La aceptación de este marco exige revisión independiente y enlaces válidos. La
validación del modelo y la estimación de asistencia siguen siendo trabajo abierto.
