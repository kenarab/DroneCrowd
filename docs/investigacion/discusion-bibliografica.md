# Discusión de alternativas y protocolo de investigación

25/10/2026. Resultado: estudiar primero **SR**, luego **CLIP-EBC**, con **MPCount**
como alternativa. Esta prioridad se basa en tarea y artefactos visibles. No hay
evidencia todavía de que alguno supere a STNNet en manifestaciones locales.

## Intercambio realizado

| Paso | Documento | Alcance |
| --- | --- | --- |
| Investigación Astra | [Bibliografía](astra-bibliografia.md) | Publicaciones y repositorios primarios, sin descargar pesos |
| Lectura crítica Fable | [Ronda1](fable-bibliografia-ronda1.md) | Paquete documental, sin herramientas |
| Objeciones y código SR | [Astra ronda2](astra-bibliografia-ronda2.md) | Rectificaciones y verificación primaria adicional |
| Réplica Fable | [Ronda2](fable-bibliografia-ronda2.md) | Concesiones y desacuerdos, sin herramientas |
| Cierre Astra | [Ronda3](astra-bibliografia-ronda3.md) | Precisiones pendientes, sin nuevas fuentes |

Astra: `gpt-6-astra`, agente separado. Fable: `claude-fable-5-1`, confirmado en
los JSON de dos llamadas exitosas mediante `claude --model fable`. El primer
intento falló antes de obtener respuesta por autenticación vencida; el autor
renovó el acceso. Solicitudes y JSON privados bajo `outputs/investigacion/`.
La revisión bibliográfica de Fable fue crítica documental, no búsqueda web
independiente. La discusión entre ambos fue mediada por el coordinador.

## Rectificaciones que afectan decisiones

- Las30 imágenes históricas son del split oficial de test y fueron observadas
  por nosotros al evaluar. Eso no demuestra entrenamiento sobre ellas. La cohorte
  ya consultada tampoco debe presentarse como test final ciego.
- Las fachadas problemáticas pertenecen al material local; el sesgo histórico
  pertenece al benchmark original. No se pueden mezclar para concluir que hubo
  cancelación dentro de aquella MAE. Se mantiene MAE y se añaden estratos cuando
  exista referencia suficiente.
- Once centros efectivos no hacen matemáticamente imposible entrenar: permiten
  gradientes y diagnóstico. No sustentan cobertura o generalización.
- Marcas parciales más dudosas no constituyen una cota superior del total.
  Tampoco resolver nueve dudas valida las demás marcas ni la exhaustividad.
- DM-Count tiene modelos y checkpoints, además de una pérdida. Su menor prioridad
  en esta selección acotada no demuestra incompatibilidad de dominio.
- Evaluar un checkpoint congelado no exige reentrenarlo con dos semillas.
  Inferencia determinista, variación de entrenamientos y muestreo son preguntas
  distintas. La comparación causal de las recetas requiere controles propios.
- Sesgos de signo opuesto no son requisito para combinar modelos. Se exige medir
  errores conjuntos y validar la regla; acuerdo entre modelos no equivale a verdad.
- «Cuatro recortes» en el primer informe era ambiguo: cuatro de entrenamiento,
  siete totales; cuatro negativos totales es otra clasificación.

Fable corrigió expresamente los puntos principales en su réplica. Su ronda2
contiene otra errata: habla de22marcas tras resolver9dudas. Son22anteriores+9=31
nominales; el coordinador propuso una adicional y la v3 tiene32. Sólo las9 tienen
esa confirmación humana expresa. Las etiquetas siguen siendo parciales. También
se precisa que el marco000 y SPEC007 permiten controles acotados ya autorizados;
la bibliografía no constituye por sí misma una receta de entrenamiento nueva.

## Qué falta para ejecutar SR correctamente

La lectura primaria de [preprocesado](https://raw.githubusercontent.com/wbshu/SR/main/Dataset/preprocessor.py),
[entrenador](https://raw.githubusercontent.com/wbshu/SR/main/trainer.py) y
[evaluación](https://raw.githubusercontent.com/wbshu/SR/main/Utils/model_test.py)
identificó listas externas82/30 secuencias, exigencia de300cuadros, recorte512 en
entrenamiento y evaluación nativa. Nuestra copia tiene150cuadros por secuencia.
No se verificaron los IDs de aquellas listas ni que el checkpoint proceda de
esa configuración exacta. No se halló log de evaluación visible en la release.

Falta verificar la unidad de salida y su integración: no copiar el factor100
de STNNet. El constructor intenta descargar una inicialización antes de cargar
pesos; la carga offline restringida debe resolverse antes del ensayo. Son
obstáculos de reproducción pendientes, no prueba de inferioridad de SR.

## Cambios incorporados al marco

El [marco000](../../sdd/000-investigacion-reproducible/SPEC.md) distingue presencia,
cabeza localizable, sombra e indeterminación; frontera semiabierta; confirmación
por punto; y cotas sólo cuando estén justificadas. Añade recursos no evaluables,
términos independientes de código/pesos/datos y métricas/tolerancias congeladas
antes de comparar. Una región por condición ayuda a cubrir casos, sin demostrar
validez estadística para Buenos Aires.

No se decidió una arquitectura híbrida. La revisión técnica ayuda a encontrar
errores, pero la validación de etiquetas y conclusiones necesita evidencia local
y escrutinio profesional. El trabajo se presenta como investigación reproducible,
abierta a discusión con competencia técnica y profesional y honestidad intelectual.
