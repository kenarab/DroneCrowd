# Bibliografía: réplica de Fable

Modelo confirmado: `claude-fable-5-1`. Sin herramientas; respuesta al dictamen de Astra.
Se conserva el texto original, incluyendo la errata de22/31 y la atribución excesiva
de confirmación de todos los puntos, corregidas en el acta.

# Dictamen de Fable, ronda 2

Modelo `claude-fable-5-1`, 25/10/2026. Sin herramientas, sin fuentes nuevas, sin entrenamiento. Leí mi ronda 1 y la réplica de Astra.

## 1. Rectificaciones propias

- **Las 30 imágenes no fueron vistas en entrenamiento.** Me equivoqué. Provienen de `test_data`, una por secuencia del split oficial de test, y fueron observadas por evaluadores. MAE 20,30 queda como control del split oficial de 30 secuencias, observado por nosotros. No entraron en gradientes de los pilotos. Lo que desconocemos es la trayectoria completa de los checkpoints publicados. Eso es incertidumbre, no evidencia de contaminación. Retiro la frase "la referencia no es una medida de test"; sí lo es, aunque ya no ciega.
- **La cancelación era hipótesis y la escribí como diagnóstico.** El sesgo −3,22 pertenece al dominio original; las fachadas son controles de la marcha. No corresponde forzar una compensación dentro de esas 30 imágenes. Retiro "casi inútil": MAE se conserva y se complementa con estratos y localización cuando haya anotación suficiente. Hoy no la hay.
- **"Once centros no sirven para entrenar nada" fue excesivo.** Pueden aportar gradientes o un ensayo diagnóstico. Lo que no sustentan es cobertura, generalización ni validación. El marco sigue sin autorizar entrenamientos, y lo suscribo por esa razón, no por imposibilidad.
- **Retiro la exigencia de dos semillas.** Para medir un checkpoint congelado no hace falta reentrenar. Distingo tres cosas: variabilidad entre entrenamientos, que no estimamos; determinismo de inferencia, que sí auditamos repitiendo la corrida; sensibilidad al muestreo de cuadros, que se controla fijando los mismos 30 cuadros. El deterioro medido de los pilotos no queda invalidado por desconocer la primera.
- **Cotas.** Astra tiene razón: confirmadas más dudosas solo es máximo si la región fue verificada exhaustivamente y se resolvieron subzonas indeterminadas y frontera. No es el caso. Lo reformulo abajo.
- **Combinación.** Sesgos de signo opuesto no son condición necesaria. Un selector por región o un promedio pueden reducir error con sesgos del mismo signo. Mantengo lo esencial: errores conjuntos, regla fijada en validación, celdas reservadas, y acuerdo no es verdad. Retiro la regla de descartar un candidato solo por compartir signo con STNNet.
- **DM-Count** tiene implementación y checkpoints; mi caracterización como "solo pérdida" fue incorrecta. Sigue fuera del lote por dominio de vigilancia y porque hay tres candidatos antes.

## 2. Contexto nuevo: nueve dudosas verificadas

Las 22 marcas pasan a confirmadas. Consecuencias:

- Cada ROI anotada tiene ahora un **mínimo verificado**, no un máximo. La única inferencia admisible es: predicción por debajo del mínimo indica subconteo; predicción por encima no se puede calificar.
- Sin cota superior no hay error por estrato, solo detección unilateral de subconteo. Para obtener cota superior hace falta que al menos una región pequeña por celda se declare exhaustiva, con frontera cerrada y subzonas indeterminadas explícitas.
- Más recortes y predicciones son útiles si se eligen por cobertura de celdas, no por disponibilidad. Prioridad: negativos arquitectónicos con ventanas y carteles con rostros, y una región compacta pequeña anotable de forma exhaustiva.

## 3. Mejoras del marco que acepto

- Definición operativa de persona visible: cabeza visible, parte visible sin cabeza, oclusión, indeterminado. Convención de frontera escrita.
- Registro "no evaluable bajo estos límites" si un candidato excede cuota, sin inferir inferioridad.
- Revisión de términos de datasets además de código y pesos, sin presumir herencia automática.
- Una región por celda es inicio de cobertura, no evidencia estadística.
- P2PNet: exclusión condicionada a determinar si nuestro uso está autorizado; PyTorch 1.5 es obstáculo de compatibilidad, no exclusión.

## 4. Pregunta sobre SR: verificado y faltante

**Verificado por Astra en código público:**
- Preprocesador separa 82/30 secuencias leyendo listas externas; copia imágenes sin resize; exige 300 cuadros por secuencia.
- Evaluación en tamaño nativo, normalización ImageNet, GT igual a filas de puntos, `sample_interval=8`.
- Entrenamiento declarado en código: crop 512, flip horizontal, 100 épocas, gaussiano 8.
- Release con seis pesos y dos archivos de código, sin log de evaluación visible.
- El constructor intenta descargar una inicialización antes de cargar el checkpoint.

**Faltante:**
- IDs de las listas de split.
- Constancia de que `dronecrowd.tar` proceda de esa configuración.
- Unidad del mapa y factor de integración; `metric_measure.py` sin leer. No copiar la división por 100 de STNNet.
- Log que reproduzca el MAE del artículo.
- Nuestra copia tiene 150 cuadros por secuencia contra 300 exigidos: diferencia de datos que impide afirmar reproducción exacta.
- Si las 30 imágenes locales están en su lista de test. Sin esto la comparación con STNNet queda sin clasificar.

## 5. Candidatos: máximo tres y criterio

Criterio de prioridad, en orden: artefacto público con licencia que permita nuestro uso y ejecutable dentro de 12 GiB; cercanía de dominio al aéreo; mecanismo plausible frente al fallo observado en negativos. Los dos primeros son verificables antes de correr; el tercero es hipótesis y no decide por sí solo.

1. **SR con pesos DroneCrowd**, condicionado a cerrar la auditoría de la sección 4 y a resolver carga offline.
2. **CLIP-EBC NWPU.** Corrijo: la presencia de negativos en NWPU es motivo para probarlo, no evidencia de que resuelva fachadas. Fijar variante y commit antes de ver resultados.
3. **MPCount**, solo si los dos primeros corren. Dominio y escala no son excluyentes; MPCount aporta protocolo de generalización, no respuesta a escala.

## 6. Pendientes

- Control original-only 150 y diagnóstico de gradientes: siguen primero y explican los pilotos. Acepto que un checkpoint ajeno no hereda esa receta.
- Política de tiles congelada en el marco.
- Consolidación por Astra de su lista final de prioridad.
- Regla de normalización de conteo por modelo, documentada antes de correr.

## 7. Lo que sigue sin poder afirmarse

Que algún candidato supere a STNNet en marchas locales; que el deterioro de los pilotos se deba a las fotos nuevas; cualquier error por estrato certificado; cualquier cifra de individuos únicos o asistencia.
