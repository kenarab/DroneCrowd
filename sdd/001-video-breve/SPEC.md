# Prueba breve con un video propio

## Objetivo y responsable

Decidir si vale la pena recuperar DroneCrowd con poco tiempo y liberar espacio.
DroneCrowd posee el experimento; Nexo aporta operación y retención. El autor
autoriza preparar agentes, reproducir/adaptar un entorno y hacer commits locales.
Primero evaluar pesos existentes con inferencia; entrenamiento es una fase separada.

## Secuencia y límites

1. Conservar cambios del autor en commits independientes.
2. Diagnosticar ml sin modificarlo; registrar versiones y una operación sintética.
3. Evaluar adaptación de STNNet para conteo sin dependencias de tracking ni descarga
   VGG automática cuando se cargan pesos completos. Comprobar shapes y carga exacta.
4. Verificar ZIP de autores y hashes antes de cargar pesos, con deserialización
   restringida. No tratar una página accesible como pesos ya disponibles.
5. Con video identificado: máximo30 fotogramas, lote1, resolución inicial pequeña,
   máximo120s de ejecución; registrar resize/crops, normalización y escala de salida.
   No asignar arbitrariamente factor de conteo ni compararlo con ground truth ausente.
6. Mostrar muestra revisable, conteos y límites. No métricas de precisión sin anotaciones.
   Cualquier ampliación se decide tras la muestra y el preflight de recursos.

Primera revisión acotada de entorno/compatibilidad: aproximadamente1h de trabajo.
Si requiere portar extensiones antiguas, no prolongarla a varios días por defecto:
registrar el bloqueo y decidir otro modelo o retirar el ejercicio. Una prueba
útil podría requerir2–4h adicionales si pesos y adaptación son compatibles;
es estimación de trabajo, no promesa ni tiempo de GPU medido.

## Conservación

El dataset medido ocupa409,93GiB; aproximadamente404GiB son HDF5.
Dos muestras contienen mapas generados, no checkpoints. Falta comprobar el
conjunto antes de retirar nada. Mantener imágenes/anotaciones/código y cualquier
resultado único. La elección de copiar, retirar mapas o borrar el dataset queda
pendiente de la evaluación y decisión explícita del autor; no es un efecto de
ejecutar diagnósticos o pruebas.

## Aceptación

Versiones registradas, cambios del autor conservados, diagnóstico repetible,
commit por incremento. Una ejecución sintética demuestra compatibilidad básica,
no que haya pesos ni que funcione el video. Cerrar como parcial si faltan pesos,
adaptación o video, con pasos concretos y espacio liberado real separado.
