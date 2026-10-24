# Resolución nativa y ampliación para la marcha

## Encargo y aceptación

El autor pide probar resolución original y aplicar upscaling donde falte.
Continuación de la prueba negativa a 960×540, sin entrenamiento ni cambios al
venv compartido. Pesos, carga restringida y normalización se conservan.

Ensayo acotado a un fotograma por video: 0265 al 90% (20,570550 s), 0269 al 50%
(4,704700 s) y trim al 50% (5,496667 s). Verificar PTS y hash del video con la
evidencia previa. Los dos MOV tienen 3840×2160; trim tiene 960×540.

Siete configuraciones secuenciales: los MOV a resolución nativa y, como control
con igual geometría de inferencia, sus imágenes previas de 960×540 ampliadas 4×;
trim a tamaño nativo, ampliado 2× y ampliado 4×. Ampliación FFmpeg Lanczos,
sin descargas ni modelos externos. No recupera información ausente. No multiplicar
conteos por el factor de ampliación ni por el área.

## Geometría y recursos

Inferencia de mosaico: núcleo 320×128, halo de contexto de 96 píxeles por lado,
tile máximo 512×320, lote 1. Orígenes y límites pares, mapa de densidad a mitad
de tamaño. Recortar el halo en la salida y escribir cada píxel global una sola
vez; sin sumar solapes. En bordes sólo se usa el contexto existente de la imagen.
El halo reduce artefactos de borde, sin afirmar equivalencia con inferencia del
fotograma completo. Cambia el contexto respecto de los cuatro crops anteriores.

Conservar NPY crudo global y calcular las mismas cuatro sumas de cuadrantes,
limitadas inferiormente a cero y divididas por 100, como en el ensayo anterior.
No aplicar clamp por cada tile. Guardar geometría, filtros, hash del RGB
transformado, versiones, tiempos y recursos. La imagen ampliada queda en memoria;
no persistir duplicados RGB grandes. Entradas nativas nuevas: una por MOV.

Preflight Nexo: estimación transitorio 1 GiB, retención 0,25 GiB y RAM 1,25 GiB
por proceso, con margen Nexo y reservas de 20 GiB disco / 2 GiB RAM. Reconsultar
GPU y pool. Scope High 1,5 GiB / Max 2 GiB / swap 0,5 GiB y timeout externo de
120 s por configuración, secuenciales. No ampliar cuotas por fallos. Salidas de
este ensayo en `outputs/resolucion-original/`, máximo 128 MiB adicionales.
Temporales/cachés en `~/home.codex/tmp`; originales de videos y ZIP intactos.

## Verificación y entrega

- Probar cobertura única, alineación y conservación de masa con modelo sintético
  sólo en fixtures; presupuesto/geometría y hash antes de lecturas grandes.
- Extraer native sin redimensionar y comprobar PTS y procedencia.
- Ejecutar las siete configuraciones y revisar imágenes/mapas visualmente.
- Tabla comparativa y figuras locales, distinguiendo resolución real e interpolada.
- Sin anotaciones, evaluar plausibilidad y ubicación; no afirmar precisión ni un
  total de convocatoria, ni interpretar un conteo mayor como una mejora demostrada.
- Revisión independiente por el agente autorizado existente; commits atómicos
  de especificación, implementación validada e informe. No publicar artefactos.

## Estado

Plan en ejecución. Resultados y criterio de utilidad pendientes.
