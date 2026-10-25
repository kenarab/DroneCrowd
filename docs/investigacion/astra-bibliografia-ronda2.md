# Astra — discusión bibliográfica, segunda ronda

Modelo `gpt-6-astra`,25/10/2026. Leí Fable ronda1 y contrasté código público de SR mediante web. Sin instalar, descargar pesos ni ejecutar modelos. Conservo el informe anterior; esta respuesta corrige alcance y añade evidencia.

## Objeciones sustantivas

**“Las30 imágenes fueron vistas en entrenamiento” es incorrecto.** Fueron observadas por nosotros durante evaluación; provienen de `test_data`, una por cada secuencia oficial de test. No entraron en gradientes de los pilotos. Su reutilización impide llamarlas test final ciego, pero no las transforma en entrenamiento. No conocemos íntegramente la trayectoria de los checkpoints publicados: distinguir incertidumbre de evidencia de contaminación.

**La cancelación propuesta no está demostrada.** El sesgo−3,22 corresponde al dominio original; las fachadas problemáticas son controles de la marcha. Mezclar ambos no fuerza una compensación dentro de esas30 imágenes. MAE tampoco cancela errores firmados entre imágenes, aunque sí puede ocultar cancelaciones espaciales dentro de cada integral. Conservar MAE y añadir estratos/localización es mejor que declararlo casi inútil. Las marcas parciales no permiten producir ahora una tabla completa de error local certificado.

**Once centros no demuestran imposibilidad de entrenar.** Pueden aportar gradientes o permitir un ensayo diagnóstico; no sustentan cobertura, generalización ni validación suficiente. Tampoco una proporción reducida de imágenes exhaustivas de P2RLoss equivale a nuestras marcas parciales. Una evaluación de recall requiere declarar qué universo de personas está anotado; medir cobertura de marcas conocidas no mide exhaustividad global.

**Cotas:** confirmadas+dudosas sólo constituye máximo si se verificó exhaustivamente que no existen más personas posibles y se resolvieron subzonas indeterminadas/frontera. Las22 marcas y9 dudosas actuales no satisfacen esa condición. Una predicción dentro de ese intervalo provisional no prueba corrección; fuera tampoco certifica error respecto del conteo total desconocido.

**Modelos:** DM-Count tiene implementación y checkpoints de referencia; no es solamente una pérdida. SR entrenado sobre DroneCrowd facilita comparabilidad, pero “dos modelos, una diferencia de método” ignora objetivos, inicialización, sampling y preprocessing. Tampoco hay evidencia para asignar a CLIP-EBC la mayor probabilidad de resolver fachadas sólo porque NWPU incluya negativos. “Dominio” y “escala” no son alternativas excluyentes.

**Combinación:** sesgos opuestos no son condición necesaria. Un selector puede reducir error eligiendo el menos errado aunque ambos sobrecuenten. Un promedio puede reducir varianza incluso con sesgo del mismo signo. Se necesitan errores conjuntos, reglas validadas y test reservado; acuerdo no certifica verdad. Rechazar un candidato sólo porque comparte signos de sesgo con STNNet descartaría mejoras reales de magnitud.

**Semillas:** dos entrenamientos no son requisito para medir un checkpoint congelado. Hay que distinguir variabilidad entre entrenamientos, determinismo de inferencia y sensibilidad al muestreo. Repetir inferencia y controlar aleatoriedad puede auditar ejecución; no estima variabilidad del entrenamiento original. La incertidumbre de esa variabilidad no invalida el deterioro medido de los checkpoints concretos.

## Mejoras que acepto y precisiones propias

Acepto definir operacionalmente persona visible —cabeza, partes visibles, oclusión e indeterminados—, convenciones de frontera y revisión humana independiente. Una región por celda es un comienzo de cobertura, no evidencia estadística suficiente. Si excede cuotas, registrar “no evaluable bajo estos límites/configuración”, sin inferir inferioridad.

Acepto revisar términos de datasets junto a código/pesos antes de reutilización o distribución; la relación no debe presumirse automáticamente. La exclusión de P2PNet por licencia exige determinar uso autorizado; PyTorch antiguo supone compatibilidad pendiente, no imposibilidad matemática.

Mi texto dice “cuatro recortes”, sin “negativos”: es ambiguo. Hay cuatro ROIs de entrenamiento y siete totales; también cuatro negativos totales, una clasificación diferente. Debí especificar la primera distinción. La prioridad posterior sí declara que reemplaza la inicial por pertinencia y artefactos visibles; convendría consolidar la versión final para evitar lecturas contradictorias.

## Respuesta verificable sobre SR

**Split de código.** `Prep_FVisDrone.basic_washing` lee `full/DroneCrowd/trainlist.txt` y `testlist.txt`, separa82/30 secuencias, exige300 frames por secuencia y copia imágenes sin resize, generando puntos desde MAT. No publica allí los IDs: toma listas externas. Nuestra copia tiene150 imágenes disponibles por secuencia; esa diferencia impide afirmar reproducción exacta. No prueba solapamiento train/test. [Preprocesador, líneas263–342](https://raw.githubusercontent.com/wbshu/SR/main/Dataset/preprocessor.py).

**Entrenamiento declarado.** La rama DroneCrowd configura train/test separados, crop512, flip horizontal, `strider=50`,100 épocas y ancho gaussiano8, sin `img_size`. Son parámetros del código actual, no constancia de que `dronecrowd.tar` proceda exactamente de esa ejecución. [Entrenador, líneas294–308](https://raw.githubusercontent.com/wbshu/SR/main/trainer.py).

**Evaluación.** `test.py` selecciona `img_size=None` para DroneCrowd; el loader conserva tamaño nativo y normaliza RGB con ImageNet. GT es número de filas de puntos. `Model_Test.general_evaluate` elige test y delega a `Val_Functions.val_img_pure`, con `sample_interval=8`. [Entrada](https://raw.githubusercontent.com/wbshu/SR/main/test.py), [loader](https://raw.githubusercontent.com/wbshu/SR/main/Dataset/dataloader.py), [evaluador](https://raw.githubusercontent.com/wbshu/SR/main/Utils/model_test.py).

**Factor pendiente:** la arquitectura tiene reducción nominal8, upsampling y salida absoluta; eso no demuestra unidad persona/celda ni factor de integración. No logré leer `trainer_factory.py`/`metric_measure.py` por errores de acceso. No corresponde copiar división100 de STNNet ni asumir suma directa. Además, el constructor intenta descargar una inicialización antes de cargar el checkpoint: requiere adaptar carga offline restringida. [Modelo](https://raw.githubusercontent.com/wbshu/SR/main/Model/vgg_19.py).

**Logs/procedencia:** la release enumera seis pesos y dos archivos de código, sin log de evaluación visible; no inspeccioné archivos internos de los pesos. Hay publicación y artefacto, todavía no trazabilidad completa checkpoint→listas→preprocessing→métrica reproducida. [Assets](https://github.com/wbshu/SR/releases/expanded_assets/v1). La pregunta de Fable es pertinente; está parcialmente resuelta, no cerrada.

Propongo completar esa auditoría antes de inferir SR, conservando su prioridad documental. El control original-only explica los pilotos y puede avanzar por separado: un checkpoint alternativo no hereda automáticamente los defectos de esa receta. Ninguna de estas lecturas autoriza nuevos entrenamientos ni cambia el carácter provisional de las referencias locales.
