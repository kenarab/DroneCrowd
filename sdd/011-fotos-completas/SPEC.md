# Inferencia automática del original en las fotos completas

Encargo25/10/2026: ejecutar el modelo actual sin esperar anotaciones. Además de
19recortes, procesar las8fotos aportadas, sin entrenar ni seleccionar pesos nuevos.
Las anotaciones sirven para comprobar errores, no son requisito de inferencia.

## Diseño previo

Congelar fuentes, dimensiones y hashes. Preparar RGB nativo local en CPU con
Pillow; no interpolar ni usar restauración generativa. Para alineación del modelo,
excluir explícitamente hasta3píxeles de borde derecho/inferior si una dimensión no
es múltiplo de4, registrando fuente y área efectiva. Una foto tiene5159px de ancho.

La panorámica5200×12480 se divide en bandas con contexto adicional, antes de
entrar al lector/modelo. Cada panel máximo24millones de píxeles; núcleo de cada
banda disjunto, alineado a4, halo96. Mantener dentro de cada panel los mismos
tiles de núcleo320×128 y contexto96 del adaptador existente. No ampliar sus
ventanas CUDA ni límites de RAM. Ensamblar sólo núcleos con cobertura única;
sumar contribuciones crudas de todas las bandas y aplicar el clamping una sola
vez sobre los4cuadrantes de la foto completa, dividiendo por100. No sumar conteos
ya truncados por panel. Registrar raw_total y masa negativa. Alinear inicios y
contextos de bandas a8; los bordes finales conservan la geometría efectiva a4.
No prometer igualdad con una pasada sin tiles: el halo96 es una aproximación.

Una ejecución GPU por panel, lote1, máximo2GiB RAM y120segundos, preflight previo;
secuencial. Ante fallo de recursos conservar evidencia y detener ese panel sin
ampliar cuotas. No reconstruir HDF5, cambiar entorno compartido ni servir fotos
originales automáticamente. El montaje privado sólo incluirá vistas derivadas.

## Aceptación

- Verificación de hashes RGB/checkpoint, forma finita y cobertura única sin
  duplicaciones entre núcleos/bandas. Pruebas de geometría en límites y panorama.
- Original intacto, sin etiquetas necesarias ni actualización de parámetros.
- Informe por foto con dimensiones fuente/efectivas, paneles, integración, tiempos,
  memoria y hash de código/entradas; no interpretar como precisión ni convocatoria.
- Vista privada con miniatura, mapa y enlace a regiones revisadas, sin identificar
  puntos como personas verificadas. Mantener fuera de Git fuentes, pesos y mapas.
- Revisión independiente antes de GPU y commits atómicos tras comprobaciones.

No se aplica todavía máscara validada calle/vereda/plaza: incluye arquitectura y
otras áreas. La distorsión de la panorámica puede alterar la escala aparente. Es
una aplicación exploratoria del modelo existente, no un conteo aceptado del evento.

## Estado

Completadas8fotos y21bandas GPU,5111tiles, sin modificar fuentes ni pesos.
Sólo3columnas derechas excluidas de la fuente de5159px; sin padding. Orientación
EXIF1en todas. Cobertura única, mapas y conteos globales verificados por revisor
independiente Codex (modelo no identificado),4pruebas geométricas aprobadas.
Tiempo sumado reportado132,36s, RSSmáximo1,109GiB/CUDA161,09MiB, cada ejecución
dentro de2GiB/120s. Vistas privadas8JPEG/HTML comprobadas visualmente y por HTTPS
autenticado; originales y salidas fuera de Git. Implementación de panelesLuna,
vistas del coordinador. [Resultados y reproducción](../../docs/inferencia-fotos-completas-2026-10-25.md).
La máscara de suelo y la precisión siguen pendientes; no se aceptaron cifras
de asistencia ni se incorporaron propuestas como etiquetas de entrenamiento.
