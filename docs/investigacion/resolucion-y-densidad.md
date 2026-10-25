# Ampliación de imágenes y conteo de personas dispersas

Consulta del autor, 25/10/2026. Ampliar puede cambiar la respuesta del modelo,
pero los ensayos locales no muestran una mejora general ni precisión validada.
No se ejecuta otra campaña por incorporar esta nota.

## Qué permite y qué no permite el upscaling

Importa el tamaño aparente de las cabezas y el contexto, además de la resolución
total de la foto. La bibliografía trata explícitamente escala, perspectiva y
campos receptivos: [Liu, Salzmann y Fua, Context-Aware Crowd Counting, CVPR 2019](https://arxiv.org/abs/1811.10452).
Que una imagen sea 4K no asegura detalle suficiente de personas lejanas.

La interpolación Lanczos aumenta el tamaño sin añadir observaciones. Puede
acercar la escala de los patrones a la que espera una red, como hipótesis de
compatibilidad a comprobar. También cambia el contexto físico visto por nuestros
tiles de tamaño fijo y puede reforzar respuestas a sombras o arquitectura. No
es equivalente a obtener una imagen nativa con más detalle.

La superresolución generativa necesita otra validación. Mejor apariencia no
garantiza mayor fidelidad al original: [Blau y Michaeli, The Perception-Distortion
Tradeoff, CVPR 2018](https://arxiv.org/abs/1711.06077). Esta referencia estudia
restauración de imágenes, no prueba efectos concretos en nuestras multitudes.
El criterio metodológico aquí es comprobar el conteo contra referencia antes
de usar detalles sintetizados como evidencia de personas.

## Evidencia local existente

Mismos recortes y checkpoint de densidad, nativo frente a Lanczos 2×:

| Región | Nativo | Ampliado 2× |
| --- | ---: | ---: |
| Foto, peatones dispersos | 9,58 | 8,79 |
| Foto, grupo | 21,06 | 10,75 |
| Fachada sin personas discernibles | 28,12 | 19,36 |
| Cúpula sin personas discernibles | 2,07 | 4,70 |
| Calle vacía | 1,39 | 3,28 |

Son integrales regionales de densidad. El descenso o aumento no demuestra mejora
en positivos con etiquetas parciales; los negativos muestran que no hay protección
general contra falsos positivos. Las referencias humanas posteriores no reescriben
las evaluaciones congeladas ni sus antiguas cotas provisionales.
[Método y resultados originales](../controles-marcha-2026-10-25.md).
El ensayo de [video nativo e interpolado](../resultado-resolucion-original-2026-10-24.md)
también encontró diferencias grandes sin demostrar precisión.

## Requisito sobre personas aisladas

Un contador por densidad puede representar tanto personas aisladas como grupos.
La unidad deseada es una persona por masa integrada correspondiente; regiones
vacías deben producir aproximadamente cero. Que el método sea de densidad no
justifica fallar en las zonas dispersas.

Separar dos comprobaciones: error del conteo regional y ubicación de la respuesta.
Una suma cercana a la referencia puede ocultar compensación espacial. Los máximos
magenta son un postprocesado del mapa, no detecciones individuales verificadas.
Para evaluar localización o error cuantitativo se necesitan regiones exhaustivas,
una convención de frontera y tratamiento de las oclusiones, como exige el marco 000.

Decisión actual: conservar nativo como referencia; no ampliar todas las fotos como
arreglo general. Priorizar la validación en dispersos, negativos y densos y la
comparación con candidatos ya investigados. Un futuro ensayo de escala mantendrá
fuentes/ROIs congelados y comprobará conteo y falsos positivos; no elegirá escala
por obtener simplemente un número mayor. Ampliar 2× implica 4 veces los píxeles y
requiere estimación de recursos propia, sin relajar cuotas automáticamente.

Revisión independiente: GPT-6 Luna, 25/10/2026. Comprobó las cinco parejas
de resultados contra el control publicado, el enlace del índice y los límites
de las conclusiones; sin objeciones.
