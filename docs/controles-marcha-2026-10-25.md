# Controles visuales de la marcha y dos checkpoints STNNet

STNNet cuenta de forma plausible peatones distinguibles en algunos recortes,
pero confunde texturas arquitectónicas con personas y responde débilmente en
multitudes densas. La cabeza de densidad del checkpoint de localización mejora
algunos controles positivos, sin resolver el fondo: la fachada seleccionada
recibe todavía más masa. No usar estos modelos sin adaptar como estimadores
validados de la convocatoria.

| Control | Referencia visual provisional | den 1× | den 2× | loc: densidad 1× | loc: densidad 2× |
|---|---|---:|---:|---:|---:|
|Foto, peatones dispersos|7–8 candidatos|9,58|8,79|7,61|10,67|
|Foto, grupo|12–19 candidatos|21,06|10,75|16,09|12,71|
|Fachada|Sin personas discernibles|28,12|19,36|33,85|42,10|
|Cúpula|Sin personas discernibles|2,07|4,70|2,07|6,09|
|Calle vacía|Sin personas discernibles|1,39|3,28|1,15|4,50|
|Video, multitud densa|Sin recuento fiable|0,53|0,88|2,03|3,25|

Las cifras son integrales de densidad dentro del recorte, en unidades de personas.
No son detecciones individuales. `loc` identifica el checkpoint entrenado con
localización; esta comparación usa su salida den1 y no decodifica picos de la
cabeza de localización. 2× significa Lanczos sobre los mismos píxeles originales.

## Referencias y figuras

- [Personas distinguibles y puntos revisados](../outputs/validacion-controles/reporte-rois/personas.png).
- [Negativos: arquitectura y calle](../outputs/validacion-controles/reporte-rois/negativos.png).
- [Recortes densos sin anotación exhaustiva](../outputs/validacion-controles/reporte-rois/densos-sin-recuento.png).
- [CSV de las 32 comparaciones](../outputs/validacion-controles/reporte-rois/counts.csv).

El coordinador marcó puntos visibles y candidatos dudosos; el revisor independiente
comprobó las imágenes antes de consultar predicciones numéricas de los ROIs.
La versión v2 desplazó dos puntos ambiguos junto a un paraguas a dudosos y retiró
la cota total de un recorte de video con bordes ambiguos. Luego se congeló y no
se modificó al ver las salidas. Los mapas globales de la fase anterior ya habían
sido vistos: la selección no se presenta como completamente ciega.

Son referencias provisionales de agentes, sin validación humana externa. En los
negativos se afirma ausencia de personas discernibles, sin certificar cada píxel
de una azotea distante. Los intervalos describen candidatos visibles y no son
cotas certificadas de personas ocultas. Además, la masa de una Gaussiana puede
cruzar el límite del ROI aunque el centro de cabeza quede dentro o fuera.
Por estos motivos no se informa un MAE certificado para la marcha ni se ajusta
un multiplicador usando estos intervalos.

Ocho controles: cinco procedentes de los dos fotogramas nativos y tres de la
foto aportada `Copy of IMG_8418.JPG`. Las fotos se identifican aparte; no se
asume correspondencia temporal con el video. Coordenadas semiabiertas, puntos
y fuentes están en `outputs/validacion-controles/rois-referencia-v2.json`, SHA-256
`8ced28dcf1d1c197086b2a654428f295cabe2d2124d8f5989e73c7fc7f37e7cf`.
Los RGB con contexto de 128 píxeles y sus hashes están en `rois-preparados/`,
manifiesto SHA-256 `1e71b3e228c14c222af2e2b12f4177c3a62a0adb269ca36ef865696e37356a12`.

Inferencia por mosaicos con núcleo 320×128 y halo 96; sólo se integra el ROI
central tras ensamblar sin duplicados. Suma float64, división por 100 y clamp
inferior después de sumar. Los mapas crudos conservan negativos. Las figuras
usan BOX y compensación de área sólo para visualizar la misma grilla espacial;
la escala de color es compartida dentro de cada fila, no entre ROIs. No se
multiplica el conteo por el área ni por el factor de ampliación.

## Checkpoints, pruebas y recursos

El checkpoint den completo tiene 41 tensores; loc, 61. Ambos cargan nombres,
formas, tipos y finitud estrictos con `weights_only=True`. La metadata NumPy
float64 antigua usa la misma excepción mínima temporal, ligada a los hashes
exactos de las dos entradas del ZIP aportado. No se descargó VGG16 ni se
habilitó tracking. Procedencia loc: `outputs/validacion-controles/weights-localization.json`;
el ZIP original conserva la entrada para regenerar su copia temporal.

En el mismo piloto histórico de seis imágenes, den obtuvo MAE 19,38 y loc 15,42.
Esto motiva el contraste, sin extrapolar el resultado a todo el benchmark ni
elegir el modelo por dos recortes de la marcha. No se completó otra evaluación
de 30 imágenes loc porque la falla de fondo ya persiste en el control objetivo.
[Validación histórica de den](calibracion-historica-2026-10-25.md).

Las cuatro ejecuciones ROI terminaron bajo scope de 2 GiB RAM, swap 0,5 GiB y
timeout 120 s; lote 1 y GPU RTX 3060. El entorno compartido no se modificó.
Diez tests de conteo/carga pasaron tras añadir la arquitectura loc explícita;
dos tests ROI verifican coordenadas, masa a diferentes escalas, clamp y errores
de límites. El revisor Codex autorizado `/root/revision_retencion_hdf5`, modelo
efectivo no expuesto, revisó código, referencias y los 32 mapas de den/loc,
verificó integrales, hashes y figuras, y aprobó el cierre de esta comparación.

El autor propone ajustar pesos con nuevas anotaciones y reutilizar datos
originales. Es la continuación prioritaria: preparar etiquetas revisables,
reservar fuentes para evaluación y probar una mezcla con replay del conjunto
original. Los controles anteriores quedan congelados como diagnóstico; las
etiquetas provisionales no pasan a GT humano por usarlas durante entrenamiento.
