# Bibliografía contrastada para replantear el conteo

Autor de esta revisión: Astra (`gpt-6-astra`). Consulta web: 25/10/2026. Se verificaron publicaciones primarias y repositorios oficiales; no se descargaron pesos, instalaron dependencias ni ejecutaron modelos. Documento de investigación para la siguiente SPEC, no decisión de arquitectura ni protocolo ya ejecutado. Las licencias indicadas corresponden al código visible; no se presume que cubran automáticamente pesos, datasets o imágenes.

## Alcance y evidencia local

El contexto recibido es STNNet con MAE20,3 sobre30 imágenes originales ya observadas, errores en arquitectura y multitudes densas, y dos ajustes rechazados. Siete recortes y22 puntos nominales con9 dudosos no son un conjunto de validación completo;19 positivos nominales de entrenamiento dejaron11 centros efectivos. Esos controles permiten descubrir fallos, no establecer cuál modelo cuenta mejor una manifestación.

Ninguno de los seis trabajos siguientes establece, en las fuentes consultadas, precisión específica para marchas argentinas o Buenos Aires. Su relevancia local es una hipótesis transferible a verificar. No se comparan sus MAE publicados directamente con20,3: cambian datos, particiones, geometría y tarea.

## Seis trabajos y qué permiten preguntar

### 1. DM-Count: evitar imponer una Gaussiana como verdad espacial

Boyu Wang, Huidong Liu, Dimitris Samaras y Minh Hoai, **Distribution Matching for Crowd Counting**, NeurIPS2020. Compara distribuciones normalizadas mediante transporte óptimo y agrega regularización de variación total. Presenta una cota de generalización y resultados en UCF-QNRF, NWPU, ShanghaiTech y UCF-CC50. La evidencia favorece su formulación frente a referencias evaluadas; no demuestra que todo kernel gaussiano sea incorrecto ni garantiza generalización geográfica. [Publicación](https://proceedings.neurips.cc/paper/2020/hash/118bd558033a1016fcc82560c65cca5f-Abstract.html).

El [repositorio oficial](https://github.com/cvlab-stonybrook/DM-Count) anuncia pesos para esos datasets; código con [licencia MIT](https://raw.githubusercontent.com/cvlab-stonybrook/DM-Count/master/LICENSE). No verifiqué la descarga ni compatibilidad de los binarios.

**Hipótesis Buenos Aires:** una referencia de densidad que no dependa del sigma local puede esclarecer si los errores dependen del blanco espacial. No basta cambiar la pérdida de STNNet y atribuir cualquier mejora a DM-Count. Tampoco acepta automáticamente puntos incompletos como GT completo: normalizar una distribución parcial cambia el objeto comparado.

### 2. MPCount: generalización desde un solo dominio

Zhuoxuan Peng y S.-H. Gary Chan, **Single Domain Generalization for Crowd Counting**, CVPR2024. Usa memoria de atención, máscara de error de contenido y consistencia de atención; agrega clasificación por parches para reducir ambigüedad entre densidad y fondo. Evalúa cambios entre datasets y escenarios de distribución fuente estrecha. Su protocolo es más pertinente al problema de transferencia que elegir sólo por resultados dentro de un dataset. [Publicación](https://openaccess.thecvf.com/content/CVPR2024/html/Peng_Single_Domain_Generalization_for_Crowd_Counting_CVPR_2024_paper.html).

El [repositorio oficial](https://github.com/Shimmer93/MPCount) muestra Apache-2.0 y pesos. Distingue una versión original con entrenamiento no determinista por interpolación y otra versión determinista: la comparación debe fijar cuál se usa.

**Hipótesis Buenos Aires:** clasificación de presencia por parche podría reducir respuesta a fachadas. Debe comprobarse que no elimine cabezas diminutas u ocluidas. El artículo no valida una máscara urbana entrenada con nuestros cuatro recortes ni una adaptación con11 centros.

### 3. P2PNet: contrastar densidad con localización explícita

Qingyu Song, Changan Wang, Zhengkai Jiang, Yabiao Wang, Ying Tai, Chengjie Wang, Jilin Li, Feiyue Huang y Yang Wu, **Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework**, ICCV2021. Predice puntos y confianza, y propone nAP para evaluar localización además del conteo. El interés es detectar cancelaciones de errores que una integral puede ocultar, no asumir que puntos sean superiores en toda densidad. [Artículo](https://arxiv.org/abs/2107.12746).

El [repositorio oficial](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet) incluye un modelo ShanghaiTechA y advierte que fue probado con PyTorch1.5. La [licencia propia](https://raw.githubusercontent.com/TencentYoutuResearch/CrowdCounting-P2PNet/main/LICENSE) limita el uso a investigación académica; no es MIT/BSD permisiva aunque liste componentes de terceros con esas licencias.

**Hipótesis Buenos Aires:** auditar puntos predichos en cornisas, ventanas y zonas poco densas facilitaría distinguir falsos positivos de densidad difusa. No convertir sus puntos en anotaciones verdaderas ni usar las marcas parciales actuales para medir exhaustividad/recall.

### 4. CAN: contexto espacial variable para perspectiva y escala

Weizhe Liu, Mathieu Salzmann y Pascal Fua, **Context-Aware Crowd Counting**, CVPR2019; preprint2018. Combina features con distintos campos receptivos y aprende sus pesos espacialmente. Reporta ventajas especialmente con efectos de perspectiva: aporta una alternativa concreta al supuesto de contexto uniforme. [Publicación](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Context-Aware_Crowd_Counting_CVPR_2019_paper.html).

El [repositorio del autor](https://github.com/weizheliu/Context-Aware-Crowd-Counting) identifica GPL-3.0 y enlaza un checkpoint ShanghaiTechB; esa implementación no debe confundirse con todas las variantes del artículo.

**Hipótesis Buenos Aires:** probar estabilidad frente al tamaño aparente de cabeza, vista oblicua y contexto de recorte. No equivale a recuperar personas ocultas ni calibrar automáticamente una cámara aérea. Cambiar tamaño de tiles altera contexto: comparar nativo y tiles con cobertura única antes de interpretar la diferencia como efecto de perspectiva.

### 5. NWPU-Crowd: negativos y localización como parte del benchmark

Qi Wang, Junyu Gao, Wei Lin y Xuelong Li, **NWPU-Crowd: A Large-Scale Benchmark for Crowd Counting and Localization**, preprint y publicación online T-PAMI2020. Contiene5109 imágenes,2.133.375 cabezas y rango0–20.033; la web de autores destaca muestras negativas, resolución alta y variación de apariencia. [Artículo](https://arxiv.org/abs/2001.03360), [proyecto oficial](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/index.html).

El proyecto enlaza checkpoints de varios métodos. El [código oficial de ejemplo](https://github.com/gjy3035/NWPU-Crowd-Sample-Code) indica MIT; no extiendo esa licencia a los datos o pesos. Los enlaces publicados no se comprobaron descargando archivos.

**Hipótesis Buenos Aires:** adoptar la práctica de medir negativos explícitos y localización, agregando fachadas y superficies locales reservadas. Entrenar con NWPU no prueba robustez a nuestra arquitectura. Lo más inmediatamente transferible es el diseño de evaluación, sin descargar todo el dataset.

### 6. DecideNet: antecedente de combinación, no garantía de mejora

Jiang Liu, Chenqiang Gao, Deyu Meng y Alexander G. Hauptmann, **DecideNet: Counting Varying Density Crowds Through Attention Guided Detection and Density Estimation**, CVPR2018; preprint2017. Combina mapas provenientes de detección y regresión mediante atención que estima su fiabilidad espacial. Los resultados en tres datasets apoyan esa combinación concreta. No avalan promediar arbitrariamente salidas de modelos distintos. [Publicación](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_DecideNet_Counting_Varying_CVPR_2018_paper.html).

No verifiqué código oficial, pesos ni licencia en las fuentes primarias consultadas; no se selecciona como implementación inmediata.

**Hipótesis Buenos Aires:** ramas con errores complementarios podrían servir para elegir método por región. Primero debe medirse complementariedad sobre anotaciones completas. Dos modelos que inventan personas en ventanas pueden reforzar el error. La confianza del detector no equivale a fiabilidad calibrada fuera de dominio.

## Requisitos propuestos para SDD y ensayos posteriores

**Problema especificado:** contar personas visibles en área y tiempo definidos. Cantidad única de asistentes exige otro protocolo temporal; sumar cuadros o vuelos duplica personas. No deducir afiliación individual.

**Datos de aceptación:** reservar por evento/video y fuente, no por cuadros cercanos; anotar exhaustivamente pequeñas regiones de varias escalas y densidades, junto con negativos arquitectónicos. Registrar incertidumbre/oclusión y convención de borde. Si sólo hay un evento, declarar ese límite en vez de presentar generalización a Buenos Aires. Las regiones imposibles de resolver quedan como indeterminadas, no como cero.

**Métricas anteriores a selección:** MAE, RMSE y sesgo por estrato; conteos predichos en negativos; precisión/recall de puntos cuando corresponda; discrepancia entre suma de subregiones y total; RAM, VRAM y tiempo. Mantener separado el control original ya consultado del test local reservado. Definir umbrales y regla de rechazo antes de mirar candidatos; la bibliografía no fija tolerancias locales válidas.

**Orden contrastable:** primero cerrar el diagnóstico de la receta STNNet y su control original-only. En paralelo documental, auditar procedencia, licencia, preprocessing y carga restringida de candidatos. Después, proponer comparación congelada de un candidato de densidad —DM-Count— y uno de generalización —MPCount— contra STNNet; sumar puntos sólo si se resuelve alcance de licencia y compatibilidad. CAN queda como comparación de contexto y DecideNet como hipótesis de diseño, no barrido obligatorio de seis implementaciones.

**Recursos:** RTX3060 de12GB no garantiza compatibilidad ni memoria suficiente. Un futuro preflight debe estimar pico, mantener reservas de20GiB disco/2GiB RAM y límites existentes, lote1, resolución acotada y ejecución secuencial. La prueba pequeña mide ejecución; aumentar resolución requiere nueva estimación. No se propone entrenamiento completo ni elevar cuotas para reproducir papers.

**Combinación condicionada:** sólo después de evaluar modelos individuales, estudiar errores conjuntos en validación. Una mezcla o selector necesita regla congelada, tratamiento de fronteras sin doble conteo y comparación con el mejor modelo individual en test no usado para elegir. Si no mejora el compromiso entre error local, negativos y coste, se descarta. Discrepancia entre modelos puede priorizar revisión humana, pero acuerdo no certifica conteo.

Estado: seis antecedentes y la actualización siguiente; arquitectura final, descargas, pesos y experimentos siguen sin decidir ni ejecutar. Datos personales permanecen locales y fuera de Git.

## Actualización 2025–2026: tres trabajos adicionales

La selección anterior aporta antecedentes, no representa por sí sola el estado actual. Esta ampliación verifica fuentes hasta septiembre2026; no es una revisión exhaustiva ni identifica un ganador. Investigación reproducible significa conservar versiones, protocolos y resultados negativos, con escrutinio de personas con competencia técnica profesional y honestidad intelectual.

**URM, CVPR2025.** Xianing Chen, Si Huo, Borui Jiang, Hailin Hu y Xinghao Chen: *Single Domain Generalization for Few-Shot Counting via Universal Representation Matching*. Destila representaciones CLIP en prototipos y construye correlaciones mediante atención. Evalúa FSC147/FSCD-LVIS: es conteo de objetos condicionado por ejemplos, no validación de cabezas en marchas. [Artículo](https://arxiv.org/abs/2505.16778). [Código oficial](https://github.com/jbr97/URM); su [README interno](https://raw.githubusercontent.com/jbr97/URM/main/URM/README.md) describe entrenamiento con PyTorch1.10. No verifiqué pesos ni licencia explícita. Hipótesis: prototipos más transferibles; faltan pruebas sobre cabezas pequeñas y fondos locales. Sin evidencia de reproducción en12GB.

**T2ICount, CVPR2025.** Yifei Qian, Zhongliang Guo, Bowen Deng, Chun Tong Lei, Shuai Zhao, Chun Pong Lau, Xiaopeng Hong y Michael P. Pound: *T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting*. Usa difusión con corrección semántica para conteo de categorías indicadas por texto; no es simplemente preguntar un número a CLIP. [Artículo](https://arxiv.org/abs/2502.20625). El [repositorio oficial](https://github.com/cha15yq/T2ICount) enlaza pesos del artículo, reproducción y logs; usa Stable Diffusion1.5 y corrige el subconjunto FSC-147-S a v2. No encontré licencia explícita en raíz/README. Su receta batch16 no acredita viabilidad en12GB. Hipótesis: sensibilidad al texto para distinguir objetos; no evidencia de robustez en manifestaciones densas.

**DCA-MoE, preprint2026.** Hao Wang: *Spatially Adaptive Cross-Layer Fusion and Density-Routed Experts for Crowd Counting*, v2 del1/9/2026. Combina DINOv3 congelado, fusión espacial y expertos de distintos contextos con supervisión DMCount; sí aborda crowd counting. El propio resumen reconoce resultados entre datasets mixtos y limitaciones de semillas/selección de métricas. [Artículo primario](https://arxiv.org/abs/2608.15213v2). El [texto completo](https://arxiv.org/html/2608.15213v2) describe ejecuciones de ocho GPUs; no verifiqué repositorio público, pesos, licencia de implementación ni reproducción en12GB. Es evidencia de investigación reciente, no candidato inmediatamente reproducible ni SOTA demostrado.

**Consecuencia para SDD:** conservar STNNet como control, sin convertir su antigüedad en criterio de elección. Antes de promover estos candidatos: fijar tarea —cabezas versus objetos—, disponibilidad legal/técnica, versión de pesos/dataset y preflight lote1. Los obstáculos documentados relegan su ejecución; no invalidan sus ideas. Ningún MAE entre protocolos distintos demuestra superioridad local, y ninguna novedad justifica reemplazar anotaciones completas por pseudoetiquetas no verificadas.

## Verificación dirigida adicional: candidatos recientes reproducibles

Ampliación solicitada posteriormente, 25/10/2026. La prioridad siguiente reemplaza el orden preliminar de candidatos; se basa en pertinencia de tarea y artefactos visibles, no en una comparación experimental local. El alcance bibliográfico se amplía por encargo explícito.

**Prioridad1: SR, CVPR2026, conteo por cuadro.** Weibo Shu y Antoni B. Chan, *Adapting Lightweight Image-based Counting Models for Video Crowd Counting*. El artículo define información espaciotemporal mediante un regularizador estadístico usado durante entrenamiento; mantiene inferencia de imagen sin módulos temporales adicionales. Incluye DRONECROWD y VSCROWD entre seis benchmarks. Esto lo vuelve especialmente pertinente para una comparación reciente con STNNet, pero no demuestra transferencia a la marcha. [Publicación CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Shu_Adapting_Lightweight_Image-based_Counting_Models_for_Video_Crowd_Counting_CVPR_2026_paper.html).

El [repositorio oficial](https://github.com/wbshu/SR) contiene código de entrenamiento/evaluación y [licencia MIT](https://raw.githubusercontent.com/wbshu/SR/main/LICENSE). Verifiqué el listado de [assets de release v1](https://github.com/wbshu/SR/releases/expanded_assets/v1): `dronecrowd.tar`,82,1MB, y modelos de otros cinco datasets. SHA-256 publicado para DroneCrowd: `22ff7ff7bfcfa6cb45b38ea0df6dd74f388acc070b41bbb2d64a7308062097eb`. Es metadata del servidor; no un hash recalculado ni validación del contenido. La portabilidad propuesta no prueba que su entrenamiento o inferencia nativa cumplan nuestros límites de12GB.

**Prioridad2: CLIP-EBC, ICME2025, densidad de personas.** Yiming Ma, Victor Sanchez y Tanaya Guha, *CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification*. Preprint inicial2024, manuscrito aceptado v3 de marzo2025. Combina clasificación por bloques con bins enteros y pérdida de regresión de densidad. Es un contador supervisado de multitudes basado en CLIP; no conteo genérico gratuito por prompt. [Artículo y referencia ICME](https://arxiv.org/abs/2403.09281).

[Código oficial](https://github.com/Yiming-M/CLIP-EBC), [MIT](https://raw.githubusercontent.com/Yiming-M/CLIP-EBC/main/LICENSE). La [release](https://github.com/Yiming-M/CLIP-EBC/releases) enumera pesos NWPU ResNet50/ViT-B16 y ShanghaiTech, además de logs. Hay variantes NWPU separadas por mejor MAE/RMSE: fijar una regla antes de comparar. El repositorio aparece archivado desde22/10/2026. La existencia de checkpoints y logs mejora auditabilidad; no acredita ejecución en12GB ni permite trasladar el ranking publicado a Buenos Aires.

**Prioridad3 condicional: MovingDroneCrowd, individuos únicos en video.** Yaowu Fan, Jia Wan, Tao Han, Antoni B. Chan y Andy J. Ma, *Video Individual Counting for Moving Drones*, ICCV2025: SDNet estima densidad compartida entre cuadros y deriva entradas/salidas. [Artículo CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Video_Individual_Counting_for_Moving_Drones_ICCV_2025_paper.html). La extensión2026 agrega Wanli Ouyang a la autoría, MovingDroneCrowd++, GD³A y tracking; no confundir sus particiones/resultados con la versión de conferencia. [Preprint2601.12500](https://arxiv.org/abs/2601.12500).

El [repositorio oficial](https://github.com/fyw1999/MovingDroneCrowd) enlaza nombres concretos de pesos SDNet/GD³A y conserva una release ICCV2025. Advierte que el SDNet disponible fue reentrenado porque los pesos originales quedaron inaccesibles. La [ficha oficial de pesos](https://huggingface.co/fyw1999/MovingDroneCrowd-Weights) indica **CC-BY-NC-4.0**; la vista de archivos no resultó accesible con esta herramienta. Son enlaces identificados y ficha comprobada, no binarios verificados. No encontré licencia explícita de código en raíz/README; no trasladar la de pesos automáticamente. Recursos12GB sin acreditar. Priorizarlo sólo si el objetivo pasa a individuos únicos y hay anotación temporal para evaluarlo; su MAE de video no es comparable con MAE por cuadro.

**Prioridad metodológica: P2RLoss, CVPR2025.** Wei Lin, Chenyang Zhao y Antoni B. Chan, *Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting*. Estudia sobreactivación de fondo en pseudoetiquetado y propone propagar confianza de puntos a regiones. Incluye evaluación semisupervisada y adaptación no supervisada. [Artículo](https://arxiv.org/abs/2505.21943). Es directamente relevante para revisar cómo pueden aparecer falsos positivos durante aprendizaje, pero no prueba que nuestro deterioro tenga su misma causa.

Hay [código oficial](https://github.com/Elin24/P2RLoss) con teacher/student, backbone VGG16BN y protocolo que selecciona imágenes etiquetadas. No verifiqué licencia explícita ni checkpoints publicados. Una pequeña proporción de imágenes exhaustivamente anotadas no equivale a pocas cabezas parciales en cada imagen. Mantenerlo como referencia de diseño; no iniciar entrenamiento para compensar la ausencia de pesos ni prometer compatibilidad12GB.

**Reverificación de DCA-MoE.** La URL `abs/2608.15213v2` devolvió `Cache miss`; la [URL sin sufijo](https://arxiv.org/abs/2608.15213) sí volvió a devolver título, autor y revisión v2 del1/9/2026. El HTML completo había sido accesible en la consulta anterior. La discrepancia de acceso queda registrada: no afirmar inexistencia por el fallo puntual, ni presentar disponibilidad reproducible de código/pesos. Su prioridad queda por debajo de SR/CLIP-EBC.

**Puerta de aceptación documental.** SR y CLIP-EBC pasan a la primera auditoría de implementación/pesos propuesta. Para cada uno: fijar commit, variante, archivo, preprocessing y protocolo; contrastar primero una muestra original con resultados reproducibles; después medir negativos/escala locales con anotaciones completas y test reservado. Para todos, disponibilidad pública, licencia explícita, compatibilidad y memoria son verificaciones distintas. No se descargó ni ejecutó ningún candidato. La revisión profesional debe poder reconstruir cada afirmación y corregirla sin defender una arquitectura por novedad, preferencia o resultado esperado.
