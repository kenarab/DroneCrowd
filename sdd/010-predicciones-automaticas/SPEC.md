# Ejecutar y mostrar las propuestas automáticas del modelo actual

Encargo del autor25/10/2026: ejecutar el clasificador actual sin esperar a completar
las etiquetas. El original ya produjo mapas en19regiones. Mostrar también sus
propuestas espaciales, sin convertir salidas del modelo en anotaciones verdaderas.

## Procedimiento previo a ejecución

Reutilizar los mapas originales verificados de las7regiones anteriores y12nuevas.
Comprobar checkpoint, escala nativa, variante de densidad, hash de controles,
identidad de fuente/recorte y hash/forma finita de cada NPY. No es necesario inferir
otra vez para aplicar el postprocesado determinista a esos mismos mapas.

Usar el procedimiento `calc_denpt` de `STNNet/mytest.py` sin calibración nueva:
mapa/100, negativos truncados a cero, máximos/mínimos locales de8celdas, contraste
mayor que0,001, componentes conexos y centro de su caja, coordenadas multiplicadas
por2. Extraer primero sobre todo el contexto y después filtrar al ROI semiabierto.
Conservar por separado integral regional y cantidad de máximos: no son equivalentes.

Los puntos automáticos son candidatos propuestos por densidad, sin probabilidad
calibrada. Dibujarlos con color distinto de las marcas humanas/agentes y guardar
su procedencia. No mezclarlos con GT ni utilizarlos para entrenar automáticamente.
Las12regiones sin etiquetas continúan pendientes; falta de marcas no significa cero.

## Corrección y área de interés

Crear v4 sin modificar v2/v3. El autor identifica otra persona entre N3 y N4 por
su sombra; registrar presencia con centro de cabeza sin resolver, en un campo
separado de los puntos de cabeza. No inventar una coordenada precisa para entrenar.
El autor cuestionó test-peaton-fuente; conservar la discrepancia y mostrar el recorte
con su nombre dentro de la imagen para evitar confundir regiones consecutivas.

Objetivo local: personas en calle, veredas y plazas, excluyendo terrazas y otras
áreas fuera de la manifestación observada. La máscara de área de interés es una
decisión explícita adicional: detectar una persona y contarla dentro del área
elegida son operaciones distintas. No sustituir una máscara por marcar como fondo
a una persona real en un techo. Esta entrega no afirma disponer ya de una máscara
semántica validada ni la aplica silenciosamente a conteos anteriores.

## Entrega y aceptación

Vista privada versionada `revision/automatico-v4/` con imagen original, marcas de
referencia, candidatos automáticos y mapa de densidad. Nombres dentro de cada PNG,
fuente/resolución, números claramente separados y leyendas sobre incertidumbre.
HTML estático sin JavaScript; PNG<4MiB, texto<512KiB. Comparación de la extracción
con el código histórico en pruebas significativas, revisión independiente y GET
HTTPS autenticado. CPU, sin nuevos pesos ni entrenamiento ni mapas HDF5.

Las propuestas automáticas permiten revisar más ejemplos sin etiquetarlos antes.
Su visualización o coincidencia con nuestras marcas no certifica precisión.

## Estado

Ejecutado CPU sobre19mapas originales verificados. Vista privada y portada
actualizadas, versiones anteriores conservadas. Cuatro pruebas aprobadas,
incluida paridad con el código histórico en bordes, mesetas y ruido. El revisor
independiente Codex (modelo no identificado) recomputó los19conjuntos de máximos
y sus integrales, verificó hashes e inspeccionó figuras01/06/17. Sin bloqueantes.
ImplementaciónLuna y ajustes del coordinador; GET HTTPS autenticados200.
Continúan pendientes las etiquetas nuevas y la máscara de suelo; ninguna
propuesta automática se incorporó a entrenamiento o referencia verdadera.
