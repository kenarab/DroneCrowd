# Plan de continuación: prueba con los videos aportados

Entrada: [SPEC](SPEC.md). Responsable: nueva sesión de DroneCrowd.
[Traspaso operativo](../../docs/traspaso-video-marcha-2026-10-24.md).

- [x] Conservar renombre y pines del autor en commits separados.
- [x] Preparar AGENTS/CLAUDE, exclusiones y lanzador ml.
- [x] Comprobar ml/CUDA y núcleo STNNet con tensores sintéticos en CPU y GPU.
- [x] Localizar los videos aportados en ~/home.agents/data/marchaUniversitaria241002.
- [x] Inventariar los tres videos y verificar metadatos, muestras decodificadas y hashes.
- [x] Reconsultar recursos y preflight Nexo tras liberar los mapas históricos.
- [ ] Verificar y obtener una variante de pesos entrenados dentro del presupuesto.
- [x] Implementar decodificación/muestreo/normalización y carga estricta de pesos.
- [x] Probar límites/checkpoints incompatibles con fixtures sintéticos.
- [ ] Ejecutar máximo30 fotogramas de un video durante máximo120s.
- [ ] Revisar fotogramas/mapas/conteos, registrar recursos, limitaciones y fallos.
- [ ] Entregar recomendación al autor sobre continuar y conservar/retirar datos.

Orden de commits sugerido: entrada y muestreo; carga e inferencia; reporte visual
y evidencia de aceptación. Cada incremento incluye su prueba relevante y debe
funcionar de manera independiente. Si se necesitan dependencias, registrar su
entorno propio como incremento separado; no alterar ml.

No repetir búsquedas de copias antiguas mientras el autor aporta los archivos.
No iniciar entrenamiento ni limpiar el dataset como requisito implícito.
No ejecutar otra sesión coordinadora desde Nexo: el autor abre la sesión.

24/10: nueve fotogramas reales preparados; hashes RGB reproducibles en una
segunda extracción. Metadatos, timestamps y límites kernel en los manifiestos
privados bajo `outputs/video-breve/20261024-validado/`. La fuente oficial de
pesos responde cuota excedida; no se ejecutó inferencia entrenada.
