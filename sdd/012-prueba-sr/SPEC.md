# Prueba acotada del contador SR

Encargo del autor, 25/10/2026: probar un modelo posterior a STNNet antes de
continuar su ajuste, verificar pesos y estimar procesamiento de videos completos.

## Hipótesis y alcance

SR (CVPR 2026), checkpoint oficial DroneCrowd, puede ejecutarse con nuestro
entorno y aportar una respuesta distinta en las regiones ya seleccionadas.
Ejecutar pesos congelados no exige etiquetas completas; sin ellas no se afirmará
superioridad ni precisión. No entrenar ni combinar conteos de distintos planos.

Código oficial fijado: `wbshu/SR`, commit
`6c5f66c823dd86dcb1329f1b007ba369e622036b`.
Release v1, `dronecrowd.tar`, 86.123.541 bytes; SHA-256 publicado
`22ff7ff7bfcfa6cb45b38ea0df6dd74f388acc070b41bbb2d64a7308062097eb`.
Descarga y comprobación local obligatorias antes de cargar pesos.

## Diseño previo

- Inspeccionar arquitectura y evaluador oficiales. Instanciar directamente
  `VGG_Baysian_Ma(make_layers(..., batch_norm=True))` sólo si claves y formas
  del checkpoint confirman esa variante; no descargar otra inicialización.
- Carga CPU `weights_only=True`, estricta, sin excepciones de deserialización
  arbitraria; comprobar claves, tipos y finitud. Modelo en eval/inference_mode.
- RGB nativo, normalización ImageNet; registrar cualquier recorte de borde para
  múltiplos de 16. La arquitectura reduce 16 y amplía bilineal 2 con
  align_corners=True según `upsample_bilinear` oficial. Salida absoluta, stride 8.
- Integrar sumando directamente celdas de salida, sin el factor 100 de STNNet.
  Mantener mapas y hashes. Procesar una entrada por vez, sin workers ni pinning.
- Prueba sintética pequeña primero. Después inferir los ocho recortes preparados
  existentes, incluyendo dispersos, densos y fondos. Límite por tile: 576 por
  576, núcleo 256 por 256 y halo 160. No ampliar este límite automáticamente.
  Congelar lista/geométrica antes de obtener predicciones nuevas. Comparar con
  STNNet sólo cuando coincidan fuentes, región efectiva y contexto; de otro
  modo presentar respuestas por separado con diferencias explícitas.
- Medir latencia sincronizando CUDA y separar carga, cómputo y proceso total.
  No extrapolar velocidad de un recorte a 4K sin declarar incertidumbre. Si el
  ensayo pequeño pasa, probar uno o dos cuadros completos mediante mosaicos
  nativos de núcleos disjuntos alineados a 16 y halo suficiente documentado.
  El upsampling align_corners=True hace que el mosaico no sea equivalente a una
  pasada completa; declarar esa aproximación, no adjudicar exactitud al conteo.

## Recursos y límites

Temporales, fuentes descargadas y cachés en `~/home.codex/tmp`; pesos/mapas e
informes privados fuera de Git. No instalar en el venv compartido. Preflight
antes de descarga/inferencia: reserva 20 GiB disco y 2 GiB RAM disponibles.
GPU RTX 3060 compartida, lote 1, proceso limitado a 2 GiB RAM, swap 512 MiB y
120 segundos por ejecución. No aumentar límites automáticamente ante fallos.

## Aceptación

Hash oficial comprobado, estado estricto, prueba de geometría/integración y
revisión independiente antes de GPU; resultados finitos con fuente y código
trazables. Informar tiempos, memoria y límites. Conservar fallos como evidencia.
No servir originales ni publicar mapas/pesos/datos privados. Documentación y
código con commits atómicos y publicación en el fork autorizado.

La estimación de duración enumerará videos, cuadros y política de muestreo.
El conteo por instante y el número de individuos únicos son tareas distintas:
dividir clips no elimina duplicaciones entre zonas o tiempos. No sumar planos.

El autor precisó el objetivo durante la implementación: estimación simultánea
a una hora determinada, no participantes únicos acumulados. La cobertura espacial
y las diferencias temporales se declararán antes de sumar regiones disjuntas.

## Revisión y estado

Auditoría previa de fuentes: GPT-6 Astra. Confirmó suma directa, normalización,
riesgos de descarga automática, acumulación de mapas y workers del evaluador.
Al escribir esta SPEC, implementación e inferencia estaban pendientes. Después
se completaron cuatro pruebas, equivalencia CPU bit a bit en tres tamaños, ocho
controles y tres cuadros GPU. Todos dentro de cuotas; hash oficial verificado.
Conteo de SR se integra directamente. No se acreditó mejora local ni se realizó
el barrido de videos completos. [Informe](../../docs/prueba-sr-2026-10-25.md).
