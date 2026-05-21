# RadiographMetrics_Cobb_Essential

Esta versión del dataset está organizada para trabajar únicamente con los elementos necesarios para describir la curva escoliótica en pixeles y el ángulo de Cobb.

## Contenido

- `metricas_cobb_resumen.csv`: tabla principal por paciente.
- `curvas_en_pixeles/`: un archivo CSV por paciente con todos los puntos de la curva en pixeles.
- `overlays_cobb/`: un archivo PNG por paciente con el sobrelape de la radiografía, la curva y las rectas perpendiculares del ángulo de Cobb.
- `diccionario_metricas_cobb.json`: descripción explícita de cada campo.

## Qué contiene la tabla principal

Para cada paciente, la tabla incluye únicamente:

- punto de inflexión superior
- punto de inflexión inferior
- pendiente de la recta superior
- pendiente de la recta inferior
- punto del ápice
- valor del ángulo de Cobb
- ruta al archivo con los puntos de la curva en pixeles
- ruta al overlay con la visualización del Cobb

## Formato de los puntos

Los puntos están expresados en pixeles sobre el sistema de coordenadas de la imagen.

- `x_px`: coordenada horizontal
- `y_px`: coordenada vertical

## Archivos de curva

Cada archivo dentro de `curvas_en_pixeles/` contiene dos columnas:

- `x_px`
- `y_px`

Cada fila representa un punto de la curva escoliótica.

## Archivos de overlay

Cada archivo dentro de `overlays_cobb/` muestra:

- la radiografía
- la curva escoliótica
- las rectas perpendiculares usadas para describir el ángulo de Cobb

## Convención usada

- El punto de inflexión superior corresponde al punto ubicado más arriba en la imagen.
- El punto de inflexión inferior corresponde al punto ubicado más abajo en la imagen.
- Las pendientes se expresan con la misma convención del código original: `dx/dy`.
