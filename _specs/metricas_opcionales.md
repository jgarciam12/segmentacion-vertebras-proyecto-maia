# Spec: Métricas Opcionales de Evaluación
**Proyecto:** Segmentación multiclase de vértebras en radiografías de columna  
**Estado:** Segunda iteración — implementar una vez que los modelos estén entrenados y haya resultados base que analizar.

---

## Contexto

Estas métricas complementan las obligatorias (Dice por clase, mIoU, AP@50 + mAP) definidas en `procesamiento.md`. No son bloqueantes para comparar modelos entre sí, pero agregan valor clínico y granularidad en la evaluación. Se implementan en una segunda iteración.

---

## AP@75

**Descripción:** Average Precision con umbral de IoU más estricto (≥ 0.75 entre predicción y ground truth para considerar una detección correcta).

**Por qué es útil:**
AP@50 puede aprobar detecciones con bordes relativamente imprecisos. AP@75 exige que la máscara predicha se ajuste mucho mejor al contorno real de la vértebra, lo cual es más relevante clínicamente.

**Cuándo priorizarlo:**
Si AP@50 es alto pero se observan visualmente bordes imprecisos en las máscaras predichas, AP@75 cuantifica exactamente cuánto se pierde al exigir mayor precisión de contorno.

| Modelo | Aplica |
|---|---|
| Mask R-CNN + ResNet50-FPN | Sí |
| Mask R-CNN + ResNet101-FPN | Sí |
| nnU-Net | No (segmentación semántica) |
| Mask R-CNN + EfficientNet-FPN | Sí |

---

## AP@[.5:.95]

**Descripción:** Promedio de AP calculado en 10 umbrales de IoU desde 0.50 hasta 0.95 con pasos de 0.05. Es el estándar oficial del benchmark COCO.

**Por qué es útil:**
Es la métrica más exigente y completa para modelos de instancia. No favorece ni penaliza un umbral específico sino que evalúa el rendimiento en un rango completo de precisión de contorno.

**Cuándo priorizarlo:**
Si se quiere publicar resultados comparables con la literatura de segmentación por instancia o participar en benchmarks.

| Modelo | Aplica |
|---|---|
| Mask R-CNN + ResNet50-FPN | Sí |
| Mask R-CNN + ResNet101-FPN | Sí |
| nnU-Net | No |
| Mask R-CNN + EfficientNet-FPN | Sí |

---

## HD95 — Hausdorff Distance 95th Percentile

**Descripción:** Mide la distancia máxima entre el borde predicho y el borde real de la vértebra, usando el percentil 95 en lugar del máximo absoluto para ser robusta a outliers.

```
Rango: píxeles (o mm si se conoce la escala física)
Valor ideal: 0  →  cuanto más bajo mejor
```

**Por qué es útil:**
Dice e IoU miden área de overlap pero no dicen nada sobre la precisión del borde. En aplicaciones clínicas como el cálculo del ángulo de Cobb, la posición exacta del contorno de cada vértebra impacta directamente en la medición. Un modelo con Dice alto pero HD95 alto puede ser inaceptable clínicamente.

**Por qué el percentil 95 y no el máximo absoluto:**
El máximo absoluto (HD100) es sensible a un único píxel mal predicho en el borde. HD95 descarta el 5% de los puntos más alejados, siendo más representativo del error real de contorno.

**Cuándo priorizarlo:**
Cuando el proyecto avance hacia validación clínica o cálculo automático del ángulo de Cobb. Es la métrica que más directamente refleja la utilidad real de las segmentaciones para un radiólogo.

| Modelo | Aplica |
|---|---|
| Mask R-CNN + ResNet50-FPN | Sí |
| Mask R-CNN + ResNet101-FPN | Sí |
| nnU-Net | Sí |
| Mask R-CNN + EfficientNet-FPN | Sí |

---

## Resumen de métricas opcionales por modelo

| Métrica | Modelo 1 | Modelo 2 | Modelo 3 | Modelo 4 | Prioridad |
|---|---|---|---|---|---|
| AP@75 | Sí | Sí | No | Sí | Media |
| AP@[.5:.95] | Sí | Sí | No | Sí | Baja |
| HD95 | Sí | Sí | Sí | Sí | Alta |

**Orden de implementación sugerido:** HD95 primero (mayor valor clínico), luego AP@75, finalmente AP@[.5:.95].

---

## Fine-tuning guiado con búsqueda de hiperparámetros (iteración futura)

**Contexto:**
Los modelos de la primera iteración se entrenan con hiperparámetros definidos manualmente (LR inicial, α/β de la loss, patience, etc.). Una vez obtenidos los resultados de esos modelos, existe la posibilidad de refinar los mejores usando los pesos ya entrenados como punto de partida.

**Estrategia propuesta — Warm Start + Random Search acotado:**

```
Fase 1 (ya hecha):
    Entrenamiento completo con hiperparámetros iniciales
    → se obtienen pesos entrenados + métricas + rango aproximado
      de hiperparámetros que funcionan

Fase 2 (iteración futura):
    Random Search en espacio reducido alrededor de los valores
    que funcionaron en la Fase 1
    → se parte de los pesos ya entrenados (no desde cero)
    → el entrenamiento es mucho más corto al estar ya casi convergido
    → se buscan valores más finos que mejoren marginalmente las métricas
```

**Por qué es viable con hardware limitado:**
A diferencia de un Grid Search desde cero sobre todo el espacio de hiperparámetros (prohibitivo), este enfoque acota la búsqueda a un vecindario pequeño de valores ya validados. El costo computacional es manejable porque cada entrenamiento de la Fase 2 parte de un modelo casi convergido.

**Cuándo activar esta estrategia:**
Cuando los modelos de la primera iteración muestren resultados prometedores pero con margen de mejora identificable en algún hiperparámetro específico. No tiene sentido hacer fine-tuning de hiperparámetros si el modelo base no aprendió nada útil.

**Hiperparámetros candidatos a explorar:**

| Hiperparámetro | Rango inicial | Espacio de búsqueda sugerido |
|---|---|---|
| LR base | 1e-3 | [5e-4, 1e-3, 2e-3] |
| α (peso Dice Loss) | 0.5 | [0.3, 0.5, 0.7] |
| β (peso Focal Loss) | 0.5 | [0.3, 0.5, 0.7] |
| γ (Focal Loss) | 2.0 | [1.0, 2.0, 3.0] |
| Patience early stopping | 7–10 | [5, 7, 10] |
