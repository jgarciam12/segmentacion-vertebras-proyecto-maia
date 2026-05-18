# Spec: Procesamiento y Entrenamiento
**Proyecto:** Segmentación multiclase de vértebras en radiografías de columna  
**Estado:** Borrador — listo para implementación

---

## Resumen de modelos a entrenar

Se entrenan 4 modelos bajo el mismo protocolo base (con excepción de nnU-Net que es autoconfigurado):

| # | Modelo | Backbone | Rol | Prioridad |
|---|---|---|---|---|
| 1 | Mask R-CNN | ResNet50-FPN | Línea base principal | Alta |
| 2 | Mask R-CNN | ResNet101-FPN | Comparativa de capacidad | Media |
| 3 | nnU-Net | Autoconfigurado | Baseline de referencia externo | Alta |
| 4 | Mask R-CNN | EfficientNet-FPN | Exploración opcional | Baja |

**Orden de ejecución recomendado:** Modelo 1 → Modelo 3 → Modelo 2 → Modelo 4.  
Modelo 2 solo si Modelo 1 muestra resultados insuficientes. Modelo 4 como exploración final.

---

## Restricciones de hardware

| Componente | Especificación |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 (4GB VRAM) |
| CPU | AMD Ryzen 5 5600X 6-Core @ 3.70GHz |
| RAM | 16GB |
| Almacenamiento | 447GB SSD |

**Implicaciones:**
- Batch size máximo: **1** (puede intentarse 2 con FP16 activo).
- Mixed precision (FP16) **obligatorio** para no exceder VRAM.
- Resolución de entrada: máximo **512px** en el lado largo.
- Modelo 2 (ResNet101) tiene riesgo real de OOM. Si ocurre, correrlo en Google Colab o Kaggle (GPU T4, 16GB VRAM gratuita).

---

## Arquitectura de los modelos Mask R-CNN (Modelos 1, 2 y 4)

```
Input (radiografía preprocesada, 1 canal replicado a 3)
        |
[Backbone preentrenado]  ← ResNet50 / ResNet101 / EfficientNet
        |
[Feature Pyramid Network (FPN)]  ← captura multi-escala
        |
[Region Proposal Network (RPN)]  ← propone regiones candidatas
        |
[ROI Align]  ← extrae features por región propuesta
        |
    ┌───┴───┐
[Bbox head] [Mask head]  ← cabezas de bounding box y máscara
        |
Output: máscara por instancia + clase (23 clases: background + C3–C7 + T1–T12 + L1–L5)
```

**Canales de entrada:** 1 canal (escala de grises replicado a 3 para compatibilidad con encoder ImageNet).  
**Canales de salida:** 23 clases.  
**Activación de salida:** Softmax (clases mutuamente excluyentes).  
**Transfer learning:** pesos preentrenados en COCO.

---

## Loss combinada

Se utiliza una loss combinada para atacar simultáneamente el desbalance de clases y los ejemplos difíciles:

```
Loss = α · Dice Loss + β · Focal Loss
```

| Componente | Función | Parámetros iniciales |
|---|---|---|
| Dice Loss | Ataca el desbalance background/foreground, optimiza overlap por clase | α = 0.5 |
| Focal Loss | Pone más peso en ejemplos difíciles (vértebras pequeñas, bordes ambiguos) | β = 0.5, γ = 2.0 |

**Nota:** α y β son hiperparámetros a ajustar según resultados. Punto de partida: α = β = 0.5.

**Aplicación:** Solo para Modelos 1, 2 y 4. nnU-Net gestiona su propia loss internamente.

---

## Estrategia de entrenamiento: descongelamiento progresivo con LR diferenciales

### Concepto

El entrenamiento se divide en fases. En cada fase se descongela un bloque adicional del encoder, aplicando learning rates decrecientes hacia las capas más tempranas para evitar la destrucción de los pesos preentrenados (catastrophic forgetting).

### Fases de entrenamiento

```
Fase 1 — Encoder 100% congelado
  Qué aprende:  Decoder + RPN + cabezas de bounding box y máscara
  Epochs máx:   15
  LR decoder:   1e-3

Fase 2 — Descongelar último bloque del encoder
  Qué aprende:  Decoder + bloque 4 del encoder
  Epochs máx:   10
  LR decoder:   1e-3
  LR bloque 4:  1e-4

Fase 3 — Descongelar penúltimo bloque del encoder
  Qué aprende:  Decoder + bloques 3 y 4 del encoder
  Epochs máx:   8
  LR decoder:   1e-3
  LR bloque 4:  1e-4
  LR bloque 3:  1e-5

  (continuar según plateau del loss)

Total epochs estimado: 30–40 epochs reales
```

### Learning rates diferenciales por capa

| Sección de la red | LR relativo |
|---|---|
| Decoder + cabezas | LR base (1e-3) |
| Encoder bloque 4 (último) | LR base × 0.1 |
| Encoder bloque 3 | LR base × 0.01 |
| Encoder bloque 2 | LR base × 0.001 |
| Encoder bloque 1 (primero) | LR base × 0.0001 |

**Principio:** las capas más cercanas al input tienen representaciones más genéricas y valiosas. Entrenarlas con LR alto destruye el conocimiento preentrenado.

---

## Estrategia de early stopping y LR scheduling

### Métrica de monitoreo

| Métrica | Rol |
|---|---|
| Validation Loss | Criterio primario de early stopping y checkpoint |
| Mean Dice | Métrica secundaria de referencia (detecta si el modelo aprende clases minoritarias) |

**Nota:** Accuracy descartado como métrica de monitoreo. Con el desbalance de clases existente (mayoría de píxeles es background), un modelo que prediga todo como background tendría accuracy artificialmente alta sin haber aprendido nada útil.

### ReduceLROnPlateau

Antes de disparar el early stopping, reducir el LR para dar al modelo oportunidad de salir de un plateau:

```
Patience:  3 epochs sin mejora en val loss
Factor:    0.5 (reduce LR a la mitad)
```

### Early stopping por fase

```
Patience por fase:  7–10 epochs sin mejora en val loss
Acción:             parar la fase actual y avanzar a la siguiente
                    (o detener el entrenamiento si es la última fase)
```

### Model checkpointing

Guardar el mejor modelo según **val loss** en cada fase, no el último epoch.

```
Guardar cuando:  val loss mejora respecto al mejor anterior
Qué guardar:     pesos completos del modelo en ese punto
```

---

## nnU-Net (Modelo 3) — consideraciones especiales

nnU-Net es un framework autocontenido. No se interviene en:
- Arquitectura interna
- Loss function
- Learning rates
- Estrategia de entrenamiento

**Lo que sí se controla:**
- Formato y estructura del dataset de entrada (debe cumplir el formato nnU-Net).
- Conversión de las máscaras al formato requerido.
- Configuración de modalidad (radiografía 2D).

**Rol en el proyecto:** baseline de referencia externo. Provee un número de comparación honesto e independiente de cualquier decisión de diseño manual.

---

## Resumen comparativo de configuración por modelo

| Parámetro | Modelo 1 | Modelo 2 | Modelo 3 | Modelo 4 |
|---|---|---|---|---|
| Backbone | ResNet50-FPN | ResNet101-FPN | Autoconfigurado | EfficientNet-FPN |
| Transfer learning | COCO | COCO | Interno | ImageNet/COCO |
| Loss | Dice + Focal | Dice + Focal | Autoconfigurada | Dice + Focal |
| Descongelamiento progresivo | Sí | Sí | No | Sí |
| LR diferenciales | Sí | Sí | No | Sí |
| Early stopping | Por fase | Por fase | Interno | Por fase |
| Epochs estimados | 30–40 | 30–40 | Autoconfigurado | 30–40 |
| Riesgo OOM (4GB VRAM) | Bajo | Alto | Bajo | Bajo |
| Batch size | 1 | 1 | Auto | 1 |
| Mixed precision (FP16) | Obligatorio | Obligatorio | Automático | Obligatorio |

---

## Consideraciones pendientes de resolver antes de implementar

1. **Conversión de máscaras semánticas a instancia:** las máscaras ground truth son semánticas (ID de clase por píxel). Mask R-CNN necesita máscaras de instancia. La conversión es directa dado que cada clase es una instancia única por imagen, pero debe verificarse en el pipeline del data loader.
2. **Vértebras parcialmente visibles:** definir si se segmentan sin clasificar o se excluyen del ground truth. Impacta directamente en el formato de las anotaciones.
3. **IDs Entity (23–35):** mapear a background (ID 0) como decisión provisional. Confirmar antes de generar el dataset final.
4. **Stats de normalización:** calcular media y desviación estándar del propio dataset si no se usan stats de ImageNet. Pendiente de confirmar según modelo preentrenado elegido.
