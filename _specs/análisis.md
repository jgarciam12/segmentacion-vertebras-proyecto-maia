# Análisis y Estado del Arte
**Proyecto:** Segmentación multiclase de vértebras en radiografías de columna  
**Contexto:** Documento de exploración y debate técnico. Recoge las estrategias analizadas, arquitecturas evaluadas, ventajas, desventajas y conclusiones que guiaron las decisiones de diseño del proyecto.

---

## Definición del problema

Tarea de **segmentación multiclase por instancia** sobre radiografías de columna vertebral (con y sin escoliosis). El objetivo es identificar, delimitar y etiquetar cada vértebra individualmente (C3–C7, T1–T12, L1–L5) asignándole su etiqueta anatómica correcta.

### Distinción crítica: segmentación semántica vs por instancia

| Tipo | Descripción | Problema en este contexto |
|---|---|---|
| Semántica | Clasifica cada píxel por categoría, sin distinguir instancias | Si dos vértebras se tocan, las trata como un solo objeto |
| Por instancia | Detecta y delimita cada objeto individual por separado | Permite individualizar cada vértebra aunque estén en contacto |

**Conclusión:** La naturaleza del problema exige segmentación por instancia. Modelos de segmentación semántica pura (como U-Net estándar) presentan limitaciones cuando las vértebras se tocan o son poco visibles.

---

## Restricciones del problema que condicionan la arquitectura

1. **Dataset pequeño (250 imágenes):** transfer learning casi obligatorio para compensar.
2. **Precisión espacial fina requerida:** el modelo necesita mecanismos para recuperar detalle espacial (skip connections o equivalente).
3. **Orden anatómico estricto:** las vértebras tienen una secuencia vertical fija (C7 siempre sobre T1, L5 siempre bajo L4). El modelo debe capturar contexto global para no confundir vértebras adyacentes.
4. **Vértebras parcialmente visibles o en contacto:** uno de los retos principales. Se evaluará la estrategia de segmentar sin clasificar en estos casos.
5. **No todas las imágenes cubren la columna completa:** algunas radiografías son de segmento lumbar o torácico solamente. Las clases ausentes deben contemplarse en la evaluación.

---

## Arquitecturas analizadas

### 1. U-Net

**Descripción:** Arquitectura encoder-decoder con skip connections. Estándar de facto en segmentación biomédica semántica.

**Ventajas:**
- Muy eficiente y rápida de entrenar.
- Excelente para bordes definidos gracias a sus skip connections.
- Funciona bien con datasets pequeños.
- Amplia documentación en imágenes médicas.

**Desventajas:**
- Realiza segmentación semántica, no por instancia.
- Si dos vértebras se tocan, las trata como un único objeto.
- En un estudio comparativo sobre TC obtuvo DICE de 69.2%, muy inferior a Mask R-CNN.
- Requeriría post-procesamiento complejo para individualizar vértebras en contacto.

**Resultado de la evaluación:** Descartada como modelo principal. Sus skip connections y eficiencia son valiosas, pero la limitación de segmentación semántica es incompatible con el requisito de individualizar vértebras.

---

### 2. Mask R-CNN

**Descripción:** Framework de segmentación por instancia de dos etapas. Backbone + FPN + RPN + ROI Align + cabezas de bounding box y máscara.

**Ventajas:**
- Detecta y delimita cada vértebra como instancia independiente.
- Robusto ante vértebras en contacto o superpuestas.
- En estudios sobre TC alcanzó DICE del 99.9% superando significativamente a U-Net.
- Transfer learning disponible desde COCO.
- Arquitectura directamente alineada con el problema de segmentación por instancia.

**Desventajas:**
- Alta complejidad computacional.
- Tiempos de entrenamiento e inferencia más largos.
- Puede tener dificultades con objetos muy pequeños.
- Los resultados de DICE del 99.9% fueron en CT, no en radiografías 2D (diferente modalidad, mayor dificultad en X-ray por proyección y solapamiento).

**Resultado de la evaluación:** **Modelo principal seleccionado.** Se probarán dos variantes de backbone: ResNet50-FPN y ResNet101-FPN.

---

### 3. YOLOv8

**Descripción:** Detector de una sola etapa con capacidad de segmentación por instancia.

**Ventajas:**
- Velocidad de procesamiento superior, permite análisis en tiempo real.
- Buen equilibrio velocidad/precisión.
- Fine-tuning sobre COCO disponible.

**Desventajas:**
- Las máscaras de segmentación son menos precisas en bordes comparado con Mask R-CNN.
- Sensible a ruido y baja iluminación, condición frecuente en radiografías.
- La precisión clínica requerida en bordes de vértebras es incompatible con su nivel de detalle en máscaras.

**Resultado de la evaluación:** No seleccionado para este proyecto. Su ventaja principal (velocidad en tiempo real) no es el objetivo prioritario. La precisión de bordes es insuficiente para uso clínico.

---

### 4. nnU-Net

**Descripción:** Framework autoconfigurado de segmentación biomédica. Analiza el dataset y ajusta automáticamente preprocesamiento, topología de red e hiperparámetros.

**Ventajas:**
- Extremadamente robusto, ha superado enfoques especializados en múltiples desafíos médicos.
- Autoconfiguración elimina decisiones de diseño manuales.
- Ajusta la complejidad de la red al tamaño del dataset automáticamente.
- Considerado estándar de facto en segmentación biomédica automatizada.
- Más confiable para datasets pequeños precisamente por su autoajuste.

**Desventajas:**
- No realiza segmentación de instancias de forma nativa.
- Es una caja negra: no se puede intervenir en arquitectura, loss ni estrategia de entrenamiento.
- Internamente es segmentación semántica optimizada.

**Resultado de la evaluación:** Seleccionado como **baseline de referencia externo**. Al ser autoconfigurado provee un número de referencia honesto e independiente de decisiones de diseño manual.

---

### 5. Transformers (ViT, Swin Transformer, TransUNet, SpineFormer)

**Descripción:** Arquitecturas basadas en mecanismos de autoatención. Dividen la imagen en parches/tokens procesados secuencialmente. Se usan frecuentemente en arquitecturas híbridas CNN + Transformer.

**Ventajas:**
- Capturan dependencias de largo alcance y contexto global de la columna completa.
- Más robustos en casos de bajo contraste o degeneración discal.
- SpineFormer ha demostrado consistencia superior a nnU-Net en casos anatómicamente complejos.

**Desventajas:**
- Costo computacional muy elevado.
- Requieren datasets significativamente más grandes para generalizar correctamente.
- Pueden perder detalle espacial fino por la división en parches.
- Inviables con el hardware disponible (GTX 1650, 4GB VRAM) y el tamaño del dataset.

**Resultado de la evaluación:** No seleccionado para este proyecto. Escalón natural a explorar en una iteración futura con mayor dataset y hardware.

---

## Encoders evaluados

El encoder es el backbone que extrae features de la imagen. Se evaluaron tres candidatos para usarse dentro de Mask R-CNN.

### ResNet50

**Descripción:** Red residual de 50 capas. Backbone estándar de Mask R-CNN en su implementación original.

**Ventajas:**
- Muy probado en medical imaging y segmentación general.
- Buen balance entre capacidad y tamaño.
- ~25M parámetros, manejable con hardware limitado (4GB VRAM).
- Transfer learning desde COCO directamente compatible con Mask R-CNN.

**Desventajas:**
- Menos capacidad que ResNet101 para capturar features complejas.

**Resultado:** **Seleccionado como backbone principal (Modelo 1).** Mejor balance capacidad/datos para un dataset de 250 imágenes.

---

### ResNet101

**Descripción:** Red residual de 101 capas. Variante más profunda de ResNet50.

**Ventajas:**
- Mayor capacidad para capturar representaciones complejas.
- Puede captar mejor la variabilidad morfológica de las vértebras en escoliosis.

**Desventajas:**
- ~44M parámetros, casi el doble que ResNet50.
- Con 250 imágenes la profundidad adicional puede no justificarse y agregar ruido en el fine-tuning.
- Riesgo real de OOM con GTX 1650 (4GB VRAM).
- Podría comportarse igual o peor que ResNet50 por exceso de capacidad relativa al tamaño del dataset.

**Resultado:** **Seleccionado como backbone comparativo (Modelo 2).** Se ejecuta solo si los recursos lo permiten o en hardware externo (Colab/Kaggle).

---

### EfficientNet-B4

**Descripción:** Arquitectura que escala simultáneamente profundidad, ancho y resolución de forma balanceada. Más eficiente en parámetros que ResNet.

**Ventajas:**
- Mejor accuracy por parámetro que ResNet.
- Captura features multi-escala de forma nativa.
- Menor consumo de VRAM que ResNet101.
- Potencialmente mejor para datasets pequeños por su eficiencia paramétrica.

**Desventajas:**
- Integración con Mask R-CNN no es estándar (requiere adaptar la FPN a sus salidas).
- Menor documentación específica en contextos médicos comparado con ResNet.
- Mayor complejidad de implementación, más difícil de diagnosticar si falla.

**Resultado:** **Seleccionado como exploración opcional (Modelo 4).** Se evalúa solo si los modelos 1 y 2 no son satisfactorios.

---

### Comparativa de encoders

| Encoder | Parámetros | VRAM estimada | Dataset pequeño | Integración Mask R-CNN | Seleccionado |
|---|---|---|---|---|---|
| ResNet50 | ~25M | ~2.5–3.5GB | Bueno | Nativa y estándar | Sí (principal) |
| ResNet101 | ~44M | ~3.5–4.5GB | Moderado | Nativa y estándar | Sí (comparativa) |
| EfficientNet-B4 | ~19M | ~2.5–3GB | Muy bueno | Requiere adaptación | Sí (opcional) |

---

## Estrategias de entrenamiento evaluadas

### Estrategia de Loss

#### Loss consideradas

**Cross-Entropy sola — Descartada**
- Descartada como loss única porque con el desbalance de clases existente (mayoría de píxeles es background) el modelo aprende a predecir background con alta confianza sin aprender las clases de vértebras.

**Dice Loss**
- Optimiza directamente el overlap entre predicción y ground truth por clase.
- Ataca el desbalance background/foreground de forma natural.
- No se ve tan afectada por la dominancia del background.

**Focal Loss**
- Variante de Cross-Entropy que pone más peso en los ejemplos difíciles (vértebras pequeñas, bordes ambiguos, clases poco representadas).
- Parámetro γ (gamma) controla el enfoque en ejemplos difíciles. Valor inicial sugerido: γ = 2.0.

**Loss combinada — Seleccionada**
```
Loss = α · Dice Loss + β · Focal Loss
```
- Combina las fortalezas de ambas: Dice ataca el desbalance global, Focal ataca los casos difíciles.
- α y β son hiperparámetros. Punto de partida: α = β = 0.5.
- Aplica a Modelos 1, 2 y 4. nnU-Net gestiona su propia loss internamente.

---

### Estrategia de descongelamiento progresivo con LR diferenciales

#### Problema que resuelve

Con 250 imágenes, entrenar todo el modelo desde el inicio destruye los pesos preentrenados antes de que el decoder aprenda algo útil (catastrophic forgetting). Si se congela el encoder permanentemente, el modelo nunca se adapta a las características específicas de las radiografías de columna.

#### Estrategia seleccionada: descongelamiento por fases guiado por el loss

El entrenamiento se divide en fases. En cada fase se descongela un bloque adicional del encoder, monitoreando el validation loss como criterio de avance.

```
Fase 1 — Encoder 100% congelado
  Qué aprende:  Decoder + RPN + cabezas (bounding box y máscara)
  Razonamiento: el decoder necesita aprender a interpretar las features
                del encoder antes de que tenga sentido modificar el encoder

Fase 2 — Descongelar último bloque del encoder
  Qué aprende:  Decoder + features de alto nivel del encoder
  Razonamiento: las capas más profundas del encoder capturan semántica
                de alto nivel, más relevante para adaptar al dominio radiológico

Fase 3 — Descongelar bloques anteriores progresivamente
  Qué aprende:  Decoder + features de menor nivel
  Razonamiento: las capas tempranas capturan bordes y texturas genéricas,
                que ya son útiles tal como están y necesitan menos ajuste
```

#### Learning rates diferenciales

Al descongelar cada bloque nuevo, no se entrena con el mismo LR que el decoder. Las capas preentrenadas tienen representaciones valiosas que un LR alto destruiría:

| Sección | LR relativo | Razón |
|---|---|---|
| Decoder + cabezas | LR base (1e-3) | Aprende desde cero, necesita LR alto |
| Encoder bloque 4 (último) | LR base × 0.1 | Features de alto nivel, ajuste fino |
| Encoder bloque 3 | LR base × 0.01 | Features intermedias, ajuste muy fino |
| Encoder bloque 2 | LR base × 0.001 | Features genéricas, casi no tocar |
| Encoder bloque 1 (primero) | LR base × 0.0001 | Detectores de bordes básicos, no modificar |

---

### Estrategias de early stopping evaluadas

#### Early stopping global simple — Descartado para este problema

Detener el entrenamiento si no mejora en N epochs. Descartado como estrategia única porque no considera las distintas fases del descongelamiento progresivo. Al descongelar una nueva capa, el loss puede subir temporalmente antes de mejorar, lo cual dispararía un early stopping global prematuramente.

#### Early stopping por fase — Seleccionado

Cada fase tiene su propio criterio de early stopping independiente. Cuando una fase dispara el early stopping, se avanza a la siguiente fase (o se termina el entrenamiento si es la última):

```
Patience por fase: 7–10 epochs sin mejora en val loss
```

#### ReduceLROnPlateau como paso previo al early stopping

Antes de disparar el early stopping, reducir el LR para dar al modelo una oportunidad de salir del plateau sin necesidad de parar:

```
Patience:  3 epochs sin mejora en val loss
Factor:    0.5 (reduce LR a la mitad)
```

El flujo por fase queda:
```
Plateau detectado (3 epochs)
    → Reducir LR × 0.5
    → Si sigue sin mejorar (7–10 epochs desde el plateau original)
        → Early stopping de la fase
        → Avanzar a siguiente fase o terminar
```

#### Métrica de monitoreo

- **Accuracy — Descartada:** con el desbalance de clases, un modelo que prediga todo como background tiene accuracy artificialmente alta sin aprender nada.
- **Validation Loss — Seleccionada como criterio primario** de early stopping y checkpoint.
- **Mean Dice — Seleccionado como métrica secundaria** de referencia para verificar que el modelo está aprendiendo clases minoritarias (vértebras cervicales) y no solo las mayoritarias (lumbares).

#### Model checkpointing

Guardar el mejor modelo según val loss en cada fase, no el último epoch. El último epoch casi nunca es el mejor.

---

## Restricciones de hardware y su impacto en las decisiones

| Componente | Especificación | Impacto en decisiones |
|---|---|---|
| GPU | GTX 1650 (4GB VRAM) | Batch size = 1, FP16 obligatorio, ResNet101 en riesgo de OOM |
| CPU | Ryzen 5 5600X | Sin limitaciones relevantes para entrenamiento |
| RAM | 16GB | Suficiente para data loading y augmentation |

**Mixed precision (FP16):** obligatorio para todos los modelos en este hardware. Reduce consumo de VRAM casi a la mitad sin pérdida significativa de precisión.

**Epochs totales estimados con estas restricciones:** 30–40 epochs por modelo, aproximadamente 1–3 horas por modelo con batch size 1 y FP16 activo.

**Alternativa para modelos que excedan VRAM:** Google Colab o Kaggle (GPU T4, 16GB VRAM, gratuito).

---

## Comparativa de modelos evaluados

| Modelo | Tipo segmentación | Dataset pequeño | VRAM requerida | Precisión bordes | Seleccionado |
|---|---|---|---|---|---|
| U-Net | Semántica | Muy bueno | Baja | Alta | No |
| Mask R-CNN | Instancia | Moderado | Media-Alta | Muy alta | Sí (principal) |
| YOLOv8 | Instancia | Bueno | Media | Media | No |
| nnU-Net | Semántica optimizada | Muy bueno | Variable (auto) | Alta | Sí (baseline) |
| Transformers | Semántica/Instancia | Malo | Muy alta | Media | No |

---

## Conclusión de la fase de análisis

El problema requiere segmentación por instancia para poder individualizar vértebras en contacto o con baja visibilidad. Esto descarta U-Net semántico como modelo principal. Mask R-CNN es la arquitectura más directamente alineada con el problema. nnU-Net se incorpora como baseline de referencia por su robustez en datasets pequeños. Los Transformers quedan como escalón futuro si se amplía el dataset y el hardware lo permite.

---

## Consideraciones pendientes de resolver

1. **Vértebras parcialmente visibles o no visibles:** evaluar estrategia de segmentar sin clasificar (asignar clase genérica "vértebra no identificable") para estos casos.
2. **IDs Entity (23–35):** presentes en 7 imágenes Normal. Decisión provisional: mapear a background. Confirmar antes de implementar.
3. **Radiografías de columna parcial:** las métricas por clase deben excluir clases ausentes legítimamente, no penalizarlas como errores.
4. **Orden secuencial de vértebras:** las CNN no modelan explícitamente el orden anatómico. Estrategia a definir: post-procesamiento con restricciones anatómicas o incorporación en arquitectura en iteración futura.
