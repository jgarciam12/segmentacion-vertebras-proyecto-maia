# Spec: Preprocesamiento
**Proyecto:** Segmentación multiclase de vértebras en radiografías de columna  
**Estado:** Cerrado

---

## Contexto del dataset

| Atributo | Detalle |
|---|---|
| Total de imágenes | 250 |
| Imágenes Normal | 71 (`N_*.jpg`) |
| Imágenes Escoliosis | 179 (`S_*.jpg`) |
| Ground truth principal | `LabelMultiClass_ID_PNG/` — máscaras 16-bit PNG (`uint16`, modo `I;16`) |
| Referencia de etiquetas | `labels_dictionary.json` |
| Índice del dataset | `dataset_index.csv` |

**Distribución de clases anatómicas (IDs 0–22):**

| Región | Vértebras | IDs |
|---|---|---|
| Background | — | 0 |
| Cervical | C3–C7 | 1–5 |
| Torácica | T1–T12 | 6–17 |
| Lumbar | L1–L5 | 18–22 |

**IDs problemáticos (23–35):** etiquetados como `Entity X` sin semántica anatómica clara. Presentes en 7 imágenes del grupo Normal. Tratamiento: mapear a background (ID 0) salvo decisión contraria posterior.

---

## Consideraciones previas al pipeline

- Las imágenes de **radiografías son RGB** aunque su contenido es monocromático; algunas presentan tintes azulados o amarillentos según el equipo de origen.
- Las **máscaras multiclase ID son 16-bit** (`uint16`). Abrirlas sin especificar el modo correcto puede truncar o corromper los IDs de clase. Usar `PIL` con manejo explícito o `cv2.imread(..., cv2.IMREAD_UNCHANGED)`.
- **No todas las imágenes cubren la columna completa.** Algunas radiografías son de segmento lumbar o torácico solamente. Las clases ausentes deben contemplarse en la loss y en las métricas.
- Existe una **brecha de resolución significativa** entre grupos: las imágenes Normal son considerablemente más pequeñas (~200×900px) que las de Escoliosis (hasta ~2547×4156px).
- El **dataset_index.csv** provee el mapeo completo imagen → todas sus máscaras, usar como fuente canónica para el data loader.

---

## Pipeline de preprocesamiento

### Paso 1 — Conversión a escala de grises

**Aplica a:** imagen radiográfica únicamente (las máscaras no se tocan).

- Usar conversión perceptual de luminosidad:  
  `L = 0.299·R + 0.587·G + 0.114·B`
- Esto colapsa correctamente los tintes de color a una intensidad equivalente en gris, manteniendo coherencia entre imágenes de distintos equipos.
- Resultado: imagen monocanal. Puede mantenerse como 1 canal o replicarse a 3 canales según requiera la arquitectura del modelo.

### Paso 2 — ROI crop (Region of Interest)

**Aplica a:** imagen radiográfica + todas las máscaras.

- Usar la **máscara binaria** (`LabelBinaryJPG/`) a resolución original para calcular el bounding box de la columna vertebral.
- Aplicar un **margen generoso** (sugerido: ≥10% del tamaño del bounding box en cada lado) para no cortar vértebras en casos de escoliosis severa con desplazamiento lateral pronunciado.
- Recortar imagen y máscaras con las mismas coordenadas de crop.
- Objetivo: eliminar background irrelevante (tejidos blandos, pulmones, pelvis), reducir el desbalance de clases artificial y enfocar el modelo en la región anatómica de interés.
- **Razón del orden:** recortar antes de redimensionar garantiza que el resize trabaje únicamente sobre la región relevante, aprovechando toda la resolución disponible en la columna y no desperdiciándola en background que se descartará igual.

### Paso 3 — Estandarización de resolución

**Aplica a:** imagen radiográfica + todas las máscaras asociadas (ya recortadas).

- Definir un tamaño fijo de entrada que preserve la relación de aspecto vertical de la columna. Se recomienda un formato rectangular tipo **512×1024** o **384×768** (ancho × alto) en lugar de cuadrado, para no aplanar la geometría anatómica.
- Interpolación para la imagen: **bilinear**.
- Interpolación para todas las máscaras: **nearest neighbor** (nunca bilinear ni cúbica — mezclaría IDs de clase).
- Aplicar el mismo resize a imagen y a todas sus máscaras correspondientes en el mismo paso.

### Paso 4 — CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Aplica a:** imagen radiográfica únicamente.

- Aplicar CLAHE sobre la imagen en escala de grises ya recortada.
- Mejora el contraste local sin saturar zonas ya brillantes, resaltando los bordes de las vértebras.
- Parámetros sugeridos como punto de partida: `clipLimit=2.0`, `tileGridSize=(8,8)`. Ajustar según inspección visual.

### Paso 5 — Augmentation

**Aplica a:** imagen radiográfica + máscara multiclase ID (sincronizadas).

Todas las transformaciones geométricas deben aplicarse **de forma sincronizada** a imagen y máscara usando la misma seed o un pipeline que procese ambas en el mismo paso (ej. `albumentations`).

| Transformación | Permitida | Observación |
|---|---|---|
| Flip horizontal | Sí | La columna es anatómicamente simétrica en el plano frontal |
| Flip vertical | **No** | Invertiría el orden anatómico (L5 quedaría encima de C7) |
| Rotación | Sí, con límite | Máximo ±10°. Simula variabilidad de posicionamiento del paciente. No debe resultar en imagen completamente horizontal |
| Escala aleatoria (zoom) | Sí | Simula variabilidad de tamaño de pacientes |
| Deformación elástica | Sí | Especialmente útil para simular variabilidad de curvaturas en escoliosis |
| Ajuste de brillo/contraste | Sí (solo imagen) | Simula variabilidad de equipos radiográficos |
| Traslación | Sí, leve | Dentro de márgenes que no expulsen la columna del frame |

### Paso 6 — Estandarización

**Decisión confirmada:** se usan modelos preentrenados (Mask R-CNN con backbones ResNet50, ResNet101 y EfficientNet preentrenados en COCO/ImageNet). La estandarización utiliza los **stats de ImageNet**:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

**Consideración sobre canales:** las radiografías son monocanal (escala de grises) pero los backbones preentrenados esperan 3 canales. El canal único se replica a 3 canales antes de aplicar la estandarización, manteniendo compatibilidad con los pesos preentrenados.

**¿Por qué stats de ImageNet y no del propio dataset?**
Los backbones preentrenados aprendieron sus pesos asumiendo esa distribución de entrada. Usar stats del propio dataset desalinearía la distribución esperada por el encoder y degradaría la calidad del transfer learning.

**nnU-Net:** maneja su propia normalización internamente de forma autoconfigurada. No se aplica este paso para ese modelo.

**Las máscaras no se normalizan bajo ningún concepto.** Cada valor de píxel es un ID de clase, no una intensidad. Normalizar destruiría la identidad de las etiquetas.

---

## Consideraciones pendientes (a resolver en etapa de modelo/métricas)

1. **Vértebras parcialmente visibles o no visibles:** uno de los retos del proyecto es segmentar vértebras aun cuando no son completamente visibles en la radiografía. Una opción a evaluar es segmentarlas pero no clasificarlas (asignarles una clase genérica de "vértebra no identificable" en lugar de su etiqueta específica). Requiere revisar el ground truth disponible para estos casos.

2. **Radiografías de columna parcial:** dado que no todas las imágenes cubren la columna completa, las métricas por clase deben contemplar la ausencia legítima de ciertas vértebras en un subconjunto de imágenes. Ignorar esta distinción en la evaluación produciría métricas artificialmente bajas para clases que simplemente no están presentes en esa imagen.

---

## Resumen del pipeline final

```
Imagen RAW + Máscaras
        |
[1] Conversión a escala de grises (conversión perceptual L = 0.299R + 0.587G + 0.114B)
        |
[2] ROI crop con máscara binaria a resolución original (con márgenes generosos)
        |
[3] Resize estandarizado sobre ROI (bilinear para imagen, nearest neighbor para máscaras)
        |
[4] CLAHE (solo imagen)
        |
[5] Augmentation sincronizada imagen-máscara
        |
[6] Estandarización con stats ImageNet (mean/std) — replicar canal gris a 3 canales previo
        |
Tensor listo para el modelo
```
