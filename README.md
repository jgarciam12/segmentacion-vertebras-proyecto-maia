# MAIA — Segmentación de Vértebras en Radiografías de Columna

Sistema de segmentación multiclase de vértebras en radiografías de columna vertebral. Combina un detector YOLOv8 con un segmentador MedSAM fine-tuneado para identificar y delinear individualmente las vértebras T1–T12 y L1–L5, con soporte para columnas normales y casos de escoliosis.

---

## Arquitectura del sistema

```
Radiografía (JPEG/PNG)
        |
  [YOLOv8 Detector]         → bounding boxes por vértebra + confidence score
        |
  [MedSAM Segmentador]      → máscara de instancia por región propuesta
        |
  [Fusion Engine]           → imagen con overlay + tabla de resultados
        |
  FastAPI (backend)  ←→  Streamlit (frontend)
```

El pipeline es híbrido: YOLOv8 proporciona localización rápida y MedSAM (ViT-B fine-tuneado) genera la segmentación precisa de cada instancia.

---

## Dataset

| Atributo | Detalle |
|---|---|
| Total de imágenes | 250 radiografías |
| Normales | 71 (`N_*.jpg`) |
| Escoliosis | 179 (`S_*.jpg`) |
| Clases | 18 (background + T1–T12 + L1–L5) |
| Máscaras canónicas | `Scoliosis_Dataset/LabelMultiClass_ID_PNG/` — PNG 16-bit |
| Índice maestro | `Scoliosis_Dataset/indice_dataset.csv` |

El dataset **no está versionado en el repositorio** (incluido en `.gitignore`). Debe colocarse manualmente en `Scoliosis_Dataset/` siguiendo la estructura original.

---

## Estructura del repositorio

```
.
├── Scoliosis_Dataset/          # Dataset de radiografías (no versionado)
├── models/                     # Pesos del modelo entrenado (.pth)
├── notebooks/                  # Notebooks de experimentación y entrenamiento
├── old_notebooks/              # Versiones anteriores (archivo)
├── src/
│   ├── data/
│   │   └── preprocess.py       # Pipeline de preprocesamiento
│   └── inference/
│       └── predictor.py        # Motor de inferencia híbrido (YOLOv8 + MedSAM)
├── deployment/
│   ├── fastap_app/             # Backend FastAPI
│   │   ├── api.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── front/                  # Frontend Streamlit
│       ├── app.py
│       ├── Dockerfile
│       └── requirements.txt
├── preprocesamiento.md         # Especificación del pipeline de preprocesamiento
├── procesamiento.md            # Especificación de entrenamiento
├── análisis.md                 # Análisis arquitectónico y decisiones de diseño
└── requirements.txt            # Dependencias base
```

---

## Requisitos

- Python 3.11+
- CUDA (opcional pero recomendado — la inferencia también corre en CPU)
- Docker y Docker Compose (para despliegue contenedorizado)

### Dependencias principales

| Componente | Librería |
|---|---|
| Detección | `ultralytics` (YOLOv8) |
| Segmentación | `segment-anything` (SAM/MedSAM) |
| Backend | `fastapi`, `uvicorn` |
| Frontend | `streamlit` |
| Preprocesamiento | `opencv-python`, `albumentations`, `pillow` |
| Deep learning | `torch`, `torchvision` |

---

## Instalación local

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd segmentacion-vertebras-proyecto-maia

# 2. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Instalar dependencias del backend
pip install -r deployment/fastap_app/requirements.txt

# 4. Instalar segment-anything (MedSAM)
pip install git+https://github.com/facebookresearch/segment-anything.git

# 5. Colocar los pesos del modelo en models/
# El archivo esperado es: models/medsam_G_best_20260515_1500.pth
# (375MB — se distribuye por separado via Git LFS o enlace externo)
```

---

## Ejecución local (sin Docker)

### Backend (FastAPI)

```bash
# Desde la raíz del proyecto
set PYTHONPATH=.    # Windows
# export PYTHONPATH=.  # Linux/Mac

uvicorn deployment.fastap_app.api:app --host 0.0.0.0 --port 8000 --reload
```

La API queda disponible en `http://localhost:8000`.  
Documentación interactiva en `http://localhost:8000/docs`.

### Frontend (Streamlit)

```bash
# En otra terminal
pip install -r deployment/front/requirements.txt
streamlit run deployment/front/app.py
```

La interfaz queda disponible en `http://localhost:8501`.  
En la barra lateral, configurar el endpoint de la API como `http://localhost:8000`.

---

## Ejecución con Docker

```bash
# Construir y levantar ambos servicios
docker build -t maia-api -f deployment/fastap_app/Dockerfile .
docker build -t maia-front -f deployment/front/Dockerfile deployment/front/

docker run -d -p 8000:8000 --name maia-api maia-api
docker run -d -p 8501:8501 --name maia-front maia-front
```

Acceder a la interfaz en `http://localhost:8501`.

> **Nota:** si se quiere usar GPU dentro del contenedor, agregar `--gpus all` al `docker run` del backend y usar la imagen base con soporte CUDA.

---

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Health check del servicio |
| `POST` | `/predict` | Recibe imagen (multipart), retorna JSON con imagen segmentada (base64), tabla de vértebras detectadas, confidence scores y bounding boxes |

### Ejemplo de uso con curl

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@radiografia.jpg" \
  | python -m json.tool
```

---

## Uso de la interfaz web

1. Abrir `http://localhost:8501` en el navegador.
2. Subir una radiografía de columna en formato JPEG o PNG.
3. El sistema procesa la imagen y muestra:
   - Panel izquierdo: radiografía original.
   - Panel derecho: radiografía con segmentación superpuesta.
   - Tabla inferior: vértebras detectadas con etiqueta anatómica, confidence score y coordenadas del bounding box.
4. El slider **Alpha** en la barra lateral controla la opacidad del overlay.

---

## Notebooks de experimentación

Los notebooks en `notebooks/` documentan el proceso de desarrollo e incluyen:

| Notebook | Contenido |
|---|---|
| `medsam_ablation_*.ipynb` | Ablation study de MedSAM con distintos preprocesos y loss functions |
| `yolov8_*.ipynb` | Experimentos con YOLOv8 para detección de instancias |
| `maskrcnn_*.ipynb` | Experimentos con Mask R-CNN (ResNet50/101/EfficientNet) |
| `hybrid_yolov8_medsam_v1.ipynb` | Pipeline híbrido final |

Para ejecutar los notebooks, el dataset debe estar presente en `Scoliosis_Dataset/`.

---

## Pipeline de preprocesamiento

Ver [`preprocesamiento.md`](preprocesamiento.md) para la especificación completa. Resumen:

1. **Resize** a formato rectangular (512×1024) preservando relación de aspecto.
2. **Conversión a escala de grises** con conversión perceptual.
3. **ROI crop** usando la máscara binaria + margen ≥10%.
4. **CLAHE** para mejora de contraste local.
5. **Augmentation** sincronizada imagen-máscara (flip horizontal, rotación ±10°, deformación elástica).
6. **Normalización** z-score con stats de ImageNet.

---

## Entrenamiento

Ver [`procesamiento.md`](procesamiento.md) para la especificación completa.

El entrenamiento de los modelos Mask R-CNN sigue una estrategia de **descongelamiento progresivo** del encoder con **learning rates diferenciales por bloque**, usando una loss combinada `Dice + Focal` para manejar el desbalance de clases. Los notebooks de `notebooks/` contienen los runs de entrenamiento completos.

---

## Hardware de referencia

El proyecto fue desarrollado y entrenado con:

| Componente | Especificación |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 (4GB VRAM) |
| CPU | AMD Ryzen 5 5600X @ 3.70GHz |
| RAM | 16GB |

La inferencia en CPU es funcional pero significativamente más lenta. Para entrenamiento se recomienda mínimo 8GB VRAM.
