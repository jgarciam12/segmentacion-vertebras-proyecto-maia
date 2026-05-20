# Plan: nnU-Net
**Notebook:** `nnunet.ipynb`  
**Estado:** Listo para implementar  
**Rol:** Baseline de referencia externo. No se interviene en arquitectura, loss, LR ni estrategia de entrenamiento.

---

## Particularidades de nnU-Net

nnU-Net es un framework autocontenido. El notebook actúa como orquestador de comandos CLI y de conversión de datos, no como implementador del modelo. Las tres secciones del notebook tienen responsabilidades distintas a los notebooks de Mask R-CNN:

| Sección | Qué hace el notebook | Qué hace nnU-Net |
|---|---|---|
| Preprocesamiento | Convierte el dataset al formato nnU-Net | Ejecuta su propio preprocesamiento interno |
| Procesamiento | Lanza los comandos de entrenamiento nnU-Net | Gestiona arquitectura, loss, LR, epochs internamente |
| Métricas | Calcula Dice/clase y mIoU sobre las predicciones exportadas | Reporta sus propias métricas internas (referencia secundaria) |

**No aplica:** AP@50 ni mAP (nnU-Net produce segmentación semántica, no por instancia).

---

## Dependencias

```
nnunetv2 torch numpy pandas matplotlib scikit-learn nibabel
```

nnU-Net requiere que las variables de entorno estén configuradas antes de correr cualquier comando:

```
nnUNet_raw        → carpeta de datos crudos en formato nnU-Net
nnUNet_preprocessed → carpeta de datos preprocesados por nnU-Net
nnUNet_results    → carpeta de resultados y checkpoints
```

---

## Sección 1 — Preprocesamiento

El objetivo es convertir el dataset MaIA al formato requerido por nnU-Net v2 (estructura de carpetas, nombres de archivos, `dataset.json`).

### Funciones de carga y validación

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `load_dataset_index` | `(csv_path: str) -> pd.DataFrame` | DataFrame con rutas imagen→máscaras | Carga `dataset_index.csv` como fuente canónica |
| `load_image` | `(image_path: str) -> np.ndarray` | Array RGB uint8 (H, W, 3) | Carga imagen desde disco con PIL |
| `load_mask` | `(mask_path: str) -> np.ndarray` | Array uint16 (H, W) | Carga máscara 16-bit con `cv2.IMREAD_UNCHANGED` |
| `load_binary_mask` | `(binary_mask_path: str) -> np.ndarray` | Array binario (H, W) | Carga máscara binaria para cálculo del ROI |

### Funciones de transformación

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `to_grayscale` | `(image: np.ndarray) -> np.ndarray` | Array uint8 (H, W) | Conversión perceptual L = 0.299R + 0.587G + 0.114B |
| `map_entity_ids` | `(mask: np.ndarray) -> np.ndarray` | Array uint16 (H, W) | Mapea IDs 23–35 a 0 (background) |
| `compute_roi` | `(binary_mask: np.ndarray, margin: float = 0.1) -> tuple` | (x1, y1, x2, y2) | Bounding box de la columna con margen ≥10% |
| `crop_to_roi` | `(image: np.ndarray, mask: np.ndarray, roi: tuple) -> tuple` | (image_crop, mask_crop) | Aplica el mismo crop a imagen y máscara |

**Nota sobre el preprocesamiento para nnU-Net:** Solo se aplican hasta el paso de ROI crop (pasos 1–2 del pipeline general). El resize, CLAHE, augmentation y normalización los gestiona nnU-Net internamente de forma autoconfigurada. No se aplica estandarización ImageNet.

### Split del dataset

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `split_dataset` | `(df: pd.DataFrame, train=0.70, val=0.15, seed=42) -> tuple` | (train_df, val_df, test_df) | Split estratificado por columna `type` (Normal/Scoliosis). Mismo split que los notebooks Mask R-CNN para comparación honesta. |

### Conversión al formato nnU-Net

nnU-Net v2 requiere una estructura específica de carpetas y un archivo `dataset.json`:

```
nnUNet_raw/
└── Dataset001_Spine/
    ├── imagesTr/       ← imágenes de entrenamiento + validación (NIfTI .nii.gz, 1 canal)
    ├── labelsTr/       ← máscaras correspondientes (NIfTI .nii.gz, uint8)
    ├── imagesTs/       ← imágenes de test
    └── dataset.json    ← metadata del dataset
```

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `create_nnunet_folder_structure` | `(base_path: str, dataset_id: str = 'Dataset001_Spine') -> dict` | Dict con paths creados | Crea la estructura de carpetas requerida por nnU-Net v2 |
| `image_to_nifti` | `(image: np.ndarray) -> nib.Nifti1Image` | NIfTI image | Convierte array uint8 monocanal a NIfTI. nnU-Net espera (H, W, 1) como canal de modalidad |
| `mask_to_nifti` | `(mask: np.ndarray) -> nib.Nifti1Image` | NIfTI image | Convierte máscara uint16 → uint8 (IDs 0–22 ya mapeados). nnU-Net acepta segmentaciones uint8 |
| `save_nifti` | `(nifti_img: Nifti1Image, path: str) -> None` | — | Guarda imagen NIfTI en disco comprimida (.nii.gz) |
| `generate_dataset_json` | `(output_path: str, num_training: int, labels: dict) -> None` | — | Genera `dataset.json` con metadata: modalidad (X-ray 2D), número de clases (23), mapeo de labels |
| `convert_dataset` | `(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, nnunet_raw_path: str) -> None` | — | Orquesta la conversión completa: itera sobre todos los splits, aplica grayscale + ROI crop + map_entity_ids, guarda en formato NIfTI en las carpetas correctas |

---

## Sección 2 — Procesamiento (Entrenamiento)

El entrenamiento se ejecuta mediante comandos CLI de nnU-Net. El notebook los lanza con `subprocess` o celdas de shell (`!`).

### Funciones de orquestación

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `set_nnunet_env_vars` | `(raw_path: str, preprocessed_path: str, results_path: str) -> None` | — | Configura las variables de entorno `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` para la sesión |
| `run_fingerprint_extraction` | `(dataset_id: int = 1) -> None` | — | Ejecuta `nnUNetv2_extract_fingerprint -d {dataset_id}`. Analiza estadísticas del dataset para autoconfigurar el plan |
| `run_plan_and_preprocess` | `(dataset_id: int = 1, config: str = '2d') -> None` | — | Ejecuta `nnUNetv2_plan_and_preprocess -d {dataset_id} -c {config}`. Genera el plan de entrenamiento y preprocesa los datos |
| `run_training` | `(dataset_id: int = 1, config: str = '2d', fold: int = 0) -> None` | — | Ejecuta `nnUNetv2_train {dataset_id} {config} {fold}`. `fold=0` usa el primer fold del split interno de nnU-Net (entrenamos 1 fold, no los 5 por limitación de tiempo) |
| `run_predict` | `(input_folder: str, output_folder: str, dataset_id: int = 1, config: str = '2d', fold: int = 0) -> None` | — | Ejecuta `nnUNetv2_predict` sobre las imágenes de test |
| `export_best_checkpoint` | `(results_path: str, output_path: str, dataset_id: int = 1, config: str = '2d', fold: int = 0) -> None` | — | Copia el checkpoint `checkpoint_best.pth` que genera nnU-Net desde `nnUNet_results/...` al path de salida indicado |

**Configuración usada:** `2d` (radiografías 2D). nnU-Net detecta automáticamente la configuración óptima, pero se fuerza `2d` para coherencia con el resto de modelos.

**Nota sobre folds:** nnU-Net hace 5-fold CV por defecto. Para este proyecto se entrena solo `fold=0` para reducir tiempo de cómputo. El test set de evaluación final es el mismo que los demás modelos (`test_df`), no el fold de validación interno de nnU-Net.

---

## Sección 3 — Métricas

nnU-Net produce máscaras semánticas (no por instancia). Las métricas se calculan sobre las predicciones exportadas por `run_predict`.

**No aplica:** AP@50, mAP (segmentación semántica, no por instancia).

### Carga de predicciones

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `load_nifti_prediction` | `(pred_path: str) -> np.ndarray` | Array uint8 (H, W) | Carga máscara predicha por nnU-Net desde NIfTI |
| `load_nifti_ground_truth` | `(gt_path: str) -> np.ndarray` | Array uint8 (H, W) | Carga máscara ground truth desde NIfTI |
| `collect_predictions_and_targets` | `(test_df: pd.DataFrame, predictions_folder: str) -> tuple` | (predictions, targets) | Itera sobre el test set y carga pares predicción-GT |

### Métricas de pixel

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `compute_dice` | `(pred: np.ndarray, gt: np.ndarray) -> float` | Dice en [0,1] | Dice entre dos máscaras binarias de la misma clase |
| `compute_iou` | `(pred: np.ndarray, gt: np.ndarray) -> float` | IoU en [0,1] | IoU entre dos máscaras binarias |
| `compute_dice_per_class` | `(predictions: list, targets: list, num_classes: int = 23) -> dict` | `{class_id: dice}` | Dice por clase sobre el test set. Excluye clases ausentes en cada imagen |
| `compute_mean_dice` | `(dice_per_class: dict) -> float` | mean Dice global | Promedio de Dice sobre clases presentes |
| `compute_miou` | `(predictions: list, targets: list, num_classes: int = 23) -> float` | mIoU global | IoU promedio sobre clases presentes |

### Evaluación completa

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `evaluate_model` | `(predictions: list, targets: list) -> dict` | Dict con métricas | Dice/clase, mean Dice, mIoU |
| `print_metrics_report` | `(metrics: dict) -> None` | — | Tabla resumen de métricas (sin AP, que no aplica a nnU-Net) |

---

## Pipeline final

```python
# === CONFIGURACIÓN ===
DATASET_ID       = 1
NNUNET_CONFIG    = '2d'
FOLD             = 0
SEED             = 42
NNUNET_RAW       = 'nnunet_data/raw'
NNUNET_PREP      = 'nnunet_data/preprocessed'
NNUNET_RESULTS   = 'nnunet_data/results'
PREDICTIONS_DIR  = 'nnunet_data/predictions'

# === PREPROCESAMIENTO — Conversión al formato nnU-Net ===
set_nnunet_env_vars(NNUNET_RAW, NNUNET_PREP, NNUNET_RESULTS)

index_df                  = load_dataset_index('MaIA_Scoliosis_Dataset/dataset_index.csv')
train_df, val_df, test_df = split_dataset(index_df, seed=SEED)

create_nnunet_folder_structure(NNUNET_RAW)
convert_dataset(train_df, val_df, test_df, NNUNET_RAW)
generate_dataset_json(f'{NNUNET_RAW}/Dataset001_Spine/dataset.json', ...)

# === PROCESAMIENTO — Entrenamiento nnU-Net ===
run_fingerprint_extraction(DATASET_ID)
run_plan_and_preprocess(DATASET_ID, config=NNUNET_CONFIG)
run_training(DATASET_ID, config=NNUNET_CONFIG, fold=FOLD)
run_predict(
    input_folder=f'{NNUNET_RAW}/Dataset001_Spine/imagesTs',
    output_folder=PREDICTIONS_DIR,
    dataset_id=DATASET_ID,
    config=NNUNET_CONFIG,
    fold=FOLD
)

# === GUARDAR MODELO FINAL ===
export_best_checkpoint(
    results_path=NNUNET_RESULTS,
    output_path='models/nnunet_best.pth',
    dataset_id=DATASET_ID,
    config=NNUNET_CONFIG,
    fold=FOLD
)

# === MÉTRICAS ===
predictions, targets = collect_predictions_and_targets(test_df, PREDICTIONS_DIR)
metrics              = evaluate_model(predictions, targets)
print_metrics_report(metrics)
```
