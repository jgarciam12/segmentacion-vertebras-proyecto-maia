# Plan: Mask R-CNN + ResNet101-FPN
**Notebook:** `maskrcnn_resnet101_fpn.ipynb`  
**Estado:** Listo para implementar  
**Advertencia OOM:** ResNet101 tiene riesgo real de OOM en GTX 1650 (4GB VRAM). FP16 es obligatorio. Si ocurre OOM, ejecutar en Google Colab o Kaggle (GPU T4, 16GB VRAM).

---

## Diferencias respecto a `maskrcnn_resnet50_fpn`

Este notebook es estructuralmente idéntico al de ResNet50. Las únicas diferencias son:

| Aspecto | ResNet50 | ResNet101 |
|---|---|---|
| Carga del modelo | `maskrcnn_resnet50_fpn(weights=COCO_V1)` | `maskrcnn_resnet101_fpn(weights=COCO_V1)` |
| Parámetros del encoder | ~25M | ~44M |
| Riesgo OOM (4GB VRAM) | Bajo | Alto |
| Checkpoint path | `best_resnet50_phase{n}.pth` | `best_resnet101_phase{n}.pth` |

Todas las funciones, clases y el pipeline son los mismos. El plan completo a continuación es una copia exacta de `maskrcnn_resnet50_fpn.md` con los cambios aplicados en los puntos indicados.

---

## Dependencias

```
torch torchvision albumentations opencv-python pillow numpy pandas matplotlib scikit-learn pycocotools
```

---

## Sección 1 — Preprocesamiento

### Funciones de carga

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `load_image` | `(image_path: str) -> np.ndarray` | Array RGB uint8 (H, W, 3) | Carga imagen desde disco con PIL |
| `load_mask` | `(mask_path: str) -> np.ndarray` | Array uint16 (H, W) | Carga máscara 16-bit con `cv2.IMREAD_UNCHANGED` para no truncar IDs |
| `load_binary_mask` | `(binary_mask_path: str) -> np.ndarray` | Array binario (H, W) | Carga máscara binaria para cálculo del ROI |
| `load_dataset_index` | `(csv_path: str) -> pd.DataFrame` | DataFrame con rutas imagen→máscaras | Carga `dataset_index.csv` como fuente canónica de rutas |

### Funciones de transformación

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `to_grayscale` | `(image: np.ndarray) -> np.ndarray` | Array uint8 (H, W) | Conversión perceptual: L = 0.299R + 0.587G + 0.114B |
| `map_entity_ids` | `(mask: np.ndarray) -> np.ndarray` | Array uint16 (H, W) | Mapea IDs 23–35 a 0 (background); el resto intacto |
| `compute_roi` | `(binary_mask: np.ndarray, margin: float = 0.1) -> tuple` | (x1, y1, x2, y2) | Bounding box de la columna sobre la máscara binaria con margen ≥10% |
| `crop_to_roi` | `(image: np.ndarray, mask: np.ndarray, roi: tuple) -> tuple` | (image_crop, mask_crop) | Aplica el mismo crop a imagen y máscara con las coordenadas del ROI |
| `resize_pair` | `(image: np.ndarray, mask: np.ndarray, target_size: tuple) -> tuple` | (image_r, mask_r) | Resize bilinear para imagen, nearest neighbor para máscara |
| `apply_clahe` | `(image: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8,8)) -> np.ndarray` | Array uint8 (H, W) | CLAHE sobre imagen monocanal para mejorar contraste local |
| `replicate_to_3ch` | `(image: np.ndarray) -> np.ndarray` | Array uint8 (H, W, 3) | Replica canal gris a 3 canales para compatibilidad con encoder ImageNet |
| `normalize_image` | `(image: np.ndarray) -> torch.Tensor` | Tensor float32 (3, H, W) | Estandarización con mean/std ImageNet: [0.485,0.456,0.406] / [0.229,0.224,0.225] |
| `semantic_to_instance` | `(mask: np.ndarray) -> dict` | `{boxes, masks, labels}` | Convierte máscara semántica a formato instancia para Mask R-CNN. Por imagen cada clase es una instancia única. |

### Augmentation

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `build_augmentation_pipeline` | `() -> albumentations.Compose` | Pipeline de albumentations | Flip horizontal, rotación ±10°, zoom aleatorio, deformación elástica, brillo/contraste solo en imagen |
| `apply_augmentation` | `(image: np.ndarray, mask: np.ndarray, pipeline: Compose) -> tuple` | (image_aug, mask_aug) | Augmentation sincronizada imagen-máscara con la misma seed |

### Split y Dataset

| Función/Clase | Firma | Retorna | Descripción |
|---|---|---|---|
| `split_dataset` | `(df: pd.DataFrame, train=0.70, val=0.15, seed=42) -> tuple` | (train_df, val_df, test_df) | Split estratificado por columna `type` (Normal/Scoliosis) |
| `SpineDataset` | `(df, mode: str, aug_pipeline=None, target_size=(512,1024))` | Dataset PyTorch | Aplica el pipeline completo en `__getitem__`. `mode='train'` aplica augmentation; `'val'`/`'test'` no. Retorna `(image_tensor, target_dict)` compatible con Mask R-CNN |
| `collate_fn` | `(batch: list) -> tuple` | (images_list, targets_list) | Collate para DataLoader: Mask R-CNN espera lista de tensores, no batch tensorial |
| `create_dataloaders` | `(train_ds, val_ds, test_ds, batch_size=1) -> tuple` | (train_dl, val_dl, test_dl) | Crea los 3 DataLoaders con `collate_fn` |

---

## Sección 2 — Procesamiento (Entrenamiento)

### Construcción del modelo

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `build_model` | `(num_classes: int = 23) -> MaskRCNN` | Modelo Mask R-CNN | **`maskrcnn_resnet101_fpn(weights=COCO_V1)`** (diferencia clave vs ResNet50). Reemplaza `box_predictor` y `mask_predictor` con `num_classes` |

### Control del encoder

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `freeze_encoder` | `(model) -> None` | — | Congela todos los parámetros del backbone (`requires_grad=False`) |
| `unfreeze_block` | `(model, block_idx: int) -> None` | — | Descongela `layer{block_idx}` del backbone ResNet101. `block_idx` en [1,2,3,4] |
| `get_param_groups` | `(model, base_lr: float = 1e-3) -> list` | Lista de param groups | Grupos con LR diferencial: decoder=base_lr, layer4=×0.1, layer3=×0.01, layer2=×0.001, layer1=×0.0001 |

### Optimizador y scheduling

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `build_optimizer` | `(param_groups: list, weight_decay: float = 1e-4) -> Optimizer` | SGD | SGD con momentum=0.9 y weight_decay=1e-4 |
| `build_scheduler` | `(optimizer, patience: int = 3, factor: float = 0.5) -> Scheduler` | ReduceLROnPlateau | Reduce LR a la mitad si val_loss no mejora en `patience` epochs |

### Loops de entrenamiento

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `train_one_epoch` | `(model, dataloader, optimizer, scaler, device) -> float` | val_loss promedio | Loop de entrenamiento con `torch.cuda.amp.GradScaler` (FP16 obligatorio, especialmente crítico aquí) |
| `validate_one_epoch` | `(model, dataloader, device) -> float` | val_loss promedio | Loop de validación sin gradientes |

### Checkpointing

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `save_checkpoint` | `(model, path: str, val_loss: float) -> None` | — | Guarda pesos completos cuando val_loss mejora |
| `load_checkpoint` | `(model, path: str) -> model` | Modelo con pesos cargados | Carga el mejor checkpoint de una fase |
| `save_final_model` | `(model, path: str) -> None` | — | Guarda el estado final del modelo entrenado en un `.pth` fijo. Se llama una sola vez al terminar todas las fases. |

### Entrenamiento por fases

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `train_phase` | `(model, train_dl, val_dl, optimizer, scheduler, device, max_epochs: int, patience: int = 7, checkpoint_path: str) -> float` | Mejor val_loss | Entrena una fase con early stopping + checkpointing |
| `run_progressive_training` | `(model, train_dl, val_dl, device) -> model` | Modelo entrenado | 3 fases: Fase 1 (encoder congelado, 15 epochs), Fase 2 (descongelar layer4, 10 epochs), Fase 3 (descongelar layer3, 8 epochs) |

---

## Sección 3 — Métricas

### Métricas de pixel (Dice e IoU)

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `compute_dice` | `(pred: np.ndarray, gt: np.ndarray) -> float` | Dice en [0,1] | Dice entre dos máscaras binarias de la misma clase |
| `compute_iou` | `(pred: np.ndarray, gt: np.ndarray) -> float` | IoU en [0,1] | IoU entre dos máscaras binarias de la misma clase |
| `compute_dice_per_class` | `(predictions: list, targets: list, num_classes: int = 23) -> dict` | `{class_id: dice}` | Dice por clase sobre el test set. Excluye clases ausentes en cada imagen |
| `compute_miou` | `(predictions: list, targets: list, num_classes: int = 23) -> float` | mIoU global | IoU promedio sobre clases presentes en cada imagen |
| `compute_mean_dice` | `(dice_per_class: dict) -> float` | mean Dice global | Promedio de Dice sobre todas las clases presentes |

### Métricas de detección (AP)

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `compute_ap50_per_class` | `(predictions: list, targets: list) -> dict` | `{class_id: AP@50}` | AP@50 por clase con pycocotools. IoU ≥ 0.50 para detección correcta |
| `compute_map` | `(ap_per_class: dict) -> float` | mAP global | Promedio de AP@50 sobre todas las clases |

### Evaluación completa

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `run_inference` | `(model, dataloader, device) -> tuple` | (predictions, targets) | Inferencia sobre test set |
| `evaluate_model` | `(predictions: list, targets: list) -> dict` | Dict con todas las métricas | Dice/clase, mean Dice, mIoU, AP@50/clase, mAP |
| `print_metrics_report` | `(metrics: dict) -> None` | — | Tabla resumen de métricas por clase y globales |

---

## Pipeline final

```python
# === CONFIGURACIÓN ===
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SIZE  = (512, 1024)
NUM_CLASSES  = 23
SEED         = 42

# === PREPROCESAMIENTO ===
index_df                      = load_dataset_index('MaIA_Scoliosis_Dataset/dataset_index.csv')
train_df, val_df, test_df     = split_dataset(index_df, seed=SEED)

aug_pipeline = build_augmentation_pipeline()
train_ds     = SpineDataset(train_df, mode='train', aug_pipeline=aug_pipeline, target_size=TARGET_SIZE)
val_ds       = SpineDataset(val_df,   mode='val',   target_size=TARGET_SIZE)
test_ds      = SpineDataset(test_df,  mode='test',  target_size=TARGET_SIZE)

train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, batch_size=1)

# === PROCESAMIENTO (ENTRENAMIENTO) ===
model        = build_model(num_classes=NUM_CLASSES).to(DEVICE)
model        = run_progressive_training(model, train_dl, val_dl, DEVICE)

# === GUARDAR MODELO FINAL ===
save_final_model(model, 'models/maskrcnn_resnet101_fpn_best.pth')

# === MÉTRICAS ===
predictions, targets = run_inference(model, test_dl, DEVICE)
metrics              = evaluate_model(predictions, targets)
print_metrics_report(metrics)
```
