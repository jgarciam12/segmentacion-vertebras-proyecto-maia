# Plan: Mask R-CNN + EfficientNet-FPN
**Notebook:** `maskrcnn_efficientnet_fpn.ipynb`  
**Estado:** Listo para implementar  
**Prioridad:** Baja (exploración opcional, ejecutar después de ResNet50 y nnU-Net)

---

## Diferencias arquitectónicas respecto a los planes ResNet

EfficientNet **no está soportado nativamente** como backbone de Mask R-CNN en torchvision. La integración requiere:

1. Cargar EfficientNet-B4 desde `timm` como extractor de features
2. Envolverlo con `torchvision.ops.FeaturePyramidNetwork` para añadir FPN
3. Registrarlo como backbone personalizado en `torchvision.models.detection.MaskRCNN`

La estrategia de descongelamiento progresivo opera sobre los **stages** de EfficientNet-B4 (stages 0–6) en lugar de los `layer1`–`layer4` de ResNet.

---

## Dependencias

```
torch torchvision timm albumentations opencv-python pillow numpy pandas matplotlib scikit-learn pycocotools
```

---

## Sección 1 — Preprocesamiento

Idéntica al plan `maskrcnn_resnet50_fpn.md`. Las siguientes funciones se copian sin modificación:

| Función | Firma | Descripción |
|---|---|---|
| `load_image` | `(image_path: str) -> np.ndarray` | Carga imagen RGB uint8 |
| `load_mask` | `(mask_path: str) -> np.ndarray` | Carga máscara 16-bit uint16 |
| `load_binary_mask` | `(binary_mask_path: str) -> np.ndarray` | Carga máscara binaria para ROI |
| `load_dataset_index` | `(csv_path: str) -> pd.DataFrame` | Carga `dataset_index.csv` |
| `to_grayscale` | `(image: np.ndarray) -> np.ndarray` | L = 0.299R + 0.587G + 0.114B |
| `map_entity_ids` | `(mask: np.ndarray) -> np.ndarray` | Mapea IDs 23–35 a 0 |
| `compute_roi` | `(binary_mask, margin=0.1) -> tuple` | Bounding box + margen ≥10% |
| `crop_to_roi` | `(image, mask, roi) -> tuple` | Crop sincronizado imagen-máscara |
| `resize_pair` | `(image, mask, target_size) -> tuple` | Bilinear + nearest neighbor |
| `apply_clahe` | `(image, clip_limit=2.0, tile_grid=(8,8)) -> np.ndarray` | CLAHE monocanal |
| `replicate_to_3ch` | `(image: np.ndarray) -> np.ndarray` | (H,W) → (H,W,3) |
| `normalize_image` | `(image: np.ndarray) -> torch.Tensor` | Estandarización ImageNet stats |
| `semantic_to_instance` | `(mask: np.ndarray) -> dict` | Semántica → instancia para Mask R-CNN |
| `build_augmentation_pipeline` | `() -> Compose` | Flip, rotación ±10°, zoom, elástica, brillo/contraste |
| `apply_augmentation` | `(image, mask, pipeline) -> tuple` | Augmentation sincronizada |
| `split_dataset` | `(df, train=0.70, val=0.15, seed=42) -> tuple` | Split estratificado Normal/Scoliosis |
| `SpineDataset` | `(df, mode, aug_pipeline, target_size)` | Dataset PyTorch completo |
| `collate_fn` | `(batch) -> tuple` | Collate lista para Mask R-CNN |
| `create_dataloaders` | `(train_ds, val_ds, test_ds, batch_size=1) -> tuple` | 3 DataLoaders |

---

## Sección 2 — Procesamiento (Entrenamiento)

### Construcción del backbone EfficientNet-FPN

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `build_efficientnet_backbone` | `(model_name: str = 'efficientnet_b4', out_channels: int = 256) -> BackboneWithFPN` | Backbone con FPN | Carga EfficientNet-B4 desde timm, extrae features en stages [2,3,4,6], añade FPN con `out_channels=256` |
| `build_model` | `(num_classes: int = 23) -> MaskRCNN` | Modelo Mask R-CNN personalizado | Construye `MaskRCNN` con el backbone EfficientNet-FPN. Reemplaza `box_predictor` y `mask_predictor` con `num_classes`. Pesos del backbone: ImageNet (vía timm) |

### Control del encoder

EfficientNet-B4 tiene 7 stages (0–6). El descongelamiento opera en stages en lugar de layers:

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `freeze_encoder` | `(model) -> None` | — | Congela todos los parámetros del backbone EfficientNet |
| `unfreeze_stage` | `(model, stage_idx: int) -> None` | — | Descongela el stage indicado del backbone (`model.backbone.body.blocks[stage_idx]`). Llamar en orden descendente: 6 → 5 → 4 → ... |
| `get_param_groups` | `(model, base_lr: float = 1e-3) -> list` | Lista de param groups | Grupos con LR diferencial: decoder/FPN=base_lr, stage6=×0.1, stage5=×0.01, stage4=×0.001 |

### Optimizador y scheduling

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `build_optimizer` | `(param_groups: list, weight_decay: float = 1e-4) -> Optimizer` | SGD | SGD con momentum=0.9 y weight_decay=1e-4 |
| `build_scheduler` | `(optimizer, patience: int = 3, factor: float = 0.5) -> Scheduler` | ReduceLROnPlateau | Reduce LR a la mitad si val_loss no mejora en `patience` epochs |

### Loops de entrenamiento

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `train_one_epoch` | `(model, dataloader, optimizer, scaler, device) -> float` | val_loss promedio | Loop con FP16 obligatorio |
| `validate_one_epoch` | `(model, dataloader, device) -> float` | val_loss promedio | Loop sin gradientes |

### Checkpointing

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `save_checkpoint` | `(model, path: str, val_loss: float) -> None` | — | Guarda pesos cuando val_loss mejora |
| `load_checkpoint` | `(model, path: str) -> model` | Modelo con pesos cargados | Carga checkpoint de una fase |
| `save_final_model` | `(model, path: str) -> None` | — | Guarda el estado final del modelo entrenado en un `.pth` fijo. Se llama una sola vez al terminar todas las fases. |

### Entrenamiento por fases

| Función | Firma | Retorna | Descripción |
|---|---|---|---|
| `train_phase` | `(model, train_dl, val_dl, optimizer, scheduler, device, max_epochs, patience=7, checkpoint_path) -> float` | Mejor val_loss | Fase completa con early stopping + checkpointing |
| `run_progressive_training` | `(model, train_dl, val_dl, device) -> model` | Modelo entrenado | 3 fases sobre stages de EfficientNet: Fase 1 (congelado, 15 epochs), Fase 2 (descongelar stage6, 10 epochs), Fase 3 (descongelar stage5, 8 epochs) |

---

## Sección 3 — Métricas

Idéntica al plan `maskrcnn_resnet50_fpn.md`. Todas las funciones se copian sin modificación:

| Función | Firma | Descripción |
|---|---|---|
| `compute_dice` | `(pred, gt) -> float` | Dice binario |
| `compute_iou` | `(pred, gt) -> float` | IoU binario |
| `compute_dice_per_class` | `(predictions, targets, num_classes=23) -> dict` | Dice por clase, excluye ausentes |
| `compute_mean_dice` | `(dice_per_class) -> float` | Promedio Dice |
| `compute_miou` | `(predictions, targets, num_classes=23) -> float` | mIoU sobre clases presentes |
| `compute_ap50_per_class` | `(predictions, targets) -> dict` | AP@50 por clase |
| `compute_map` | `(ap_per_class) -> float` | mAP global |
| `run_inference` | `(model, dataloader, device) -> tuple` | Inferencia test set |
| `evaluate_model` | `(predictions, targets) -> dict` | Dict con todas las métricas |
| `print_metrics_report` | `(metrics) -> None` | Tabla resumen |

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
save_final_model(model, 'models/maskrcnn_efficientnet_fpn_best.pth')

# === MÉTRICAS ===
predictions, targets = run_inference(model, test_dl, DEVICE)
metrics              = evaluate_model(predictions, targets)
print_metrics_report(metrics)
```
