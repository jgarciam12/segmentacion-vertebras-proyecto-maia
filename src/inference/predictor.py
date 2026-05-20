import io
import os
import numpy as np
import cv2
import torch
import base64
from fastapi.responses import JSONResponse
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

class Predictor:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⌛ Iniciando Pipeline Híbrido (YOLO + MedSAM) en {self.device}...")

        # --- 1. CARGAR YOLO ---
        yolo_path = "src/models/yolov8_vertebra_detect.pt" 
        if os.path.exists(yolo_path):
            self.yolo = YOLO(yolo_path)
            print(f"👁️ YOLOv8 cargado exitosamente.")
        else:
            print(f"⚠️ ERROR: No encuentro el modelo YOLO en {yolo_path}")

        # --- 2. CARGAR MEDSAM ---
        base_model_path = "src/models/sam_vit_b_01ec64.pth"
        self.sam = sam_model_registry["vit_b"](checkpoint=base_model_path)
        
        finetuned_weights = torch.load(model_path, map_location=self.device)
        new_state_dict = {}
        for k, v in finetuned_weights.items():
            new_key = k
            if not k.startswith(("image_encoder", "prompt_encoder", "mask_decoder")):
                new_key = f"mask_decoder.{k}"
            new_key = new_key.replace("mask_decoder.upscale_conv1.", "mask_decoder.output_upscaling.0.")
            new_key = new_key.replace("mask_decoder.upscale_layer_norm.", "mask_decoder.output_upscaling.1.")
            new_key = new_key.replace("mask_decoder.upscale_conv2.", "mask_decoder.output_upscaling.3.")
            new_state_dict[new_key] = v

        self.sam.load_state_dict(new_state_dict, strict=False)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        print("🚀 MedSAM ensamblado. ¡Pipeline Listo!")

    def predict(self, image_bytes: bytes, alpha: float = 0.5):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        self.predictor.set_image(img_rgb)
        
        # Ya no necesitamos generar capas extra, solo MedSAM puro para la fusión
        img_medsam_pure = np.zeros((h_orig, w_orig, 3), dtype=np.uint8) 

        results = self.yolo(img_rgb, conf=0.35, iou=0.45, verbose=False)
        
        detecciones_validas = []
        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].item()) # NUEVO: Capturar porcentaje de confianza
                cls_id = int(box.cls[0].item())
                label_name = str(self.yolo.names[cls_id])
                
                detecciones_validas.append({
                    'box': np.array([int(x1), int(y1), int(x2), int(y2)]),
                    'y_top': int(y1),
                    'label': label_name,
                    'confianza': conf
                })

        # Ordenar de arriba a abajo
        detecciones_validas.sort(key=lambda d: d['y_top'])

        info_textos = []
        datos_tabla = [] # NUEVO: Lista para alimentar el Dataframe de Streamlit

        # --- PROCESAMIENTO HÍBRIDO ---
        for det in detecciones_validas:
            input_box = det['box']
            
            # NUEVO: Llenar los datos de la fila para esta vértebra
            datos_tabla.append({
                "Vértebra": det['label'],
                "Confianza (%)": round(det['confianza'] * 100, 2),
                "Bounding Box": f"[{input_box[0]}, {input_box[1]}, {input_box[2]}, {input_box[3]}]"
            })

            # Segmentación con MedSAM
            masks, _, _ = self.predictor.predict(
                point_coords=None, point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            if masks[0].any():
                color = list(np.random.choice(range(80, 256), size=3))
                mask_res = cv2.resize(masks[0].astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                img_medsam_pure[mask_res > 0] = color

                text_x = input_box[0]
                text_y = max(det['y_top'] - 5, 25)
                info_textos.append({'label': det['label'], 'x': int(text_x), 'y': int(text_y)})

        # --- GENERAR LIENZO DE FUSIÓN ---
        img_fusion = img_rgb.copy()
        mask_active = np.any(img_medsam_pure > 0, axis=-1)
        img_fusion[mask_active] = (img_rgb[mask_active] * (1 - alpha) + img_medsam_pure[mask_active] * alpha).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        last_y_fusion = -50
        for info in info_textos:
            tx, ty = info['x'], info['y']
            if ty < last_y_fusion + 28: ty = last_y_fusion + 28
            cv2.putText(img_fusion, info['label'], (tx, ty), font, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img_fusion, info['label'], (tx, ty), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            last_y_fusion = ty

        def to_b64(img):
            _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buffer).decode('utf-8')

        print("✅ Inferencia completada y tabla extraída.")
        # NUEVO: Devolvemos la imagen Y los datos de la tabla
        return JSONResponse(content={
            "fusion": to_b64(img_fusion),
            "tabla": datos_tabla,
            "total_vertebras": len(datos_tabla)
        })