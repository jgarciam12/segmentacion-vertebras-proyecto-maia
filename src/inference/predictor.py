import io
import os
import numpy as np
import cv2
import torch
from fastapi.responses import StreamingResponse
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

class Predictor:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⌛ Iniciando Pipeline Híbrido (YOLO + MedSAM) en {self.device}...")

        # --- 1. CARGAR YOLO ---
        yolo_path = "src/models/yolov8m_B_filtrado_best.pt" 
        if os.path.exists(yolo_path):
            self.yolo = YOLO(yolo_path)
            print("👁️ YOLOv8 cargado exitosamente.")
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

    def predict(self, image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        self.predictor.set_image(img_rgb)
        final_colored_mask = np.zeros((h_orig, w_orig, 4), dtype=np.uint8)

        # --- PASO A: YOLO DETECTA LAS VÉRTEBRAS ---
        print("🔍 YOLO escaneando imagen...")
        results = self.yolo(img_rgb, verbose=False)
        
        # Iteramos sobre cada vértebra que YOLO encontró
        for box in results[0].boxes:
            # Extraemos las coordenadas de la caja
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            input_box = np.array([int(x1), int(y1), int(x2), int(y2)])

            # --- PASO B: MEDSAM SEGMENTA LA CAJA ---
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            # --- PASO C: COLOREAR ---
            if masks[0].any():
                color = list(np.random.choice(range(256), size=3)) + [200]
                mask_res = cv2.resize(masks[0].astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                final_colored_mask[mask_res > 0] = color

        print("✅ Segmentación híbrida completada.")
        _, buffer = cv2.imencode('.png', cv2.cvtColor(final_colored_mask, cv2.COLOR_RGBA2BGRA))
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")