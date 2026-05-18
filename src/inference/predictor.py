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

    def predict(self, image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        self.predictor.set_image(img_rgb)
        # Lienzo base transparente
        final_colored_mask = np.zeros((h_orig, w_orig, 4), dtype=np.uint8)

        # conf=0.35 -> Ignora detecciones con menos del 35% de seguridad
        # iou=0.45 -> Si dos cajas se superponen más del 45%, elimina la más débil
        results = self.yolo(img_rgb, conf=0.35, iou=0.45, verbose=False)
        
        # --- PRE-PROCESAMIENTO: Recolectar y Ordenar Detecciones de YOLO ---
        detecciones_validas = []
        if results[0].boxes:
            for box in results[0].boxes:
                # Extraemos los datos de YOLO
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].item())
                label_name = str(self.yolo.names[cls_id])
                
                # Guardamos los datos necesarios en una lista
                detecciones_validas.append({
                    'box': np.array([int(x1), int(y1), int(x2), int(y2)]),
                    'y_top': int(y1), # Lo usaremos para ordenar de arriba a abajo
                    'label': label_name
                })

        # SOLUCIÓN PROBLEMA 2 (Sobreposición): Ordenar estrictamente de ARRIBA a ABAJO.
        detecciones_validas.sort(key=lambda d: d['y_top'])

        # --- CAPA 1: DIBUJAR TODAS LAS MÁSCARAS DE SEGMENTACIÓN (MEDSAM) ---
        print("🎨 Generando y dibujando máscaras de MedSAM...")
        # Lista temporal para guardar info de textos para la siguiente capa
        info_textos = []

        for det in detecciones_validas:
            input_box = det['box']

            # Generar máscara con MedSAM
            masks, _, _ = self.predictor.predict(
                point_coords=None, point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            if masks[0].any():
                # 1. Dibujar máscara en el lienzo final (CAPA COLORES)
                color = list(np.random.choice(range(50, 256), size=3)) + [200]
                mask_res = cv2.resize(masks[0].astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                final_colored_mask[mask_res > 0] = color

                # 2. Guardar información de texto para después
                text_x = input_box[0] # Coordenada x1
                # Usamos y_top de YOLO para la posición del texto, con ajuste de seguridad
                text_y = max(det['y_top'] - 5, 25) 
                info_textos.append({'label': det['label'], 'x': int(text_x), 'y': int(text_y)})

        # --- CAPA 2: DIBUJAR TODOS LOS TEXTOS ENCIMA DE TODO (CON ANTI-COLISIÓN) ---
        print("🏷️ Dibujando etiquetas de YOLO con anti-colisión...")
        font = cv2.FONT_HERSHEY_SIMPLEX
        last_y_drawn = -50  # Memoria de dónde se dibujó el último texto

        for info in info_textos:
            text_x, text_y = info['x'], info['y']
            label_name = info['label']

            # ALGORITMO ANTI-COLISIÓN:
            # Si la coordenada Y actual está a menos de 28 píxeles del texto anterior...
            if text_y < last_y_drawn + 28:
                # ...lo empujamos hacia abajo para que quede justo debajo
                text_y = last_y_drawn + 28

            # Borde negro (outline) para contraste
            cv2.putText(
                img=final_colored_mask, text=label_name, org=(text_x, text_y),
                fontFace=font, fontScale=0.8, color=(0, 0, 0, 255), thickness=4, lineType=cv2.LINE_AA
            )
            # Relleno amarillo (text body)
            cv2.putText(
                img=final_colored_mask, text=label_name, org=(text_x, text_y),
                fontFace=font, fontScale=0.8, color=(255, 255, 0, 255), thickness=2, lineType=cv2.LINE_AA
            )

            # Actualizamos la memoria para el siguiente texto
            last_y_drawn = text_y

        print("✅ Segmentación híbrida completada de arriba a abajo.")
        _, buffer = cv2.imencode('.png', cv2.cvtColor(final_colored_mask, cv2.COLOR_RGBA2BGRA))
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")