import torch
import numpy as np
import segmentation_models_pytorch as smp
import cv2
import io

from fastapi.responses import StreamingResponse

from src.data.preprocess import preprocess_image


class Predictor:
    def __init__(self, model_path: str):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Arquitectura del modelo
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )

        # Cargar pesos
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Modelo cargado desde {model_path}")

    def predict(self, image_bytes: bytes):

        # Preprocesamiento
        image_tensor, (h_orig, w_orig) = preprocess_image(image_bytes=image_bytes)

        # Batch dimension
        input_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Inferencia
        with torch.no_grad():
            output = self.model(input_tensor)

            mask = torch.sigmoid(output).squeeze().cpu().numpy()

        # Binarización
        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        # Resize al tamaño original
        mask_resized = cv2.resize(
            mask_binary,
            (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST
        )

        # Convertir a PNG
        _, buffer = cv2.imencode('.png', mask_resized)

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png",
        )
