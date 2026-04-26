from fastapi import FastAPI, UploadFile, File
import torch
import segmentation_models_pytorch as smp
from preprocess import preprocess_image
import numpy as np
import cv2
from fastapi.responses import StreamingResponse
import io

app = FastAPI(title="API de Segmentación de Vértebras")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Definimos la arquitectura (igual que en el notebook)
model = smp.Unet(
    encoder_name='resnet34', 
    encoder_weights=None, 
    in_channels=3, 
    classes=1
)

# 2. Cargar los pesos (Aquí es donde usaremos el archivo .pth)
MODEL_PATH = "unet_resnet34_columna.pth"

try:
    # Esta línea fallará hasta que tengas el archivo en la carpeta
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✅ Modelo cargado exitosamente desde {MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠️ ¡Atención! No se encontró el archivo {MODEL_PATH}. La API no podrá predecir hasta que lo tengas.")

model.to(DEVICE)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer la imagen que sube el usuario
    contents = await file.read()
    
    # Usar tu función de preprocesamiento
    image_tensor, (h_orig, w_orig) = preprocess_image(contents)
    
    # Añadir dimensión de batch [1, 3, 256, 256]
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    # Realizar la predicción
    with torch.no_grad():
        output = model(input_tensor)
        # Aplicamos Sigmoid para convertir a probabilidad (0 a 1)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Umbral para binarizar (píxeles > 0.5 son vértebra)
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    
    # Redimensionar la máscara al tamaño original de la imagen
    mask_resized = cv2.resize(mask_binary, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    
    # Convertir la máscara a imagen para enviarla de vuelta
    _, buffer = cv2.imencode(".png", mask_resized)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)