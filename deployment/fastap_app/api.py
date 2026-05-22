from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from functools import partial

# --- PARCHE DE COMPATIBILIDAD ---
# Forzamos a que torch.load use weights_only=False por defecto
torch.load = partial(torch.load, weights_only=False)
# --------------------------------
from src.inference.predictor import Predictor

app = FastAPI(title="API de Segmentación (MedSAM)")

# ¡Aquí apuntamos al nuevo modelo YOLO!
predictor = Predictor(
    model_path="src/models/medsam_spine_finetuned.pth"
)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), alpha: float = 0.5):
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Formato inválido"
        )

    content = await file.read()
    return predictor.predict(content, alpha)
