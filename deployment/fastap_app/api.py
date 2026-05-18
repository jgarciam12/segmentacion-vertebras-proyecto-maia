from fastapi import FastAPI, UploadFile, File, HTTPException
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
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Formato inválido"
        )

    content = await file.read()
    return predictor.predict(content)