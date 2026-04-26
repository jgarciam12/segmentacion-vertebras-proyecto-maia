import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuración
IMG_SIZE = 256

# Definimos el pipeline de transformación
# Usamos la misma Normalización que en el entrenamiento
transform_pipeline = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess_image(image_bytes: bytes):
    """
    Función de preprocesamiento funcional recomendada.
    Recibe bytes de una imagen y devuelve el tensor listo para el modelo.
    """
    # 1. Convertir los bytes recibidos a un formato que OpenCV entienda
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Cambiar de BGR (OpenCV) a RGB (como espera el modelo)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. Guardar dimensiones originales para poder reescalar la máscara después
    h_orig, w_orig = image.shape[:2]
    
    # 4. Aplicar las transformaciones (Resize, Normalize, ToTensor)
    augmented = transform_pipeline(image=image)
    image_tensor = augmented['image']
    
    # Retornamos el tensor y las dimensiones originales
    return image_tensor, (h_orig, w_orig)