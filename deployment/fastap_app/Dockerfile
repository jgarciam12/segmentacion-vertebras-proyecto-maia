FROM python:3.13-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt

# 1. Dependencias del sistema necesarias para OpenCV y para clonar repositorios (Git)
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Copiar archivos manteniendo la estructura que espera tu código
# Estructura final en el contenedor:
# /opt/src/inference/...
# /opt/deployment/fastap_app/api.py
COPY src/ /opt/src/
COPY deployment/fastap_app/ /opt/deployment/fastap_app/

# 3. Instalación de librerías
COPY deployment/fastap_app/requirements.txt .
RUN pip install --upgrade pip

# Instalar PyTorch CPU ONLY
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Instalar los requerimientos del proyecto (incluirá segment-anything de Github)
RUN pip install --no-cache-dir -r requirements.txt

# 4. Configurar el PYTHONPATH para que Python encuentre el módulo 'src'
ENV PYTHONPATH=/opt

# 5. Permisos y Usuario
RUN adduser --disabled-password api-user && chown -R api-user:api-user /opt
USER api-user

EXPOSE 8000

# 6. Ejecutar desde /opt para que las rutas relativas coincidan
CMD ["uvicorn", "deployment.fastap_app.api:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t vertebra-api:latest -f deployment/fastap_app/Dockerfile .
# docker run -p 8000:8000 -it -e PORT=8000 vertebra-api
