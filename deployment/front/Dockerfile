FROM python:3.11-slim

# Evita que Python genere archivos .pyc y permite ver logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias
COPY deployment/front/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del frontend
COPY deployment/front/ /app/

# Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker build -t vertebra-segmentation-front:latest -f deployment/front/Dockerfile .
# docker run -p 8501:8501 -e API_URL="http://host.docker.internal:8000/predict" vertebra-segmentation-front #para local
# para la VM -e API_URL="http://172.31.xx.xx:8001"