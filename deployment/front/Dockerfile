FROM python:3.11-slim

# Evitar archivos .pyc y habilitar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt

# Copiar las dependencias e instalar
COPY deployment/front/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del front manteniendo la estructura
COPY deployment/front/ /opt/deployment/front/

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando corregido para que encuentre el ejecutable
CMD ["python", "-m", "streamlit", "run", "deployment/front/app.py", "--server.port=8501", "--server.address=0.0.0.0"]