FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# El frontend se arranca con este comando
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]