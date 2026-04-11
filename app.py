import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Segmentación de Vértebras", layout="wide")

st.title("🦴 Segmentación Automática de Vértebras")
st.write("Sube una imagen de rayos X o TAC para detectar las vértebras.")

# Barra lateral para configuración
st.sidebar.header("Configuración")
api_url = st.sidebar.text_input("URL de la API", "http://127.0.0.1:8000/predict")

uploaded_file = st.file_uploader("Elige una imagen...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen original
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True)
    
    if st.button("Segmentar Vértebras"):
        with st.spinner('Procesando con la IA...'):
            try:
                # Enviar imagen a la API
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(api_url, files=files)
                
                if response.status_code == 200:
                    # Leer la máscara devuelta
                    mask_image = Image.open(io.BytesIO(response.content))
                    
                    with col2:
                        st.subheader("Resultado (Máscara)")
                        st.image(mask_image, use_column_width=True)
                        
                    st.success("¡Segmentación completada!")
                else:
                    st.error(f"Error en la API: {response.status_code}")
            except Exception as e:
                st.error(f"No se pudo conectar con la API. Asegúrate de que 'api.py' esté corriendo. Error: {e}")