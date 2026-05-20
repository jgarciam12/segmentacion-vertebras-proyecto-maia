import streamlit as st
import requests
from PIL import Image
import io
import os
import base64
import time
import pandas as pd # <-- NUEVO: Para crear el Dataframe interactivo

# -------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------
st.set_page_config(
    page_title="MAIA - Segmentación",
    page_icon="🦴",
    layout="wide"
)

# -------------------------------------------------
# INYECCIÓN DE CSS PERSONALIZADO
# -------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #FFFFFF; }
    .stApp > header + div h1, .stApp > header + div h2, .stApp > header + div h3, .stApp > header + div p, .stApp > header + div span { color: #1e293b !important; }
    [data-testid="stFileUploadDropzone"] div div { display: none !important; }
    [data-testid="stFileUploadDropzone"] { border: 2px dashed #e2e8f0; padding: 10px; border-radius: 10px; background-color: #f8fafc; }
    .title { font-size: 40px; font-weight: bold; color: #1e293b; }
    .subtitle { font-size: 18px; color: #64748b; margin-bottom: 20px; }
    
    [data-testid="stSidebar"] { background-color: #1e293b !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] li, [data-testid="stSidebar"] div { color: #ffffff !important; }
    
    /* Hacer que la tabla ocupe todo el ancho disponible */
    [data-testid="stDataFrame"] { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">MAIA - SEGMENTACIÓN AUTOMÁTICA DE VÉRTEBRAS</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Herramienta de apoyo diagnóstico mediante Pipeline Híbrido Asistido</div>', unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
    with col_logo2:
        logo_posibles_rutas = [os.path.join("deployment", "front", "logo.png"), "logo.png"]
        logo_path = next((r for r in logo_posibles_rutas if os.path.exists(r)), None)
        if logo_path: 
            st.image(logo_path, use_container_width=True)
    
    st.header("⚙️ Configuración")
    default_url = os.getenv("API_URL", "http://api:8000/predict")
    api_url = st.text_input("URL de la API", default_url)

    alpha = st.slider(
        "Transparencia de Superposición",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    st.divider()
    st.markdown("### ℹ️ Información")
    st.write("- **Detección:** YOLOv8 Custom\n- **Segmentación:** MedSAM ViT-B\n- **Arquitectura:** Pipeline Híbrido Asistido")

# -------------------------------------------------
# CARGA DE IMAGEN
# -------------------------------------------------
st.markdown("### 📤 Cargar imagen para análisis")
uploaded_file = st.file_uploader(
    "Selecciona una radiografía AP o Lateral:",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Radiografía Original")
        st.image(image, use_container_width=True)
        st.caption(f"Resolución de entrada: {image.size[0]} x {image.size[1]} px")
        
        st.write("") 
        ejecutar_analisis = st.button("Ejecutar Análisis", use_container_width=True)

    with col2:
        st.markdown("### Vértebra segmentada")
        
        if ejecutar_analisis:
            start_time = time.time()

            with st.spinner("Procesando radiografía con IA..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{api_url}?alpha={alpha}", files=files, timeout=60)

                    if response.status_code == 200:
                        data = response.json()
                        elapsed = round(time.time() - start_time, 2)

                        # Decodificamos la imagen
                        fusion_img = Image.open(io.BytesIO(base64.b64decode(data["fusion"])))
                        st.image(fusion_img, use_container_width=True)
                        
                        # -------------------------------------------------
                        # TABLA INTERACTIVA (Idéntica a tu imagen de referencia)
                        # -------------------------------------------------
                        st.write("")
                        st.markdown("### Resultados")
                        
                        if "tabla" in data and len(data["tabla"]) > 0:
                            # 1. Indicadores principales
                            m1, m2 = st.columns(2)
                            m1.metric(label="Total de vértebras detectadas", value=data["total_vertebras"])
                            m2.metric(label="Tiempo de procesamiento", value=f"{elapsed} s")
                            
                            # 2. Dataframe / Tabla interactiva
                            df = pd.DataFrame(data["tabla"])
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.warning("⚠️ No se detectaron vértebras o falta actualizar el backend.")

                    else:
                        st.error(f"Error en API ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Error de conexión: {e}")
        else:
            st.info("Presiona el botón de 'Ejecutar Análisis' debajo de la radiografía para procesar la imagen.")