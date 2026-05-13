import streamlit as st
import requests
from PIL import Image
import io
import os
import numpy as np
import time

# -------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------

st.set_page_config(
    page_title="Segmentación de Vértebras",
    page_icon="🦴",
    layout="wide"
)

# -------------------------------------------------
# ESTILOS PERSONALIZADOS
# -------------------------------------------------

st.markdown(
    """
    <style>
    .stApp { background-color: #FFFFFF; }
    h1, h2, h3, p, span, .stMarkdown { color: #1e293b !important; }
    
    /* Ocultamos el texto interno rebelde del cargador */
    [data-testid="stFileUploadDropzone"] div div {
        display: none !important;
    }
    
    /* Personalizamos el botón de examinar que queda huérfano */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #e2e8f0;
        padding: 10px;
        border-radius: 10px;
        background-color: #f8fafc;
    }

    .title { font-size: 40px; font-weight: bold; color: #1e293b; }
    .subtitle { font-size: 18px; color: #64748b; margin-bottom: 20px; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown('<div class="title">🦴 Segmentación Automática de Vértebras</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Sistema de segmentación médica asistido por Inteligencia Artificial</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:
    # Cargar el logo de MAIA si existe en la carpeta
    logo_path = "logo.png" # Asegúrate de que el archivo se llame así y esté en esta carpeta
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.warning("⚠️ Logo no encontrado. Coloca 'logo.png' en la misma carpeta que app.py")

    st.header("⚙️ Configuración")

    default_url = os.getenv(
        "API_URL",
        "http://api:8000/predict" # Ajustado para que funcione perfecto en Docker
    )

    api_url = st.text_input("URL de la API", default_url)

    alpha = st.slider(
        "Transparencia de Superposición",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    st.divider()

    st.markdown("### ℹ️ Información")
    st.write("- **Modelo:** Hold por el momento")
    st.write("- **Framework:** PyTorch")
    st.write("- **Backend:** FastAPI")

# -------------------------------------------------
# CARGA DE IMAGEN
# -------------------------------------------------

st.markdown("### 📤 Cargar imagen para análisis")
uploaded_file = st.file_uploader(
    "Selecciona una radiografía (PNG, JPG, JPEG) para que la IA realice la segmentación automática:",
    type=["png", "jpg", "jpeg"],
    label_visibility="visible" # Esto asegura que nuestro texto en español sea el protagonista
)

# -------------------------------------------------
# PROCESAMIENTO
# -------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 1, 1])

    # ---------------------------------------------
    # IMAGEN ORIGINAL
    # ---------------------------------------------

    with col1:
        st.markdown("### Imagen Original")
        st.image(image, use_container_width=True)
        st.caption(f"Resolución: {image.size[0]} x {image.size[1]} píxeles")

    # ---------------------------------------------
    # BOTÓN SEGMENTAR
    # ---------------------------------------------

    if st.button("🚀 Ejecutar Segmentación"):

        start_time = time.time()

        with st.spinner("Procesando imagen con IA..."):

            try:
                files = {
                    "file": uploaded_file.getvalue()
                }

                response = requests.post(api_url, files=files)

                if response.status_code == 200:

                    mask_image = Image.open(
                        io.BytesIO(response.content)
                    ).convert("L")

                    # ---------------------------------
                    # MOSTRAR MÁSCARA
                    # ---------------------------------

                    with col2:
                        st.markdown("### Máscara Segmentada")
                        st.image(mask_image, use_container_width=True)

                    # ---------------------------------
                    # CREAR SUPERPOSICIÓN (OVERLAY)
                    # ---------------------------------

                    original_np = np.array(image)
                    mask_np = np.array(mask_image)

                    red_mask = np.zeros_like(original_np)
                    red_mask[:, :, 0] = mask_np

                    overlay = (
                        original_np * (1 - alpha)
                        + red_mask * alpha
                    ).astype(np.uint8)

                    overlay_image = Image.fromarray(overlay)

                    with col3:
                        st.markdown("### Superposición")
                        st.image(overlay_image, use_container_width=True)

                    # ---------------------------------
                    # MÉTRICAS
                    # ---------------------------------

                    elapsed = round(time.time() - start_time, 2)

                    st.success("✅ Segmentación completada con éxito")

                    metric1, metric2, metric3 = st.columns(3)

                    with metric1:
                        st.metric("Tiempo de Inferencia", f"{elapsed}s")

                    with metric2:
                        area = np.sum(mask_np > 0)
                        st.metric("Píxeles Segmentados", int(area))

                    with metric3:
                        coverage = round(
                            area / (mask_np.shape[0] * mask_np.shape[1]) * 100,
                            2
                        )
                        st.metric("Cobertura Total", f"{coverage}%")

                    # ---------------------------------
                    # DESCARGAS
                    # ---------------------------------

                    buffer = io.BytesIO()
                    mask_image.save(buffer, format="PNG")

                    st.download_button(
                        label="⬇️ Descargar Máscara",
                        data=buffer.getvalue(),
                        file_name="mascara_segmentada.png",
                        mime="image/png"
                    )

                else:
                    st.error(
                        f"Error en el servidor de la API. Código de estado: {response.status_code}"
                    )

            except Exception as e:
                st.error(f"Error de conexión con la IA. Verifica que el backend esté funcionando. Detalle: {e}")