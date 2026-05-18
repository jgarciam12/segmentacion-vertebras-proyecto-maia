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
    
    [data-testid="stFileUploadDropzone"] div div {
        display: none !important;
    }
    
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
    # Ajuste de ruta del logo para compatibilidad total
    logo_posibles_rutas = [
        os.path.join("deployment", "front", "logo.png"),
        "logo.png"
    ]
    
    logo_path = None
    for ruta in logo_posibles_rutas:
        if os.path.exists(ruta):
            logo_path = ruta
            break

    if logo_path:
        st.image(logo_path, use_container_width=True)
    else:
        st.warning("⚠️ Logo no encontrado. Verifica la carpeta deployment/front/")

    st.header("⚙️ Configuración")

    # Priorizamos la conexión interna de Docker
    default_url = os.getenv(
        "API_URL",
        "http://api:8000/predict" 
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
    st.write("- **Modelo:** Pipeline Híbrido (YOLOv8 + MedSAM)")
    st.write("- **Framework:** PyTorch & Ultralytics")
    st.write("- **Backend:** FastAPI")

# -------------------------------------------------
# CARGA DE IMAGEN
# -------------------------------------------------

st.markdown("### 📤 Cargar imagen para análisis")
uploaded_file = st.file_uploader(
    "Selecciona una radiografía (PNG, JPG, JPEG) para análisis automático:",
    type=["png", "jpg", "jpeg"],
    label_visibility="visible"
)

# -------------------------------------------------
# PROCESAMIENTO
# -------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### Imagen Original")
        st.image(image, use_container_width=True)
        st.caption(f"Resolución: {image.size[0]} x {image.size[1]} píxeles")

    if st.button("🚀 Ejecutar Segmentación"):

        start_time = time.time()

        with st.spinner("Procesando imagen con IA (Puede tardar un poco en CPU)..."):

            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }

                # Aumentamos el timeout a 60 segundos por si MedSAM y YOLO tardan
                response = requests.post(api_url, files=files, timeout=60)

                if response.status_code == 200:

                    # 1. Leer como imagen con color y transparencia (RGBA)
                    mask_image = Image.open(io.BytesIO(response.content)).convert("RGBA")

                    # Si la máscara no tiene el mismo tamaño, la redimensionamos
                    if mask_image.size != image.size:
                        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)

                    # --- COLUMNA 2: MÁSCARA SEGMENTADA ---
                    with col2:
                        st.markdown("### Máscara Segmentada")
                        # Creamos un fondo negro para que resalten los colores
                        black_bg = Image.new("RGB", image.size, (0, 0, 0))
                        black_bg.paste(mask_image, mask=mask_image)
                        st.image(black_bg, use_container_width=True)

                    # --- COLUMNA 3: SUPERPOSICIÓN (OVERLAY) ---
                    original_np = np.array(image.convert("RGB")).astype(np.float32)
                    mask_np = np.array(mask_image).astype(np.float32)

                    # Separar los canales de color (RGB) y el canal de transparencia (Alpha)
                    mask_rgb = mask_np[:, :, :3]
                    # Aplicamos tu slider (alpha) a la transparencia original de la API
                    mask_alpha = (mask_np[:, :, 3] / 255.0) * alpha 

                    # Mezclamos la radiografía original con los colores de la máscara
                    overlay = np.zeros_like(original_np)
                    for c in range(3): # Iteramos sobre R, G, B
                        overlay[:, :, c] = original_np[:, :, c] * (1 - mask_alpha) + mask_rgb[:, :, c] * mask_alpha

                    overlay_image = Image.fromarray(overlay.astype(np.uint8))

                    with col3:
                        st.markdown("### Superposición")
                        st.image(overlay_image, use_container_width=True)

                    # --- MÉTRICAS ---
                    elapsed = round(time.time() - start_time, 2)
                    st.success("✅ Segmentación completada con éxito")

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Inferencia", f"{elapsed}s")
                    with m2:
                        # Contamos píxeles donde hay color (donde el alpha > 0)
                        area = np.sum(mask_np[:, :, 3] > 0)
                        st.metric("Píxeles", int(area))
                    with m3:
                        total_pixels = mask_np.shape[0] * mask_np.shape[1]
                        coverage = round((area / total_pixels) * 100, 2)
                        st.metric("Cobertura", f"{coverage}%")

                    # --- DESCARGA ---
                    buf = io.BytesIO()
                    mask_image.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Descargar Máscara",
                        data=buf.getvalue(),
                        file_name="segmentacion_maia.png",
                        mime="image/png"
                    )

                else:
                    st.error(f"Error en API ({response.status_code}): {response.text}")

            except Exception as e:
                st.error(f"Error de conexión: {e}")