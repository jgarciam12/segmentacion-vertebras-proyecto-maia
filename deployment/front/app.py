import streamlit as st
import requests
from PIL import Image
import io
import os
import base64
import time
import pandas as pd

# -------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------
st.set_page_config(
    page_title="MAIA - Segmentación Vertebral",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CSS PROFESIONAL
# -------------------------------------------------
st.markdown("""
<style>

/* Fondo general */
.stApp {
    background-color: #f8fafc;
}

/* Header principal */
.main-title {
    font-size: 30px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 15px;
    color: #64748b;
    margin-bottom: 20px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Upload */
[data-testid="stFileUploader"] {
    background-color: white;
    padding: 10px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

/* Cards métricas */
.metric-card {
    background-color: white;
    padding: 15px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    text-align: center;
    box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
}

.metric-title {
    font-size: 14px;
    color: #64748b;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #0f172a;
}

/* Viewer imágenes */
.viewer-box {
    background-color: white;
    padding: 10px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Botón */
.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 48px;
    background-color: #2563eb;
    color: white;
    border: none;
    font-weight: 600;
}

.stButton button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    '<div class="main-title">MAIA - SEGMENTACIÓN AUTOMÁTICA DE VÉRTEBRAS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Pipeline híbrido de Inteligencia Artificial usando YOLOv8 + MedSAM para apoyo diagnóstico radiológico</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:

    st.markdown("## ⚙️ Configuración")

    logo_posibles_rutas = [
        os.path.join("deployment", "front", "logo.png"),
        "logo.png"
    ]

    logo_path = next((r for r in logo_posibles_rutas if os.path.exists(r)), None)

    if logo_path:
        st.image(logo_path, use_container_width=True)

    st.divider()

    default_url = os.getenv(
        "API_URL",
        "http://localhost:8000/predict"
    )

    api_url = st.text_input(
        "URL API",
        default_url
    )

    alpha = st.slider(
        "Transparencia",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    st.divider()

    st.markdown("### ℹ️ Modelo")

    st.caption("""
    - YOLOv8 Custom Detector  
    - MedSAM ViT-B  
    - Segmentación híbrida  
    - Pipeline médico IA
    """)

# -------------------------------------------------
# CONTROLES SUPERIORES
# -------------------------------------------------
top1, top2 = st.columns([4, 1])

with top1:
    uploaded_file = st.file_uploader(
        "📤 Cargar radiografía AP o lateral",
        type=["png", "jpg", "jpeg"]
    )

with top2:
    st.write("")
    st.write("")
    ejecutar_analisis = st.button(
        "🚀 Ejecutar análisis"
    )

# -------------------------------------------------
# SI HAY IMAGEN
# -------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # -------------------------------------------------
    # PLACEHOLDER RESULTADOS
    # -------------------------------------------------
    resultado_container = st.container()

    # -------------------------------------------------
    # INFERENCIA
    # -------------------------------------------------
    if ejecutar_analisis:

        with st.spinner("Procesando radiografía con modelos de IA..."):

            start_time = time.time()

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{api_url}?alpha={alpha}",
                    files=files,
                    timeout=300
                )

                if response.status_code == 200:

                    data = response.json()

                    elapsed = round(
                        time.time() - start_time,
                        2
                    )

                    fusion_img = Image.open(
                        io.BytesIO(
                            base64.b64decode(data["fusion"])
                        )
                    )

                    df = pd.DataFrame(data["tabla"])

                    # -------------------------------------------------
                    # KPIs
                    # -------------------------------------------------
                    k1, k2, k3 = st.columns(3)

                    with k1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                Vértebras Detectadas
                            </div>
                            <div class="metric-value">
                                {data["total_vertebras"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with k2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                Tiempo Inferencia
                            </div>
                            <div class="metric-value">
                                {elapsed}s
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with k3:

                        avg_conf = round(
                            df["Confianza (%)"].mean(),
                            2
                        )

                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                Confianza Promedio
                            </div>
                            <div class="metric-value">
                                {avg_conf}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("")

                    # -------------------------------------------------
                    # TABS
                    # -------------------------------------------------
                    tab1, tab2 = st.tabs([
                        "🖼️ Visualización IA",
                        "📊 Resultados"
                    ])

                    # -------------------------------------------------
                    # TAB VISUALIZACIÓN
                    # -------------------------------------------------
                    with tab1:

                        col1, col2 = st.columns(
                            2,
                            gap="medium"
                        )

                        with col1:

                            st.markdown("### Radiografía Original")

                            with st.container():
                                st.markdown(
                                    '<div class="viewer-box">',
                                    unsafe_allow_html=True
                                )

                                st.image(
                                    image,
                                    use_container_width=True
                                )

                                st.markdown(
                                    '</div>',
                                    unsafe_allow_html=True
                                )

                        with col2:

                            st.markdown("### Segmentación MedSAM")

                            with st.container():
                                st.markdown(
                                    '<div class="viewer-box">',
                                    unsafe_allow_html=True
                                )

                                st.image(
                                    fusion_img,
                                    use_container_width=True
                                )

                                st.markdown(
                                    '</div>',
                                    unsafe_allow_html=True
                                )

                    # -------------------------------------------------
                    # TAB RESULTADOS
                    # -------------------------------------------------
                    with tab2:

                        st.markdown(
                            "### Resultados del análisis"
                        )

                        st.dataframe(
                            df,
                            use_container_width=True,
                            height=500,
                            hide_index=True
                        )

                        csv = df.to_csv(index=False).encode("utf-8")

                        st.download_button(
                            label="⬇️ Descargar resultados CSV",
                            data=csv,
                            file_name="resultados_segmentacion.csv",
                            mime="text/csv"
                        )

                else:

                    st.error(
                        f"Error API ({response.status_code})"
                    )

                    st.code(response.text)

            except Exception as e:

                st.error(f"Error de conexión: {e}")

    # -------------------------------------------------
    # ESTADO INICIAL
    # -------------------------------------------------
    else:

        st.info(
            "Cargue una radiografía y presione "
            "'Ejecutar análisis'"
        )
