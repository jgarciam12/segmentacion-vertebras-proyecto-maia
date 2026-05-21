# -*- coding: utf-8 -*-
r"""
generar_curvas_cobb_y_overlays_reproducible.py

Autor
-----
David Felipe Landinez

Objetivo
--------
Generar de forma reproducible la curva escoliótica, calcular el ángulo de Cobb
y crear overlays finales directamente desde radiografías originales y máscaras
binarias de la columna.

A diferencia de versiones anteriores, este script NO parte de curvas previamente
calculadas. El flujo completo es:

    máscara binaria -> curva en pixeles -> ápice e inflexiones -> Cobb -> overlay

Estructura esperada del dataset
-------------------------------
DATASET_ROOT/
│
├── Scoliosis/
│   └── S_{ID}.jpg / png
│
├── LabelBinaryJPG/
│   └── Label_{ID}.jpg / png
│
└── RadiographMetrics/
    ├── curvas_en_pixeles/
    ├── overlays/
    ├── metricas_cobb_resumen.csv
    ├── metricas_cobb_diagnostico.csv
    ├── diccionario_metricas_cobb.json
    └── README_metricas_cobb.txt

Salidas principales
-------------------
1. RadiographMetrics/curvas_en_pixeles/curva_pixeles_{ID}.csv
   CSV con los puntos de la curva: x_px, y_px

2. RadiographMetrics/metricas_cobb_resumen.csv
   Tabla esencial con:
   - punto de inflexión superior
   - punto de inflexión inferior
   - pendiente de la recta superior
   - pendiente de la recta inferior
   - punto del ápice
   - ángulo de Cobb
   - ruta de la curva en pixeles
   - ruta del overlay

3. RadiographMetrics/overlays/overlay_cobb_{ID}.png
   Imagen con radiografía, curva, rectas perpendiculares, ápice e inflexiones.

Dependencias
------------
- Python 3.x
- numpy
- pandas
- opencv-python

Uso
---
1. Cambiar DATASET_ROOT si el dataset está en otra ubicación.
2. Ejecutar:

    python generar_curvas_cobb_y_overlays_reproducible.py

Notas metodológicas
-------------------
- La curva se obtiene desde la máscara binaria usando centroides mediales por fila.
- Los centroides se ponderan con distance transform para favorecer el eje central.
- La curva se interpola a una muestra por pixel vertical.
- Se suaviza con filtros medianos y gaussianos.
- El ápice no se selecciona solo por máxima desviación lateral. Se evalúan varios
  candidatos y se elige la curva principal con un score geométrico.
- Se penalizan candidatos demasiado altos para evitar ápices cervicales falsos.
"""

from pathlib import Path
import json
import re

import cv2
import numpy as np
import pandas as pd


# =========================================================
# 1) CONFIGURACIÓN GENERAL
# =========================================================
# Ruta raíz del dataset. Si otra persona usa el script, solo debería cambiar
# esta variable para apuntar a su copia local del dataset.
DATASET_ROOT = Path(
    r"Ruta del dataset"
)

# Carpetas de entrada
RADIOS_DIR = DATASET_ROOT / "Scoliosis"
MASKS_DIR = DATASET_ROOT / "LabelBinaryJPG"

# Carpeta de salida
RADIOMETRICS_DIR = DATASET_ROOT / "RadiographMetrics"
CURVES_DIR = RADIOMETRICS_DIR / "curvas_en_pixeles"
OVERLAYS_DIR = RADIOMETRICS_DIR / "overlays"

# Archivos de salida
OUTPUT_METRICS_CSV = RADIOMETRICS_DIR / "metricas_cobb_resumen.csv"
OUTPUT_DIAGNOSTIC_CSV = RADIOMETRICS_DIR / "metricas_cobb_diagnostico.csv"
OUTPUT_DICTIONARY_JSON = RADIOMETRICS_DIR / "diccionario_metricas_cobb.json"
README_PATH = RADIOMETRICS_DIR / "README_metricas_cobb.txt"

# Crear carpetas de salida si no existen
RADIOMETRICS_DIR.mkdir(parents=True, exist_ok=True)
CURVES_DIR.mkdir(parents=True, exist_ok=True)
OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2) PARÁMETROS DE PREPROCESAMIENTO Y CURVA
# =========================================================
MORPH_CLOSE_K = 3
MORPH_OPEN_K = 3
KEEP_LARGEST_COMPONENT = True
MIN_ROWS_VALID = 12
END_TAPER_FRAC = 0.12
BBOX_PAD = 10


# =========================================================
# 3) PARÁMETROS DEL COBB ROBUSTO
# =========================================================
# y_norm: 0.0 = extremo superior de la curva; 1.0 = extremo inferior.
CANDIDATE_Y_MIN = 0.22
CANDIDATE_Y_MAX = 0.92
INFLECTION_TOP_MIN = 0.12
MIN_SPAN_FRAC = 0.16
MIN_DEV_REL = 0.22
MIN_COBB_DEG = 7.0
CENTER_POS_MIN = 0.22
CENTER_POS_MAX = 0.78


# =========================================================
# 4) UTILIDADES DE ARCHIVOS
# =========================================================
def extract_numeric_id(path: Path):
    """Extrae el último número encontrado en el nombre del archivo."""
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return None
    return int(nums[-1])


def find_image_for_id(folder: Path, patient_id: int, prefixes=("S", "Label")):
    """Busca una imagen asociada a un ID con nombres comunes."""
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    candidates = []

    for prefix in prefixes:
        for ext in extensions:
            candidates.extend([
                folder / f"{prefix}_{patient_id}{ext}",
                folder / f"{prefix}_{patient_id:03d}{ext}",
            ])

    for p in candidates:
        if p.exists():
            return p

    patterns = []
    for ext in extensions:
        patterns.extend([
            f"*{patient_id}{ext}",
            f"*{patient_id:03d}{ext}",
        ])

    matches = []
    for pat in patterns:
        matches.extend(folder.glob(pat))

    matches = sorted(set(matches))
    return matches[0] if matches else None


def discover_cases():
    """Encuentra pares radiografía-máscara usando LabelBinaryJPG como referencia."""
    mask_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        mask_files.extend(MASKS_DIR.glob(ext))

    pairs = []
    for mask_path in sorted(mask_files):
        patient_id = extract_numeric_id(mask_path)
        if patient_id is None:
            continue

        rx_path = find_image_for_id(RADIOS_DIR, patient_id, prefixes=("S", "RX", "Radiograph"))
        if rx_path is None:
            print(f"[WARN] No se encontró radiografía para ID={patient_id}")
            continue

        pairs.append((patient_id, rx_path, mask_path))

    pairs.sort(key=lambda x: x[0])
    return pairs


# =========================================================
# 5) LECTURA Y PREPROCESAMIENTO DE MÁSCARAS
# =========================================================
def read_binary_mask(mask_path: Path):
    """
    Lee una máscara binaria de forma robusta.

    Soporta:
    - columna=1 y fondo=2;
    - columna blanca y fondo negro;
    - máscara invertida.
    """
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"No se pudo leer la máscara: {mask_path}")

    unique_vals = np.unique(img)

    # Caso heredado del dataset original: columna = 1, fondo = 2.
    if 1 in unique_vals and 2 in unique_vals and unique_vals.size <= 5:
        return img == 1

    if unique_vals.size == 1:
        raise ValueError(f"Máscara sin variación: {mask_path}")

    # Umbral de Otsu para separar columna/fondo aunque venga como JPG.
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    foreground_white = bw > 0
    n_white = int(np.sum(foreground_white))
    n_black = int(foreground_white.size - n_white)

    # La columna suele ocupar menos área que el fondo.
    if n_white <= n_black:
        mask = foreground_white
    else:
        mask = ~foreground_white

    return mask.astype(bool)


def fill_holes(mask_bool: np.ndarray):
    """Rellena huecos internos con flood fill."""
    m = (mask_bool.astype(np.uint8)) * 255
    h, w = m.shape

    flood = m.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = m | flood_inv
    return filled > 0


def keep_largest_component(mask_bool: np.ndarray):
    """Conserva únicamente el componente conectado más grande."""
    m = mask_bool.astype(np.uint8)
    num, labels = cv2.connectedComponents(m)

    if num <= 1:
        return mask_bool

    areas = []
    for lab in range(1, num):
        areas.append(np.sum(labels == lab))

    best_label = 1 + int(np.argmax(areas))
    return labels == best_label


def preprocess_mask(mask_bool: np.ndarray):
    """Aplica limpieza morfológica básica."""
    m = (mask_bool.astype(np.uint8)) * 255

    if MORPH_CLOSE_K > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    if MORPH_OPEN_K > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    mask = m > 0
    mask = fill_holes(mask)

    if KEEP_LARGEST_COMPONENT:
        mask = keep_largest_component(mask)

    return mask.astype(bool)


def cutoff_bbox(mask_bool: np.ndarray, pad: int = BBOX_PAD):
    """Obtiene un recorte alrededor de la máscara."""
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        raise ValueError("La máscara quedó vacía tras el preprocesamiento.")

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    h, w = mask_bool.shape
    y0 = max(0, y0 - pad)
    y1 = min(h - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w - 1, x1 + pad)

    return slice(y0, y1 + 1), slice(x0, x1 + 1)


# =========================================================
# 6) EXTRACCIÓN DE CURVA DESDE LA MÁSCARA
# =========================================================
def row_medial_centroid(mask_bool: np.ndarray, dist=None):
    """Calcula un punto medial por cada fila de la máscara."""
    xs_out, ys_out, weights_out = [], [], []

    h, _ = mask_bool.shape
    for y in range(h):
        cols = np.flatnonzero(mask_bool[y, :])
        if cols.size == 0:
            continue

        if dist is not None:
            weights = dist[y, cols].astype(np.float64)
            if np.sum(weights) > 0:
                x = float(np.average(cols, weights=weights))
                q = float(np.sum(weights)) / (cols.size + 1e-6)
            else:
                x = float(np.median(cols))
                q = float(cols.size)
        else:
            x = float(np.median(cols))
            q = float(cols.size)

        xs_out.append(x)
        ys_out.append(float(y))
        weights_out.append(q)

    return (
        np.asarray(xs_out, dtype=float),
        np.asarray(ys_out, dtype=float),
        np.asarray(weights_out, dtype=float),
    )


def moving_median(x: np.ndarray, k: int = 5):
    """Filtro mediano 1D implementado solo con NumPy."""
    x = np.asarray(x, dtype=float)

    if k < 3:
        return x.copy()
    if k % 2 == 0:
        k += 1

    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)

    for i in range(len(x)):
        out[i] = np.median(xp[i:i + k])

    return out


def gaussian_smooth_1d(x: np.ndarray, sigma: float):
    """Suavizado gaussiano 1D usando OpenCV."""
    v = np.asarray(x, dtype=np.float32).reshape(-1, 1)
    out = cv2.GaussianBlur(v, ksize=(0, 0), sigmaX=float(sigma)).ravel()
    return out.astype(np.float64)


def enforce_straight_ends(y: np.ndarray, x: np.ndarray, frac: float = END_TAPER_FRAC):
    """Regulariza los extremos superior e inferior de la curva."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)

    n = len(y)
    if n < 10:
        return x.copy()

    w = max(5, int(frac * n))
    w = min(w, max(5, n // 3))

    dxy = np.zeros_like(x, float)
    dxy[1:-1] = (x[2:] - x[:-2]) / (y[2:] - y[:-2] + 1e-12)
    dxy[0] = (x[1] - x[0]) / (y[1] - y[0] + 1e-12)
    dxy[-1] = (x[-1] - x[-2]) / (y[-1] - y[-2] + 1e-12)

    def hermite_segment(y_a, x_a, m_a, y_b, x_b, m_b, y_seg):
        L = y_b - y_a
        t = (y_seg - y_a) / (L + 1e-12)
        t = np.clip(t, 0.0, 1.0)

        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2

        return h00 * x_a + h10 * (L * m_a) + h01 * x_b + h11 * (L * m_b)

    x_new = x.copy()

    # Extremo superior.
    i0 = 0
    ij = min(w, n - 2)
    x_flat_start = float(np.median(x[:w]))
    m0 = 0.0
    mj = 0.9 * float(dxy[ij])
    idx = np.arange(0, ij + 1)
    x_new[idx] = hermite_segment(y[i0], x_flat_start, m0, y[ij], x[ij], mj, y[idx])

    # Extremo inferior.
    in_ = max(0, n - 1 - w)
    iN = n - 1
    x_flat_end = float(np.median(x[-w:]))
    mi = 0.9 * float(dxy[in_])
    mN = 0.0
    idx = np.arange(in_, iN + 1)
    x_new[idx] = hermite_segment(y[in_], x[in_], mi, y[iN], x_flat_end, mN, y[idx])

    return x_new


def extract_curve_from_mask(mask_bool: np.ndarray, image_shape):
    """Genera la curva escoliótica en coordenadas absolutas de imagen."""
    h_img, w_img = image_shape[:2]

    if mask_bool.shape != (h_img, w_img):
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8),
            (w_img, h_img),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    mask_pp = preprocess_mask(mask_bool)

    sly, slx = cutoff_bbox(mask_pp, pad=BBOX_PAD)
    m_crop = mask_pp[sly, slx]

    dist = cv2.distanceTransform((m_crop.astype(np.uint8)) * 255, cv2.DIST_L2, 5)
    xs, ys, wr = row_medial_centroid(m_crop, dist=dist)

    if ys.size < MIN_ROWS_VALID:
        raise ValueError(f"Segmentación insuficiente: solo {ys.size} filas válidas.")

    y_min, y_max = int(ys.min()), int(ys.max())
    y_full = np.arange(y_min, y_max + 1, dtype=float)
    x_interp = np.interp(y_full, ys, xs)

    x_s = moving_median(x_interp, k=5)

    sigma_base = float(np.clip(0.025 * len(x_s), 3.0, 18.0))
    x_s = gaussian_smooth_1d(x_s, sigma=sigma_base)

    x_s = enforce_straight_ends(y_full, x_s, frac=END_TAPER_FRAC)
    x_s = gaussian_smooth_1d(x_s, sigma=1.0)
    x_s = enforce_straight_ends(y_full, x_s, frac=END_TAPER_FRAC)

    xg = x_s + slx.start
    yg = y_full + sly.start

    info = {
        "bbox_y_start": int(sly.start),
        "bbox_y_stop": int(sly.stop),
        "bbox_x_start": int(slx.start),
        "bbox_x_stop": int(slx.stop),
        "num_curve_points": int(len(xg)),
        "smoothing_sigma_base": float(sigma_base),
    }

    return xg.astype(float), yg.astype(float), info


def save_curve_csv(patient_id: int, xg: np.ndarray, yg: np.ndarray):
    """Guarda la curva en pixeles como CSV."""
    out_path = CURVES_DIR / f"curva_pixeles_{patient_id}.csv"
    df = pd.DataFrame({"x_px": xg, "y_px": yg})
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# =========================================================
# 7) CÁLCULO ROBUSTO DE COBB
# =========================================================
def safe_grad(x: np.ndarray, y: np.ndarray):
    """Gradiente dx/dy robusto."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if x.size < 3:
        return np.zeros_like(x)

    return np.gradient(x, y)


def compute_csvl_from_curve(xg: np.ndarray, frac_base: float = 0.10):
    """CSVL geométrica auxiliar: mediana de x en la base de la curva."""
    n = len(xg)
    if n < 5:
        return float(xg[0])

    w = max(3, int(round(frac_base * n)))
    return float(np.median(xg[-w:]))


def local_maxima_abs(sig: np.ndarray, min_separation: int = 8, q: float = 0.55):
    """Detecta máximos locales de |sig| con separación mínima."""
    v = np.abs(np.asarray(sig, float))

    if v.size < 5:
        return np.array([int(np.argmax(v))], dtype=int)

    thr = float(np.quantile(v, q))

    cand = []
    for i in range(1, len(v) - 1):
        if v[i] >= thr and v[i] >= v[i - 1] and v[i] >= v[i + 1]:
            cand.append(i)

    if not cand:
        return np.array([int(np.argmax(v))], dtype=int)

    cand = sorted(cand, key=lambda i: v[i], reverse=True)

    kept = []
    for i in cand:
        if all(abs(i - j) >= min_separation for j in kept):
            kept.append(i)

    kept.sort()
    return np.asarray(kept, dtype=int)


def find_inflections_around_index(xg: np.ndarray, yg: np.ndarray, i_center: int, sigma: float = 2.5, guard: int = 2):
    """Busca inflexión superior e inferior alrededor de un candidato a ápice."""
    dx = safe_grad(xg, yg)
    d2x = safe_grad(dx, yg)
    d2x_s = gaussian_smooth_1d(d2x, sigma=sigma)

    sign = np.sign(d2x_s)
    crossings = np.where(sign[:-1] * sign[1:] < 0)[0]
    crossings = crossings[(crossings >= guard) & (crossings < len(xg) - 1 - guard)]

    if crossings.size == 0:
        w = max(5, len(xg) // 6)
        return max(1, i_center - w), min(len(xg) - 2, i_center + w)

    below = crossings[crossings <= i_center]
    above = crossings[crossings >= i_center]

    i_sup = int(below[-1]) if below.size else int(max(1, i_center - max(5, len(xg) // 6)))
    i_inf = int(above[0]) if above.size else int(min(len(xg) - 2, i_center + max(5, len(xg) // 6)))

    return i_sup, i_inf


def local_slope_dxdy(xg: np.ndarray, yg: np.ndarray, i: int, rad: int = 3):
    """Pendiente local dx/dy como mediana en una ventana alrededor del índice."""
    dx = safe_grad(xg, yg)
    a = max(0, i - rad)
    b = min(len(xg) - 1, i + rad)
    return float(np.median(dx[a:b + 1]))


def angle_between_tangents_dxdy(m1: float, m2: float):
    """Ángulo entre dos tangentes expresadas como pendientes dx/dy."""
    den = max(1e-12, 1.0 + m1 * m2)
    return float(np.degrees(np.arctan(abs(m2 - m1) / den)))


def compute_region_weight(y_norm_value: float):
    """Peso anatómico aproximado según la altura del candidato."""
    y = float(y_norm_value)

    if y < 0.22:
        return 0.0
    if y < 0.28:
        return 0.35
    if y < 0.35:
        return 0.70
    if y < 0.80:
        return 1.00
    if y < 0.90:
        return 0.85

    return 0.55


def compute_candidate_score(i_apex: int, xg: np.ndarray, yg: np.ndarray, z_raw: np.ndarray, y_norm: np.ndarray):
    """Evalúa si un candidato a ápice es válido y calcula su score."""
    H = max(1e-6, float(yg[-1] - yg[0]))

    i_inf_sup, i_inf_inf = find_inflections_around_index(xg, yg, i_apex, sigma=2.5)

    if i_inf_inf <= i_inf_sup:
        return None

    m_sup = local_slope_dxdy(xg, yg, i_inf_sup, rad=3)
    m_inf = local_slope_dxdy(xg, yg, i_inf_inf, rad=3)

    cobb_deg = angle_between_tangents_dxdy(m_sup, m_inf)

    span_frac = float((yg[i_inf_inf] - yg[i_inf_sup]) / H)
    dev_rel = float(abs(z_raw[i_apex]) / (np.max(np.abs(z_raw)) + 1e-6))
    y_apex_norm = float(y_norm[i_apex])
    y_inf_sup_norm = float(y_norm[i_inf_sup])
    center_pos = float((yg[i_apex] - yg[i_inf_sup]) / max(1e-6, (yg[i_inf_inf] - yg[i_inf_sup])))

    if y_apex_norm < CANDIDATE_Y_MIN or y_apex_norm > CANDIDATE_Y_MAX:
        return None
    if y_inf_sup_norm < INFLECTION_TOP_MIN:
        return None
    if span_frac < MIN_SPAN_FRAC:
        return None
    if dev_rel < MIN_DEV_REL:
        return None
    if cobb_deg < MIN_COBB_DEG:
        return None
    if center_pos < CENTER_POS_MIN or center_pos > CENTER_POS_MAX:
        return None

    region_weight = compute_region_weight(y_apex_norm)
    if region_weight <= 0.0:
        return None

    span_rel = float(np.clip(span_frac / 0.30, 0.0, 1.0))
    center_weight = float(np.clip(1.0 - abs(center_pos - 0.5) / 0.5, 0.0, 1.0))

    score = (
        cobb_deg
        * (0.55 + 0.45 * dev_rel)
        * (0.45 + 0.55 * span_rel)
        * (0.50 + 0.50 * center_weight)
        * region_weight
    )

    return {
        "score": float(score),
        "i_apex": int(i_apex),
        "i_inf_sup": int(i_inf_sup),
        "i_inf_inf": int(i_inf_inf),
        "m_sup": float(m_sup),
        "m_inf": float(m_inf),
        "cobb_deg": float(cobb_deg),
        "span_frac": float(span_frac),
        "dev_rel": float(dev_rel),
        "y_apex_norm": float(y_apex_norm),
        "y_inf_sup_norm": float(y_inf_sup_norm),
        "center_pos": float(center_pos),
        "region_weight": float(region_weight),
    }


def compute_curve_metrics_robust(xg: np.ndarray, yg: np.ndarray):
    """Calcula ápice, inflexiones, pendientes y Cobb de manera robusta."""
    xg = np.asarray(xg, float)
    yg = np.asarray(yg, float)

    if len(xg) < 10:
        raise ValueError("La curva tiene muy pocos puntos.")

    x_csvl = compute_csvl_from_curve(xg, frac_base=0.10)
    z_raw = xg - x_csvl

    y0, y1 = float(yg[0]), float(yg[-1])
    H = max(1e-6, y1 - y0)
    y_norm = (yg - y0) / H

    z_s = gaussian_smooth_1d(z_raw, sigma=6.0)

    peaks = local_maxima_abs(
        z_s,
        min_separation=max(6, len(z_s) // 18),
        q=0.55,
    )

    candidate_peaks = [
        int(i) for i in peaks
        if CANDIDATE_Y_MIN <= y_norm[i] <= CANDIDATE_Y_MAX
    ]

    if not candidate_peaks:
        candidate_peaks = [
            int(i) for i in peaks
            if 0.18 <= y_norm[i] <= 0.95
        ]

    candidates = []
    for i in candidate_peaks:
        cand = compute_candidate_score(i, xg, yg, z_raw, y_norm)
        if cand is not None:
            candidates.append(cand)

    selection_mode = "robust_candidates"

    if not candidates:
        relaxed = []
        for i in peaks:
            i = int(i)
            if not (0.18 <= y_norm[i] <= 0.95):
                continue

            i_inf_sup, i_inf_inf = find_inflections_around_index(xg, yg, i, sigma=2.5)
            if i_inf_inf <= i_inf_sup:
                continue

            m_sup = local_slope_dxdy(xg, yg, i_inf_sup, rad=3)
            m_inf = local_slope_dxdy(xg, yg, i_inf_inf, rad=3)
            cobb_deg = angle_between_tangents_dxdy(m_sup, m_inf)
            span_frac = float((yg[i_inf_inf] - yg[i_inf_sup]) / H)
            dev_rel = float(abs(z_raw[i]) / (np.max(np.abs(z_raw)) + 1e-6))

            score = (
                cobb_deg
                * (0.55 + 0.45 * dev_rel)
                * (0.45 + 0.55 * np.clip(span_frac / 0.30, 0.0, 1.0))
                * max(0.15, compute_region_weight(y_norm[i]))
            )

            relaxed.append({
                "score": float(score),
                "i_apex": int(i),
                "i_inf_sup": int(i_inf_sup),
                "i_inf_inf": int(i_inf_inf),
                "m_sup": float(m_sup),
                "m_inf": float(m_inf),
                "cobb_deg": float(cobb_deg),
                "span_frac": float(span_frac),
                "dev_rel": float(dev_rel),
                "y_apex_norm": float(y_norm[i]),
                "y_inf_sup_norm": float(y_norm[i_inf_sup]),
                "center_pos": float((yg[i] - yg[i_inf_sup]) / max(1e-6, (yg[i_inf_inf] - yg[i_inf_sup]))),
                "region_weight": float(max(0.15, compute_region_weight(y_norm[i]))),
            })

        if relaxed:
            candidates = relaxed
            selection_mode = "relaxed_fallback"

    if not candidates:
        valid_idx = np.where((y_norm >= 0.20) & (y_norm <= 0.95))[0]
        if valid_idx.size == 0:
            valid_idx = np.arange(len(xg))

        i_apex = int(valid_idx[np.argmax(np.abs(z_raw[valid_idx]))])
        i_inf_sup, i_inf_inf = find_inflections_around_index(xg, yg, i_apex, sigma=2.5)

        m_sup = local_slope_dxdy(xg, yg, i_inf_sup, rad=3)
        m_inf = local_slope_dxdy(xg, yg, i_inf_inf, rad=3)
        cobb_deg = angle_between_tangents_dxdy(m_sup, m_inf)

        best = {
            "score": float(cobb_deg),
            "i_apex": int(i_apex),
            "i_inf_sup": int(i_inf_sup),
            "i_inf_inf": int(i_inf_inf),
            "m_sup": float(m_sup),
            "m_inf": float(m_inf),
            "cobb_deg": float(cobb_deg),
            "span_frac": float((yg[i_inf_inf] - yg[i_inf_sup]) / H),
            "dev_rel": float(abs(z_raw[i_apex]) / (np.max(np.abs(z_raw)) + 1e-6)),
            "y_apex_norm": float(y_norm[i_apex]),
            "y_inf_sup_norm": float(y_norm[i_inf_sup]),
            "center_pos": float((yg[i_apex] - yg[i_inf_sup]) / max(1e-6, (yg[i_inf_inf] - yg[i_inf_sup]))),
            "region_weight": float(compute_region_weight(y_norm[i_apex])),
        }
        selection_mode = "absolute_fallback"

    else:
        best = max(candidates, key=lambda c: c["score"])

    return {
        "i_apex": best["i_apex"],
        "i_inf_sup": best["i_inf_sup"],
        "i_inf_inf": best["i_inf_inf"],
        "x_apex": float(xg[best["i_apex"]]),
        "y_apex": float(yg[best["i_apex"]]),
        "x_inf_sup": float(xg[best["i_inf_sup"]]),
        "y_inf_sup": float(yg[best["i_inf_sup"]]),
        "x_inf_inf": float(xg[best["i_inf_inf"]]),
        "y_inf_inf": float(yg[best["i_inf_inf"]]),
        "m_sup": float(best["m_sup"]),
        "m_inf": float(best["m_inf"]),
        "cobb_deg": float(best["cobb_deg"]),
        "selection_mode": selection_mode,
        "num_detected_peaks": int(len(peaks)),
        "num_candidate_peaks": int(len(candidate_peaks)),
        "num_valid_candidates": int(len(candidates) if candidates else 0),
        "span_frac": float(best["span_frac"]),
        "dev_rel": float(best["dev_rel"]),
        "y_apex_norm": float(best["y_apex_norm"]),
        "y_inf_sup_norm": float(best["y_inf_sup_norm"]),
        "center_pos": float(best["center_pos"]),
        "candidate_score": float(best["score"]),
        "region_weight": float(best["region_weight"]),
    }


# =========================================================
# 8) OVERLAYS ADAPTATIVOS CON OPENCV
# =========================================================
def to_uint8_for_display(img: np.ndarray):
    """Convierte una imagen a uint8 para que el overlay sea estable."""
    if img.dtype == np.uint8:
        return img

    img_f = img.astype(np.float32)
    mn = float(np.min(img_f))
    mx = float(np.max(img_f))

    if mx <= mn:
        return np.zeros_like(img_f, dtype=np.uint8)

    out = 255.0 * (img_f - mn) / (mx - mn)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_radiograph_bgr(path: Path):
    """Carga la radiografía en BGR uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img is None:
        raise IOError(f"No se pudo leer la radiografía: {path}")

    img = to_uint8_for_display(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def clip_point(x: float, y: float, w: int, h: int):
    """Asegura que un punto quede dentro de los límites de la imagen."""
    return (
        int(np.clip(round(x), 0, w - 1)),
        int(np.clip(round(y), 0, h - 1)),
    )


def compute_style(h: int, w: int):
    """Define tamaños de líneas, puntos y texto según el tamaño de cada imagen."""
    max_side = float(max(h, w))

    return {
        "curve_thick": max(1, int(round(max_side * 0.0035))),
        "line_thick": max(1, int(round(max_side * 0.0030))),
        "point_radius": max(3, int(round(max_side * 0.0075))),
        "point_edge": max(1, int(round(max_side * 0.0018))),
        "font": cv2.FONT_HERSHEY_SIMPLEX,
        "font_scale": float(np.clip(max_side / 1100.0, 0.45, 1.20)),
        "font_thick": max(1, int(round(max_side * 0.0022))),
        "text_dx": max(6, int(round(max_side * 0.010))),
        "text_dy": max(6, int(round(max_side * 0.010))),
        "perp_len": float(np.clip(0.38 * h, 90.0, 0.55 * h)),
        "box_margin": max(8, int(round(max_side * 0.012))),
        "box_pad": max(6, int(round(max_side * 0.008))),
    }


def perpendicular_segment_from_dxdy(x0, y0, m_tan_dxdy, total_len, w, h):
    """Segmento perpendicular a una tangente expresada como dx/dy."""
    eps = 1e-8
    m_tan_dxdy = float(m_tan_dxdy)

    if abs(m_tan_dxdy) < eps:
        x1 = x0 - 0.5 * total_len
        x2 = x0 + 0.5 * total_len
        y1 = y2 = y0
    else:
        m_perp = -1.0 / m_tan_dxdy
        dy = 0.5 * total_len
        y1 = y0 - dy
        y2 = y0 + dy
        x1 = x0 + m_perp * (y1 - y0)
        x2 = x0 + m_perp * (y2 - y0)

    return clip_point(x1, y1, w, h), clip_point(x2, y2, w, h)


def draw_label(img, text, org, color, style):
    """Dibuja texto con sombra negra."""
    x, y = org
    font = style["font"]
    fs = style["font_scale"]
    th = style["font_thick"]

    cv2.putText(img, text, (x + 1, y + 1), font, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, fs, color, th, cv2.LINE_AA)


def draw_point_with_edge(img, center, radius, fill_color, edge_color, edge_thick):
    """Dibuja punto con borde."""
    cv2.circle(img, center, radius + max(1, edge_thick), edge_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, center, radius, fill_color, -1, lineType=cv2.LINE_AA)


def draw_info_box(img, patient_id, angle_deg, style):
    """Caja informativa inferior izquierda."""
    text1 = f"Paciente: {patient_id}"
    text2 = f"Cobb: {angle_deg:.1f} deg"

    font = style["font"]
    fs = style["font_scale"]
    th = style["font_thick"]
    margin = style["box_margin"]
    pad = style["box_pad"]

    (w1, h1), b1 = cv2.getTextSize(text1, font, fs, th)
    (w2, h2), b2 = cv2.getTextSize(text2, font, fs, th)

    box_w = max(w1, w2) + 2 * pad
    box_h = h1 + h2 + b1 + b2 + 3 * pad

    h_img, w_img = img.shape[:2]
    x1 = margin
    y2 = h_img - margin
    x2 = min(w_img - 1, x1 + box_w)
    y1 = max(0, y2 - box_h)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), max(1, th), lineType=cv2.LINE_AA)

    tx = x1 + pad
    ty1 = y1 + pad + h1
    ty2 = ty1 + pad + h2 + b1

    draw_label(img, text1, (tx, ty1), (255, 255, 255), style)
    draw_label(img, text2, (tx, ty2), (255, 255, 255), style)


def save_overlay(patient_id: int, radiograph_path: Path, xg: np.ndarray, yg: np.ndarray, metrics: dict):
    """Guarda overlay final con curva, perpendiculares, ápice e inflexiones."""
    img = load_radiograph_bgr(radiograph_path)
    h, w = img.shape[:2]
    style = compute_style(h, w)

    red = (0, 0, 255)
    lime = (0, 255, 0)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)
    black = (0, 0, 0)

    pts = np.column_stack([np.round(xg).astype(int), np.round(yg).astype(int)])
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts = pts.reshape((-1, 1, 2))

    cv2.polylines(img, [pts], isClosed=False, color=red, thickness=style["curve_thick"], lineType=cv2.LINE_AA)

    p1_sup, p2_sup = perpendicular_segment_from_dxdy(metrics["x_inf_sup"], metrics["y_inf_sup"], metrics["m_sup"], style["perp_len"], w, h)
    p1_inf, p2_inf = perpendicular_segment_from_dxdy(metrics["x_inf_inf"], metrics["y_inf_inf"], metrics["m_inf"], style["perp_len"], w, h)

    cv2.line(img, p1_sup, p2_sup, lime, style["line_thick"], lineType=cv2.LINE_AA)
    cv2.line(img, p1_inf, p2_inf, yellow, style["line_thick"], lineType=cv2.LINE_AA)

    p_inf_sup = clip_point(metrics["x_inf_sup"], metrics["y_inf_sup"], w, h)
    p_inf_inf = clip_point(metrics["x_inf_inf"], metrics["y_inf_inf"], w, h)
    p_apex = clip_point(metrics["x_apex"], metrics["y_apex"], w, h)

    draw_point_with_edge(img, p_inf_sup, style["point_radius"], lime, black, style["point_edge"])
    draw_point_with_edge(img, p_inf_inf, style["point_radius"], yellow, black, style["point_edge"])
    draw_point_with_edge(img, p_apex, style["point_radius"], cyan, black, style["point_edge"])

    dx = style["text_dx"]
    dy = style["text_dy"]
    draw_label(img, "Inf. sup.", (p_inf_sup[0] + dx, max(0, p_inf_sup[1] - dy)), lime, style)
    draw_label(img, "Inf. inf.", (p_inf_inf[0] + dx, max(0, p_inf_inf[1] - dy)), yellow, style)
    draw_label(img, "Apice", (p_apex[0] + dx, max(0, p_apex[1] - dy)), cyan, style)

    draw_info_box(img, patient_id, metrics["cobb_deg"], style)

    out_path = OVERLAYS_DIR / f"overlay_cobb_{patient_id}.png"
    ok = cv2.imwrite(str(out_path), img)

    if not ok:
        raise IOError(f"No se pudo guardar el overlay: {out_path}")

    return out_path


# =========================================================
# 9) DICCIONARIO Y README
# =========================================================
def write_dictionary():
    """Guarda un diccionario de campos para facilitar uso por terceros."""
    dictionary = {
        "patient_id": "Identificador numérico del caso.",
        "punto_inflexion_superior_x_px": "Coordenada x del punto de inflexión superior en pixeles.",
        "punto_inflexion_superior_y_px": "Coordenada y del punto de inflexión superior en pixeles.",
        "punto_inflexion_inferior_x_px": "Coordenada x del punto de inflexión inferior en pixeles.",
        "punto_inflexion_inferior_y_px": "Coordenada y del punto de inflexión inferior en pixeles.",
        "pendiente_recta_superior_dxdy": "Pendiente dx/dy de la tangente en la inflexión superior.",
        "pendiente_recta_inferior_dxdy": "Pendiente dx/dy de la tangente en la inflexión inferior.",
        "punto_apice_x_px": "Coordenada x del ápice principal en pixeles.",
        "punto_apice_y_px": "Coordenada y del ápice principal en pixeles.",
        "angulo_cobb_deg": "Ángulo de Cobb estimado en grados.",
        "curva_pixeles_csv": "Ruta relativa al CSV con todos los puntos de la curva.",
        "overlay_cobb_png": "Ruta relativa al overlay generado.",
    }

    with open(OUTPUT_DICTIONARY_JSON, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=2, ensure_ascii=False)


def write_readme():
    """Guarda un README breve en la carpeta RadiographMetrics."""
    txt = f"""METRICAS DE CURVA ESCOLIOTICA Y COBB

Autor:
David Felipe Landinez

Este directorio fue generado de forma reproducible desde:
- radiografias originales en: {RADIOS_DIR}
- mascaras binarias en: {MASKS_DIR}

Archivos principales:
- {OUTPUT_METRICS_CSV.name}
- {OUTPUT_DIAGNOSTIC_CSV.name}
- {OUTPUT_DICTIONARY_JSON.name}
- curvas_en_pixeles/curva_pixeles_ID.csv
- overlays/overlay_cobb_ID.png

La tabla principal contiene solo las variables esenciales:
- punto de inflexion superior
- punto de inflexion inferior
- pendiente de la recta superior
- pendiente de la recta inferior
- punto del apice
- valor del angulo de Cobb
- ruta de la curva en pixeles
- ruta del overlay

Las metricas de diagnostico se dejan separadas para revisar la seleccion
del apice sin mezclar esos campos con la tabla principal.
"""

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(txt)


# =========================================================
# 10) PROCESAMIENTO PRINCIPAL
# =========================================================
def process_case(patient_id: int, rx_path: Path, mask_path: Path):
    """Procesa un caso completo: máscara -> curva -> Cobb -> overlay."""
    rx = cv2.imread(str(rx_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if rx is None:
        raise IOError(f"No se pudo leer la radiografía: {rx_path}")

    image_shape = rx.shape[:2]

    mask = read_binary_mask(mask_path)

    xg, yg, curve_info = extract_curve_from_mask(mask, image_shape=image_shape)
    curve_csv = save_curve_csv(patient_id, xg, yg)

    metrics = compute_curve_metrics_robust(xg, yg)
    overlay_path = save_overlay(patient_id, rx_path, xg, yg, metrics)

    main_row = {
        "patient_id": patient_id,
        "punto_inflexion_superior_x_px": metrics["x_inf_sup"],
        "punto_inflexion_superior_y_px": metrics["y_inf_sup"],
        "punto_inflexion_inferior_x_px": metrics["x_inf_inf"],
        "punto_inflexion_inferior_y_px": metrics["y_inf_inf"],
        "pendiente_recta_superior_dxdy": metrics["m_sup"],
        "pendiente_recta_inferior_dxdy": metrics["m_inf"],
        "punto_apice_x_px": metrics["x_apex"],
        "punto_apice_y_px": metrics["y_apex"],
        "angulo_cobb_deg": metrics["cobb_deg"],
        "curva_pixeles_csv": str(Path("curvas_en_pixeles") / curve_csv.name),
        "overlay_cobb_png": str(Path("overlays") / overlay_path.name),
    }

    diagnostic_row = {
        "patient_id": patient_id,
        "selection_mode": metrics["selection_mode"],
        "num_detected_peaks": metrics["num_detected_peaks"],
        "num_candidate_peaks": metrics["num_candidate_peaks"],
        "num_valid_candidates": metrics["num_valid_candidates"],
        "span_frac": metrics["span_frac"],
        "dev_rel": metrics["dev_rel"],
        "y_apice_norm": metrics["y_apex_norm"],
        "y_inflexion_superior_norm": metrics["y_inf_sup_norm"],
        "apex_relative_position_in_arc": metrics["center_pos"],
        "candidate_score": metrics["candidate_score"],
        "region_weight": metrics["region_weight"],
        **curve_info,
    }

    return main_row, diagnostic_row


def main():
    """Ejecuta el pipeline completo sobre todos los casos encontrados."""
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"No existe DATASET_ROOT: {DATASET_ROOT}")
    if not RADIOS_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de radiografías: {RADIOS_DIR}")
    if not MASKS_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de máscaras: {MASKS_DIR}")

    pairs = discover_cases()

    if not pairs:
        raise FileNotFoundError("No se encontraron pares radiografía-máscara.")

    rows = []
    diagnostics = []
    skipped = []

    print(f"[INFO] Casos encontrados: {len(pairs)}")

    for patient_id, rx_path, mask_path in pairs:
        try:
            main_row, diagnostic_row = process_case(patient_id, rx_path, mask_path)
            rows.append(main_row)
            diagnostics.append(diagnostic_row)
            print(f"[OK] ID={patient_id}")

        except Exception as e:
            skipped.append({
                "patient_id": patient_id,
                "radiograph": str(rx_path),
                "mask": str(mask_path),
                "error": str(e),
            })
            print(f"[ERROR] ID={patient_id}: {e}")

    df_main = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    df_diag = pd.DataFrame(diagnostics).sort_values("patient_id").reset_index(drop=True)

    df_main.to_csv(OUTPUT_METRICS_CSV, index=False, encoding="utf-8-sig")
    df_diag.to_csv(OUTPUT_DIAGNOSTIC_CSV, index=False, encoding="utf-8-sig")

    write_dictionary()
    write_readme()

    if skipped:
        skipped_path = RADIOMETRICS_DIR / "casos_omitidos.csv"
        pd.DataFrame(skipped).to_csv(skipped_path, index=False, encoding="utf-8-sig")
    else:
        skipped_path = None

    print("\n===================================")
    print(f"Casos procesados: {len(rows)}")
    print(f"Casos omitidos: {len(skipped)}")
    print(f"CSV principal: {OUTPUT_METRICS_CSV}")
    print(f"CSV diagnóstico: {OUTPUT_DIAGNOSTIC_CSV}")
    print(f"Curvas: {CURVES_DIR}")
    print(f"Overlays: {OVERLAYS_DIR}")

    if skipped_path is not None:
        print(f"Casos omitidos guardados en: {skipped_path}")


if __name__ == "__main__":
    main()
