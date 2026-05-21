import json
import re

NOTEBOOK_PATH = "notebooks/hybrid_yolov8_medsam_v1.ipynb"

# Regex para tokens HuggingFace
TOKEN_PATTERN = r"hf_[A-Za-z0-9]{20,}"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Función recursiva para limpiar tokens
def clean_obj(obj):

    if isinstance(obj, str):
        return re.sub(TOKEN_PATTERN, "[HF_TOKEN_REMOVED]", obj)

    elif isinstance(obj, list):
        return [clean_obj(x) for x in obj]

    elif isinstance(obj, dict):
        return {k: clean_obj(v) for k, v in obj.items()}

    return obj

cleaned = clean_obj(data)

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=1)

print("✅ Notebook limpiado")