# MaIA Scoliosis Dataset — versión T1–T12 y L1–L5

## Qué contiene esta versión
Esta versión del dataset está organizada para trabajar con las vértebras:
- T1 a T12
- L1 a L5

La codificación multiclase está definida con IDs consecutivos:
- 0: background
- 1 a 12: T1 a T12
- 13 a 17: L1 a L5

## Por qué el diccionario original mostraba 35 entidades
El diccionario original del dataset incluía tres grupos de etiquetas:
1. Vértebras cervicales: C3 a C7
2. Vértebras torácicas y lumbares: T1 a T12, L1 a L5
3. Entradas adicionales presentes en el diccionario original que no corresponden al conjunto final T1–L5

Por esa razón, el conteo total del diccionario original era 35.

## Cómo usar esta versión
Para cualquier tarea de segmentación multiclase o análisis por vértebra, la referencia principal debe ser:
- `LabelMultiClass_ID_PNG/`

Las otras carpetas multiclase son de apoyo visual:
- `LabelMultiClass_Gray_JPG/`: visualización en escala de grises
- `LabelMultiClass_Color_JPG/`: visualización en color

Las radiografías originales están separadas en:
- `Normal/`
- `Scoliosis/`

Las máscaras binarias se encuentran en:
- `LabelBinaryJPG/`

## Archivos incluidos
- `diccionario_etiquetas_T1_T12_L1_L5.json`  
  Diccionario de clases de esta versión, con IDs consecutivos y equivalencia frente al diccionario original.

- `resumen_diccionario_original_35_entidades.csv`  
  Resumen de las 35 entidades del diccionario original, indicando en cuántas imágenes aparece cada una y su cantidad total de píxeles.

- `resumen_version_final_T1_T12_L1_L5.csv`  
  Resumen de las clases usadas en esta versión final, con frecuencia por imagen y total de píxeles.

- `reporte_por_mascara_version_final.csv`  
  Reporte por máscara que muestra las etiquetas presentes en la versión T1–L5 y otras etiquetas detectadas al revisar la máscara original.

- `indice_dataset.csv`  
  Índice general del dataset con las rutas relativas a radiografías y máscaras.

## Recomendación práctica
Si otra persona va a entrenar un modelo con esta base, lo más directo es usar:
1. `Normal/` y `Scoliosis/` como imágenes de entrada
2. `LabelMultiClass_ID_PNG/` como máscaras multiclase
3. `diccionario_etiquetas_T1_T12_L1_L5.json` como referencia oficial de clases
