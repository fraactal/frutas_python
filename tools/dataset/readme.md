# Dataset pipeline (videos -> frames -> selección) para YOLO

Este repositorio define un flujo por etapas para construir un dataset a partir de videos, manteniendo una estructura compatible con YOLO (Ultralytics).

## Estructura esperada

```text
Frutas_GPU_camara_Yolo/
├── tools/
│   └── dataset/
│       ├── extract_frames.py
│       ├── select_best_frames.py
│       ├── requirements.txt
│       └── README.md
├── videos/
│   ├── tomate/
│   │   ├── tomate_01.mp4
│   │   └── tomate_02.mp4
│   ├── tomate_cherry/
│   │   └── cherry_01.mp4
│   └── cebolla/
│       └── cebolla_01.mp4
└── dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
```

- **videos/<clase>/**: entrada (inputs). Pones aquí todos los videos de una clase.
- **dataset/images/**: salida de imágenes (frames extraídos).
- **dataset/labels/**: etiquetas YOLO (se llenan después con etiquetado manual o auto-label).
- **dataset/data.yaml**: configuración de YOLO (paths + clases).

## 1) Crear entorno virtual (dataset_extractor)

Desde la raíz del proyecto:

```bash
python3 -m venv dataset_extractor
source dataset_extractor/bin/activate
pip install --upgrade pip
pip install -r tools/dataset/requirements.txt
```

## 2) Extraer frames de TODOS los videos de una clase

Ejemplo: extraer dataset “full” para `tomate` desde `videos/tomate/` a `dataset/images/train/`:

```bash
python3 tools/dataset/extract_frames.py   --videos_dir videos/tomate   --dataset_dir dataset   --class_name tomate   --split train   --every 8   --width 640 --height 640
```

Parámetros recomendados:
- `--every 6` a `--every 10` (en video 30 fps) para evitar duplicados.
- `--width/--height 640` para entrenar rápido y consistente.

## 3) Seleccionar las mejores imágenes (nitidez + exposición + deduplicación)

Ejemplo: curar `tomate` en `train` y dejar el resultado en `dataset/images_selected/train/`:

```bash
python3 tools/dataset/select_best_frames.py   --dataset_dir dataset   --class_name tomate   --split train   --topk 800   --phash_dist 6   --min_sharp 70
```

Notas:
- `--topk` define cuántas imágenes finales quieres por clase.
- `--phash_dist` controla cuán agresiva es la deduplicación (menor = más estricto).
- `--min_sharp` filtra frames borrosos.

## 4) data.yaml (YOLO)

Ejemplo mínimo:

```yaml
path: dataset
train: images/train
val: images/val

names:
  0: tomate
  1: tomate_cherry
  2: cebolla
```

Si entrenas usando imágenes curadas, cambia:
- `train: images_selected/train`
- `val: images_selected/val`

## 5) Importante: para entrenar necesitas labels

YOLO necesita un `.txt` por cada imagen con el mismo nombre base:

```text
dataset/images/train/img_000123.jpg
dataset/labels/train/img_000123.txt
```

Cada línea del `.txt` (formato YOLO):
```
<class_id> <x_center> <y_center> <width> <height>
```
Todos los valores normalizados 0..1.

## 6) Flujo por etapas (recomendado)

1. Crea `videos/<clase>/` y copia videos.
2. Extrae frames a `dataset/images/train`.
3. (Opcional) Selecciona mejores a `dataset/images_selected/train`.
4. Repite por clase.
5. Etiqueta (manual/auto-label) para llenar `dataset/labels/...`.
6. Entrena YOLO y genera `best.pt`.

---

# Entrenamiento en Google Colab (primer flujo con 1 fruta)

## ¿Hay que instalar Colab?
No. **Google Colab se usa online** desde tu navegador. Solo necesitas una cuenta Google.

## Pasos (resumen)
1. Abre Colab y crea un Notebook.
2. Runtime -> Change runtime type -> Hardware accelerator: **GPU**.
3. Sube tu dataset (zip) o usa Google Drive (mount).
4. Instala Ultralytics y entrena.

## Comandos típicos en Colab
```bash
!pip install -U ultralytics
!yolo checks
```

Entrenar (ejemplo con 1 clase: tomate):
```bash
!yolo detect train model=yolov8n.pt data=/content/dataset/data.yaml imgsz=640 epochs=30 batch=16 device=0
```

El modelo final queda en:
```text
runs/detect/train/weights/best.pt
```

Luego descargas `best.pt` y lo copias a tu proyecto:
```text
./models/yolo/best.pt
```

y apuntas tu contenedor a:
- `YOLO_MODEL_PATH=/models/yolo/best.pt`