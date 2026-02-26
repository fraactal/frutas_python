import os
from ultralytics import YOLO

OUT_DIR = os.getenv("OUT_DIR", "models/yolo")
MODEL_NAME = os.getenv("MODEL_NAME", "yolov8s.pt")

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, MODEL_NAME)

# Si no existe, ultralytics lo descarga y lo cachea.
# Luego lo guardamos al path de tu proyecto.
if not os.path.exists(out_path):
    print(f"⬇️ Descargando {MODEL_NAME} ...")
    model = YOLO(MODEL_NAME)
    # model.ckpt_path suele apuntar al cache; guardamos una copia al volumen del proyecto
    # Si no, igual funciona usando el cache, pero preferimos tenerlo en ./models/yolo
    try:
        src = model.ckpt_path
        if src and os.path.exists(src):
            import shutil
            shutil.copy2(src, out_path)
            print("✅ Guardado en:", os.path.abspath(out_path))
        else:
            print("⚠️ No se encontró ckpt_path, pero el modelo quedó cacheado por ultralytics.")
    except Exception as e:
        print("⚠️ No se pudo copiar desde cache:", e)
else:
    print("✅ Ya existe:", os.path.abspath(out_path))