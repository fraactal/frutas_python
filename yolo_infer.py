import os
import time
import torch
from ultralytics import YOLO
from PIL import Image

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/models/yolo/yolov8s.pt")
YOLO_DEVICE_RAW = os.getenv("YOLO_DEVICE", "cuda:0").strip().lower()
YOLO_HALF = os.getenv("YOLO_HALF", "1") == "1"
YOLO_IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "416"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))
YOLO_MAX_DET = int(os.getenv("YOLO_MAX_DET", "10"))

# Cargar modelo una vez
model = YOLO(YOLO_MODEL_PATH)

def _normalize_device(device_raw: str):
    """
    Devuelve (device_for_ultralytics, use_half)
    - device_for_ultralytics: "cpu" o int/str compatible (e.g. 0, "0", "0,1")
    - use_half: True solo si CUDA disponible y el usuario lo pidió
    """
    cuda_ok = torch.cuda.is_available()

    # Acepta formatos comunes:
    # "cpu"
    if device_raw == "cpu":
        return "cpu", False

    # "cuda" o "cuda:0" o "cuda:1"
    if device_raw.startswith("cuda"):
        if not cuda_ok:
            return "cpu", False
        # extrae índice si viene como cuda:N
        if ":" in device_raw:
            try:
                idx = int(device_raw.split(":", 1)[1])
                return idx, YOLO_HALF
            except ValueError:
                # "cuda:something" raro -> usa 0
                return 0, YOLO_HALF
        return 0, YOLO_HALF

    # "0" o "1" o "0,1"
    # Ultralytics acepta "0" o "0,1" (multi-GPU).
    # Si no hay CUDA, forzamos CPU para evitar el ValueError.
    if any(ch.isdigit() for ch in device_raw):
        if not cuda_ok:
            return "cpu", False
        return device_raw, YOLO_HALF

    # default seguro
    return ("cpu", False) if not cuda_ok else (0, YOLO_HALF)

def predict_pil(image: Image.Image):
    """
    Retorna:
      - detections: lista dict {cls, name, conf, xyxy}
      - infer_ms: latencia inferencia aproximada
    """
    t0 = time.time()

    device, use_half = _normalize_device(YOLO_DEVICE_RAW)

    results = model.predict(
        source=image,
        device=device,
        imgsz=YOLO_IMG_SIZE,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        max_det=YOLO_MAX_DET,
        half=use_half,
        verbose=False,
    )

    infer_ms = (time.time() - t0) * 1000.0

    r = results[0]
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        names = r.names  # id->name
        for b in r.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = [float(x) for x in b.xyxy[0].tolist()]
            dets.append({
                "cls": cls_id,
                "name": names.get(cls_id, str(cls_id)),
                "conf": conf,
                "xyxy": xyxy
            })

    return dets, infer_ms