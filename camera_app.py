import os
import cv2
import time
import numpy as np
from PIL import Image
from clip_infer import predict_pil, CATEGORIES

# -------- Configuración por variables de entorno --------
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
RESIZE_W = int(os.getenv("RESIZE_W", "224"))
RESIZE_H = int(os.getenv("RESIZE_H", "224"))

# Motion detection
MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "25"))      # 15-35 típico
MOTION_MIN_AREA = int(os.getenv("MOTION_MIN_AREA", "1200"))      # 500-5000 según cámara
STABLE_MS = int(os.getenv("STABLE_MS", "800"))                   # 600-1200ms típico
COOLDOWN_SEC = float(os.getenv("COOLDOWN_SEC", "1.0"))           # (ya no se usa en modo "armed")

# Runtime
PYTHONUNBUFFERED = os.getenv("PYTHONUNBUFFERED", "1") == "1"

def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara (CAM_INDEX={CAM_INDEX}). Prueba 1.")

    # Forzar MJPG + resolución (más estable en USB cams)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_H)
    return cap

def preprocess_for_motion(frame_bgr):
    # Para motion detection usamos gris y blur (reduce ruido)
    small = cv2.resize(frame_bgr, (RESIZE_W, RESIZE_H))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    return gray, small

def detect_motion(prev_gray, curr_gray):
    # Diferencia absoluta entre frames
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Área de movimiento
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_area = 0
    for c in contours:
        motion_area += cv2.contourArea(c)

    is_motion = motion_area >= MOTION_MIN_AREA
    return is_motion, int(motion_area)

def predict_from_frame(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    t0 = time.time()
    results = predict_pil(pil_img, CATEGORIES)
    dt_ms = (time.time() - t0) * 1000

    best_label = max(results, key=results.get)
    best_prob = results[best_label]

    top3 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_str = " | ".join([f"{k}:{v:.3f}" for k, v in top3])

    return best_label, best_prob, dt_ms, top3_str

def main():
    cap = open_camera()

    # Warmup (quita lag inicial)
    print("🔥 Warmup modelo...")
    dummy = Image.new("RGB", (RESIZE_W, RESIZE_H), color=(0, 0, 0))
    _ = predict_pil(dummy, CATEGORIES)
    print("✅ Warmup listo")

    print("✅ Motion-trigger ON (modo 1 disparo por evento)")
    print(f"➡️ CAM_INDEX={CAM_INDEX} | RESIZE={RESIZE_W}x{RESIZE_H}")
    print(f"➡️ MOTION_THRESHOLD={MOTION_THRESHOLD} | MOTION_MIN_AREA={MOTION_MIN_AREA}")
    print(f"➡️ STABLE_MS={STABLE_MS} | COOLDOWN_SEC={COOLDOWN_SEC} (no usado en modo armed)")
    print("🛑 Ctrl+C para salir\n")

    # Inicializar baseline
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("No se pudo leer el primer frame de la cámara.")

    prev_gray, _ = preprocess_for_motion(frame)

    last_motion_ts = time.time()  # última vez que hubo movimiento
    armed = True                  # True = listo para disparar cuando se estabilice; False = ya disparó

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            curr_gray, curr_small = preprocess_for_motion(frame)
            is_motion, motion_area = detect_motion(prev_gray, curr_gray)

            now = time.time()

            if is_motion:
                last_motion_ts = now
                armed = True  # rearmar: hubo movimiento, podrá disparar de nuevo cuando se estabilice
                # print(f"💨 movimiento area={motion_area}")  # debug opcional
            else:
                stable_ms = (now - last_motion_ts) * 1000

                # Disparo ÚNICO por evento: solo si está armado y la escena ya está estable
                if armed and stable_ms >= STABLE_MS:
                    armed = False  # se desarma hasta que haya nuevo movimiento

                    label, prob, dt_ms, top3_str = predict_from_frame(curr_small)
                    print(f"🎯 Trigger (estable {int(stable_ms)}ms) => {label} ({prob:.3f}) | infer={dt_ms:.0f}ms")
                    print(f"   Top3: {top3_str}\n")

            prev_gray = curr_gray

            # Pequeño sleep para bajar CPU del loop de captura
            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("✅ Cámara liberada")

if __name__ == "__main__":
    main()
