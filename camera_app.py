import os
import cv2
import time
import numpy as np
from PIL import Image
from mobilenet_infer import predict_pil

# -------- Configuración por variables de entorno --------
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
RESIZE_W = int(os.getenv("RESIZE_W", "224"))
RESIZE_H = int(os.getenv("RESIZE_H", "224"))

# Motion detection
MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "25"))   # 15-35 típico
MOTION_MIN_AREA = int(os.getenv("MOTION_MIN_AREA", "1200"))   # 500-5000 típico
STABLE_MS = int(os.getenv("STABLE_MS", "800"))                # 600-1200ms típico

# -------- Multi-frame voting (mejor performance/robustez) --------
FRAMES_BASE = int(os.getenv("FRAMES_BASE", "3"))              # frames por defecto
FRAMES_MAX = int(os.getenv("FRAMES_MAX", "5"))                # frames máximo si hay duda
MARGIN_THRESHOLD = float(os.getenv("MARGIN_THRESHOLD", "0.08"))  # si top1-top2 < esto, es dudoso
FRAME_GAP_SEC = float(os.getenv("FRAME_GAP_SEC", "0.03"))     # gap entre frames (30ms)

def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara (CAM_INDEX={CAM_INDEX}). Prueba 1.")

    # Forzar MJPG + resolución (mejor para webcams USB)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_H)
    return cap

def preprocess_for_motion(frame_bgr):
    small = cv2.resize(frame_bgr, (RESIZE_W, RESIZE_H))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    return gray, small

def detect_motion(prev_gray, curr_gray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_area = 0
    for c in contours:
        motion_area += cv2.contourArea(c)

    return motion_area >= MOTION_MIN_AREA, int(motion_area)

def to_pil(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def probs_from_topk(topk):
    """
    topk: [(name, prob, id), ...]
    devuelve dict name->prob
    """
    return {name: float(prob) for name, prob, _ in topk}

def avg_probs(dicts):
    """
    Promedio simple de probabilidades (solo sobre claves vistas).
    Para modelos entrenados en tus clases (futuro), todas las claves estarán.
    """
    out = {}
    n = len(dicts)
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    for k in out:
        out[k] /= n
    return out

def top2_from_probs(pdict):
    items = sorted(pdict.items(), key=lambda x: x[1], reverse=True)
    top1 = items[0] if len(items) > 0 else ("unknown", 0.0)
    top2 = items[1] if len(items) > 1 else ("unknown", 0.0)
    return top1, top2, items

def infer_multiframe(cap, first_frame_bgr, n_frames):
    """
    Usa first_frame_bgr como primer frame (ya lo tienes) y captura n_frames-1 adicionales.
    Retorna: probs_promedio, dt_ms_total
    """
    prob_dicts = []
    t0 = time.time()

    # 1) primer frame
    topk = predict_pil(to_pil(first_frame_bgr), top_k=5)
    prob_dicts.append(probs_from_topk(topk))

    # 2) frames extra
    for _ in range(n_frames - 1):
        # pequeño gap para que el siguiente frame sea realmente distinto
        time.sleep(FRAME_GAP_SEC)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        fr_small = cv2.resize(fr, (RESIZE_W, RESIZE_H))
        topk = predict_pil(to_pil(fr_small), top_k=5)
        prob_dicts.append(probs_from_topk(topk))

    # si por lectura falló y quedamos con menos dicts, promediamos igual
    probs_avg = avg_probs(prob_dicts)
    dt_ms = (time.time() - t0) * 1000
    return probs_avg, dt_ms

def main():
    cap = open_camera()

    print("✅ MobileNetV3 Camera POC (motion trigger + multiframe avg)")
    print(f"➡️ CAM_INDEX={CAM_INDEX} | RESIZE={RESIZE_W}x{RESIZE_H}")
    print(f"➡️ MOTION_THRESHOLD={MOTION_THRESHOLD} | MOTION_MIN_AREA={MOTION_MIN_AREA} | STABLE_MS={STABLE_MS}")
    print(f"➡️ FRAMES_BASE={FRAMES_BASE} | FRAMES_MAX={FRAMES_MAX} | MARGIN_THRESHOLD={MARGIN_THRESHOLD} | GAP={FRAME_GAP_SEC}s")
    print("🛑 Ctrl+C para salir\n")

    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("No se pudo leer el primer frame.")

    prev_gray, _ = preprocess_for_motion(frame)

    last_motion_ts = time.time()
    armed = True  # dispara solo 1 vez hasta nuevo movimiento

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
                armed = True
            else:
                stable_ms = (now - last_motion_ts) * 1000
                if armed and stable_ms >= STABLE_MS:
                    armed = False

                    # --- 1) Inferencia base con 3 frames ---
                    probs_avg, dt_ms = infer_multiframe(cap, curr_small, n_frames=FRAMES_BASE)
                    (l1, p1), (l2, p2), items = top2_from_probs(probs_avg)
                    margin = p1 - p2

                    used_frames = FRAMES_BASE

                    # --- 2) Si hay duda, reforzar hasta 5 frames ---
                    if FRAMES_MAX > FRAMES_BASE and margin < MARGIN_THRESHOLD:
                        probs_avg2, dt_ms2 = infer_multiframe(cap, curr_small, n_frames=FRAMES_MAX)
                        (l1, p1), (l2, p2), items = top2_from_probs(probs_avg2)
                        margin = p1 - p2
                        dt_ms = dt_ms2
                        used_frames = FRAMES_MAX

                    # Imprimir top3 promedio
                    top3 = items[:3]
                    top3_str = " | ".join([f"{k}:{v:.3f}" for k, v in top3])

                    print(f"🎯 Trigger (estable {int(stable_ms)}ms) | frames={used_frames} | infer_total={dt_ms:.0f}ms | margin={margin:.3f}")
                    print(f"   Pred: {l1} ({p1:.3f}) vs {l2} ({p2:.3f})")
                    print(f"   AvgTop3: {top3_str}\n")

            prev_gray = curr_gray
            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("✅ Cámara liberada")

if __name__ == "__main__":
    main()
