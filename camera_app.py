import os
import cv2
import time
import json
from PIL import Image
from yolo_infer import predict_pil

# -------- Configuración por variables de entorno --------
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
RESIZE_W = int(os.getenv("RESIZE_W", "640"))
RESIZE_H = int(os.getenv("RESIZE_H", "480"))

# Motion detection
MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "25"))
MOTION_MIN_AREA = int(os.getenv("MOTION_MIN_AREA", "1200"))
STABLE_MS = int(os.getenv("STABLE_MS", "800"))

# Multi-frame voting
FRAMES_BASE = int(os.getenv("FRAMES_BASE", "3"))
FRAMES_MAX = int(os.getenv("FRAMES_MAX", "5"))
MARGIN_THRESHOLD = float(os.getenv("MARGIN_THRESHOLD", "0.08"))
FRAME_GAP_SEC = float(os.getenv("FRAME_GAP_SEC", "0.03"))

LOG_FORMAT = os.getenv("LOG_FORMAT", "json")

def log_event(payload: dict):
    if LOG_FORMAT == "json":
        print(json.dumps(payload, ensure_ascii=False))
    else:
        # fallback simple
        print(payload)

def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara (CAM_INDEX={CAM_INDEX}). Prueba 1.")

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

def score_by_class(detections):
    """
    Agrega confidencias por clase (sum).
    Retorna dict: class_name -> score
    """
    scores = {}
    for d in detections:
        name = d["name"]
        scores[name] = scores.get(name, 0.0) + float(d["conf"])
    return scores

def avg_scores(dicts):
    out = {}
    n = max(1, len(dicts))
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    for k in out:
        out[k] /= n
    return out

def top2(scores):
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top1 = items[0] if len(items) > 0 else ("unknown", 0.0)
    top2_ = items[1] if len(items) > 1 else ("unknown", 0.0)
    return top1, top2_, items

def infer_multiframe(cap, first_frame_bgr, n_frames):
    """
    Ejecuta YOLO sobre n_frames y promedia scores por clase.
    Retorna: (scores_avg, dets_last, infer_total_ms, frames_ok)
    """
    score_dicts = []
    dets_last = []
    t0 = time.time()
    frames_ok = 0

    # primer frame
    dets, ms = predict_pil(to_pil(first_frame_bgr))
    dets_last = dets
    score_dicts.append(score_by_class(dets))
    frames_ok += 1

    # frames extra
    for _ in range(n_frames - 1):
        time.sleep(FRAME_GAP_SEC)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        fr_small = cv2.resize(fr, (RESIZE_W, RESIZE_H))
        dets, ms = predict_pil(to_pil(fr_small))
        dets_last = dets
        score_dicts.append(score_by_class(dets))
        frames_ok += 1

    scores_avg = avg_scores(score_dicts)
    total_ms = (time.time() - t0) * 1000.0
    return scores_avg, dets_last, total_ms, frames_ok

def main():
    cap = open_camera()

    log_event({
        "event": "startup",
        "cam_index": CAM_INDEX,
        "resize": f"{RESIZE_W}x{RESIZE_H}",
        "motion": {"threshold": MOTION_THRESHOLD, "min_area": MOTION_MIN_AREA, "stable_ms": STABLE_MS},
        "multiframe": {"base": FRAMES_BASE, "max": FRAMES_MAX, "margin_threshold": MARGIN_THRESHOLD, "gap_sec": FRAME_GAP_SEC},
    })

    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("No se pudo leer el primer frame.")

    prev_gray, _ = preprocess_for_motion(frame)

    last_motion_ts = time.time()
    armed = True

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
                stable_ms = (now - last_motion_ts) * 1000.0
                if armed and stable_ms >= STABLE_MS:
                    armed = False

                    # 1) base
                    scores_avg, dets_last, total_ms, frames_ok = infer_multiframe(cap, curr_small, FRAMES_BASE)
                    (l1, s1), (l2, s2), items = top2(scores_avg)
                    margin = s1 - s2
                    used_frames = FRAMES_BASE

                    # 2) si duda → refuerza
                    if FRAMES_MAX > FRAMES_BASE and margin < MARGIN_THRESHOLD:
                        scores_avg, dets_last, total_ms, frames_ok = infer_multiframe(cap, curr_small, FRAMES_MAX)
                        (l1, s1), (l2, s2), items = top2(scores_avg)
                        margin = s1 - s2
                        used_frames = FRAMES_MAX

                    # top3 scores promedio
                    top3 = items[:3]
                    top3_str = " | ".join([f"{k}:{v:.3f}" for k, v in top3])

                    log_event({
                        "event": "trigger_infer",
                        "stable_ms": int(stable_ms),
                        "motion_area": motion_area,
                        "frames_used": used_frames,
                        "frames_ok": frames_ok,
                        "infer_total_ms": int(total_ms),
                        "top1": {"label": l1, "score": float(s1)},
                        "top2": {"label": l2, "score": float(s2)},
                        "margin": float(margin),
                        "avg_top3": top3_str,
                        "detections_last": dets_last[:10],
                    })

            prev_gray = curr_gray
            time.sleep(0.03)

    except KeyboardInterrupt:
        log_event({"event": "shutdown"})
    finally:
        cap.release()
        log_event({"event": "camera_released"})

if __name__ == "__main__":
    main()