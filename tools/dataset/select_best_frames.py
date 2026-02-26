import cv2
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import imagehash
import shutil

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def exposure_score(gray):
    m = float(np.mean(gray))
    return float(np.exp(-((m - 128.0) ** 2) / (2 * (35.0 ** 2))))

def phash_from_bgr(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return imagehash.phash(pil)

def main():
    ap = argparse.ArgumentParser(description="Selecciona las mejores imágenes (nitidez/exposición + dedup por pHash).")
    ap.add_argument("--dataset_dir", required=True, help="Carpeta raíz dataset (ej: dataset/)")
    ap.add_argument("--class_name", required=True, help="Clase (prefijo en nombre archivo)")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--out_suffix", default="_selected", help="Sufijo de carpeta salida (default: _selected)")
    ap.add_argument("--topk", type=int, default=600)
    ap.add_argument("--phash_dist", type=int, default=6)
    ap.add_argument("--min_sharp", type=float, default=70.0)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    in_dir = dataset_dir / "images" / args.split
    out_dir = dataset_dir / ("images" + args.out_suffix) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(in_dir.glob(f"{args.class_name}_*.jpg"))
    if not imgs:
        raise RuntimeError(f"No encontré imágenes con prefijo {args.class_name}_ en {in_dir}")

    candidates = []
    for p in imgs:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sharp = variance_of_laplacian(gray)
        if sharp < args.min_sharp:
            continue

        exp = exposure_score(gray)
        score = (sharp / (sharp + 300.0)) * 0.75 + exp * 0.25
        h = phash_from_bgr(frame)

        candidates.append((score, p, h, sharp, float(np.mean(gray))))

    candidates.sort(key=lambda x: x[0], reverse=True)

    kept = []
    kept_hashes = []
    for item in candidates:
        score, p, h, sharp, mean_gray = item

        dup = False
        for kh in kept_hashes:
            if (h - kh) <= args.phash_dist:
                dup = True
                break
        if dup:
            continue

        kept.append(item)
        kept_hashes.append(h)

        if len(kept) >= args.topk:
            break

    # Limpia salida y copia
    # (si prefieres no borrar, comenta estas 2 líneas)
    for old in out_dir.glob(f"{args.class_name}_*.jpg"):
        old.unlink()

    for score, p, h, sharp, mean_gray in kept:
        shutil.copy2(p, out_dir / p.name)

    print(f"[select] in={len(imgs)} cand={len(candidates)} kept={len(kept)} out={out_dir}")

if __name__ == "__main__":
    main()