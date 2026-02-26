import cv2
import argparse
from pathlib import Path

def extract_from_video(video_path: Path, out_dir: Path, prefix: str,
                       every: int, width: int, height: int,
                       start: int, end: int | None, jpg_quality: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir video: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx >= start and (end is None or idx <= end):
            if idx % every == 0:
                if width and height:
                    frame = cv2.resize(frame, (width, height))
                name = f"{prefix}_{video_path.stem}_{idx:06d}.jpg"
                cv2.imwrite(str(out_dir / name), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
                saved += 1

        idx += 1

    cap.release()
    return saved

def main():
    ap = argparse.ArgumentParser(description="Extrae frames desde todos los videos de una clase.")
    ap.add_argument("--videos_dir", required=True, help="Carpeta de videos por clase: videos/<class_name>")
    ap.add_argument("--dataset_dir", required=True, help="Carpeta raíz del dataset (ej: dataset/)")
    ap.add_argument("--class_name", required=True, help="Nombre clase (ej: tomate)")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--every", type=int, default=10, help="1 frame cada N frames")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--jpg_quality", type=int, default=92)
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        raise RuntimeError(f"No existe videos_dir: {videos_dir}")

    dataset_dir = Path(args.dataset_dir)
    out_dir = dataset_dir / "images" / args.split

    videos = sorted([p for p in videos_dir.iterdir() if p.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi"]])
    if not videos:
        raise RuntimeError(f"No encontré videos en: {videos_dir}")

    total = 0
    prefix = args.class_name
    print(f"[extract] clase={args.class_name} split={args.split} videos={len(videos)} out={out_dir}")

    for v in videos:
        n = extract_from_video(
            v, out_dir, prefix,
            every=args.every, width=args.width, height=args.height,
            start=args.start, end=args.end, jpg_quality=args.jpg_quality
        )
        print(f"  - {v.name}: {n} frames")
        total += n

    print(f"[extract] total frames guardados: {total}")

if __name__ == "__main__":
    main()