Sí, se puede y es más simple de lo que parece: en vez de recibir una imagen por FastAPI, tu proceso Python captura frames desde la cámara (webcam/USB) y en cada frame ejecuta predict().

La idea es separar tu proyecto en 2 modos:

Modo Web (FastAPI): como lo tienes ahora.

Modo Cámara (local): un script camera_app.py que abre la webcam, toma frames y muestra el resultado en pantalla.

Abajo te dejo el “mapa mental” y un ejemplo completo listo para correr.

1) Qué cambia conceptualmente
Antes (servidor web)

Usuario sube imagen → upload_image()

predict(contents) → etiqueta

Render HTML

Ahora (cámara local)

Tu script abre webcam (cv2.VideoCapture(0))

Loop:

lee frame

convierte frame a bytes (o PIL Image)

predict(...)

dibuja etiqueta en pantalla

Tecla para salir (q)

2) Dependencias nuevas

Necesitas OpenCV para capturar cámara y mostrar ventana:

requirements (local camera):

opencv-python==4.10.0.84


Nota: tu torch==2.4.1+cpu sirve si correrás en CPU. Si quieres GPU, cambia a torch CUDA (lo vemos después).

3) Recomendación: extrae “modelo + predict” a un archivo reutilizable

Crea clip_infer.py:

import os
import torch
from PIL import Image
from io import BytesIO
from transformers import CLIPModel, CLIPProcessor

MODEL_PATH = os.getenv("MODEL_PATH", "/app/clip_model_dir")

CATEGORIES = [
    "avocado", "banana", "orange", "apple", "carrot", "tomato",
    "strawberry", "blueberry", "watermelon", "tangerine"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

def predict_pil(image: Image.Image, categories=CATEGORIES):
    inputs = processor(
        text=categories,
        images=image.convert("RGB"),
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].detach().float().cpu()

    return {categories[i]: float(probs[i]) for i in range(len(categories))}


Con esto puedes usar el mismo modelo tanto desde FastAPI como desde la cámara.

4) Script cámara local (sin FastAPI)

Crea camera_app.py:

import cv2
from PIL import Image
import numpy as np
import time

from clip_infer import predict_pil, CATEGORIES

def main():
    cap = cv2.VideoCapture(0)  # si no es 0, prueba 1 o 2
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Prueba cambiando el índice (0/1/2).")

    # Para no saturar CPU/GPU, inferimos cada X segundos
    infer_every_sec = 0.5
    last_infer = 0.0
    best_label = "..."
    best_prob = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        if now - last_infer >= infer_every_sec:
            last_infer = now

            # OpenCV entrega BGR, PIL espera RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            results = predict_pil(pil_img, CATEGORIES)
            best_label = max(results, key=results.get)
            best_prob = results[best_label]

        # Overlay del texto
        text = f"{best_label} ({best_prob:.2f})"
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Fruit/Veg Recognizer - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

Cómo correrlo
pip install -r requirements.txt
pip install opencv-python==4.10.0.84
python camera_app.py

5) ¿Y si quieres mantener FastAPI y cámara?

Puedes tener ambos:

app.py para web

camera_app.py para local

Y compartir el mismo clip_infer.py.

6) Consejos para que funcione bien (y no “se pegue”)

Inferencia cada 0.3–1.0s (como puse). Si haces inferencia en cada frame, en CPU se vuelve lento.

Si la cámara abre pero la ventana no aparece:

En Linux a veces falta libgl1 / libglib2.0-0 (si estás en Docker/WSL).

Si estás en Docker: acceder a cámara requiere pasar el dispositivo (/dev/video0) y permisos. (Te lo armo si me dices tu entorno.)

7) Importante: CLIP “clasifica por texto”, no está entrenado solo para frutas

Funciona, pero si quieres más precisión real, conviene:

usar EfficientNet/MobileNet entrenado con tus clases
o

entrenar/fine-tunear un clasificador propio

Aun así, para un demo local CLIP sirve bien.