from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from io import BytesIO
from transformers import CLIPModel, CLIPProcessor
import torch
import base64
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/clip_model_dir")
model = CLIPModel.from_pretrained(MODEL_PATH, local_files_only=True)
processor = CLIPProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
#model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
#processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

CATEGORIES = [
    "avocado", "banana", "orange", "apple", "carrot", "tomato",
    "strawberry", "blueberry", "watermelon", "tangerine"
]

def predict(image_bytes, categories):
    """Hace la predicción usando CLIP"""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = processor(
        text=categories,
        images=image,
        return_tensors="pt",
        padding=True
    )
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return {categories[i]: float(probs[0][i]) for i in range(len(categories))}

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    # Página inicial sin resultados
    return templates.TemplateResponse(
        "form.html",
        {"request": request, "result": None, "preview_url": None}
    )

@app.post("/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()

    # Predicción
    results = predict(contents, CATEGORIES)
    best_label = max(results, key=results.get)

    # Construir data URL para mostrar imagen en HTML (sin guardar en disco)
    mime = file.content_type or "image/jpeg"
    b64 = base64.b64encode(contents).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "result": (best_label, results),
            "preview_url": data_url,
        }
    )