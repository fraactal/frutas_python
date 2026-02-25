import os
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

MODEL_PATH = os.getenv("MODEL_PATH", "/app/clip_model_dir")

CATEGORIES = ["avocado", "banana", "orange", "apple", "carrot", "tomato", "unknown"]


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
