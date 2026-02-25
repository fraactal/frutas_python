import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

MOBILENET_WEIGHTS = os.getenv("MOBILENET_WEIGHTS", "/models/mobilenet/mobilenet_v3_small_imagenet.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Categorías ImageNet vienen dentro de torchvision (no se descargan)
IMAGENET_CATEGORIES = MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta["categories"]

model = mobilenet_v3_small(weights=None).to(DEVICE).eval()
state = torch.load(MOBILENET_WEIGHTS, map_location=DEVICE)
model.load_state_dict(state)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

def predict_pil(image: Image.Image, top_k: int = 5):
    x = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=0)

    topk = torch.topk(probs, k=top_k)
    ids = topk.indices.detach().cpu().tolist()
    ps = topk.values.detach().cpu().tolist()

    # Devuelve lista de (nombre, prob, id)
    out = []
    for i, p in zip(ids, ps):
        name = IMAGENET_CATEGORIES[i] if i < len(IMAGENET_CATEGORIES) else f"class_{i}"
        out.append((name, float(p), int(i)))
    return out
