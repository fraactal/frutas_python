import os
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Guardar dentro del proyecto
out_dir = os.getenv("OUT_DIR", "models/mobilenet")
os.makedirs(out_dir, exist_ok=True)

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights)
model.eval()

out_path = os.path.join(out_dir, "mobilenet_v3_small_imagenet.pth")
torch.save(model.state_dict(), out_path)
print("✅ Saved:", os.path.abspath(out_path))