import os
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-large-patch14"

# Usa la misma variable que tu contenedor
OUTPUT_DIR = os.getenv("MODEL_PATH", "./clip_model_dir")

def main():
    print("⬇️ Descargando modelo:", MODEL_ID)
    print("📁 Directorio destino:", os.path.abspath(OUTPUT_DIR))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Descargar modelo
    model = CLIPModel.from_pretrained(MODEL_ID)
    model.save_pretrained(OUTPUT_DIR)

    # Descargar processor
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(OUTPUT_DIR)

    print("\n✅ Modelo guardado correctamente")
    print("📂 Archivos descargados:")
    for f in os.listdir(OUTPUT_DIR):
        print(" -", f)

if __name__ == "__main__":
    main()