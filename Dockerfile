# Base ya cargada en tu servidor (docker load -i python-3.11.8.tar)
FROM python:3.11.8

# Modo offline + ajustes de runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INDEX=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/models \
    MODEL_PATH=/app/clip_model_dir

WORKDIR /app

# 1) Copiamos Wheels y requirements primero para aprovechar cache de capas
COPY wheels/ /wheels/
COPY requirements.txt .

# 2) Instalar SOLO desde /wheels (sin red)
#    Si quieres actualizar pip offline, coloca también su wheel en /wheels e incluye 'pip' en requirements.txt
RUN pip install --no-index --find-links=/wheels -r requirements.txt

# 3) Copiar tu código y plantillas
COPY . .

# 4) (Opcional pero recomendado) Copiar el modelo CLIP ya descargado
#    Asegúrate de tener ./clip_model_dir en tu contexto de build
COPY clip_model_dir/ /app/clip_model_dir/

EXPOSE 8000

# Ejecuta FastAPI con Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
