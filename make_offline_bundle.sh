#!/usr/bin/env bash
set -e

IMAGE_NAME="frutas-python:1"

echo "[1/3] Descargando wheels en ./wheels/"
mkdir -p wheels
docker run --rm -v "$PWD:/io" python:3.11.8 bash -lc "
  python -m pip install --upgrade pip wheel &&
  pip download --only-binary=:all: -r /io/requirements.txt -d /io/wheels
"

echo "[2/3] Construyendo imagen local $IMAGE_NAME"
docker buildx build --platform linux/amd64 -t $IMAGE_NAME --load .

echo "[3/3] Guardando imagen a tar"
docker save $IMAGE_NAME -o ${IMAGE_NAME//:/-}.tar

echo "✅ Bundle listo:"
echo "   - wheels/ (dependencias offline)"
echo "   - ${IMAGE_NAME//:/-}.tar (imagen lista para docker load)"
