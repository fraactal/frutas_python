FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 v4l-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD ["python", "camera_app.py"]
