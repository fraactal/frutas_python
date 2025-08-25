# Fruit & Vegetable Classifier with CLIP and FastAPI

Proyecto para clasificar imágenes de frutas y verduras usando el modelo CLIP de Hugging Face, servido con FastAPI y desplegado con Docker.

---

## Características

- Modelo CLIP preentrenado para clasificación zero-shot
- Web simple para subir imágenes y ver predicciones
- Lista configurable de frutas y verduras para prueba
- Contenerizado con Docker y Docker Compose

---

## Instalación y uso local

1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/fruit_clip_fastapi.git
cd fruit_clip_fastapi
```

2. Instalar dependencias

```bash
pip install -r requirements.txt
```

3. Ejecutar la app

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

4. Abrir en navegador

```
http://localhost:8000
```

---

## Uso con Docker

1. Construir la imagen

```bash
docker build -t fruit_clip_fastapi .
```

2. Ejecutar el contenedor

```bash
docker run -p 8000:8000 fruit_clip_fastapi
```

3. Abrir en navegador

```
http://localhost:8000
```

---

## Uso con Docker Compose

1. Levantar servicio con

```bash
docker-compose up --build
```

2. Abrir en navegador

```
http://localhost:8000
```

---

## Configuración

- La lista de frutas y verduras que el modelo puede predecir está en el archivo `app.py` en la variable `CATEGORIES`. Puedes editarla para agregar o quitar elementos.

---

## Autor

[@fractal](https://github.com/fractal)

---

## Licencia

Este proyecto está bajo licencia MIT. Consulta el archivo LICENSE para más detalles.
