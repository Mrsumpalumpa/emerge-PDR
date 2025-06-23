# GILGAMESH_CORE/Dockerfile

# Usa una imagen base de Python oficial, preferiblemente con Alpine para un tamaño reducido
FROM python:3.10-slim-bookworm

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app





# Copia los archivos de requerimientos e instala las dependencias
COPY requirements.txt .
ARG CACHEBUST=1
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de tu aplicación (el contenido de GILGAMESH_CORE/app/ se copia a /app/ en el contenedor)
COPY app/ .

# Expone el puerto que usará Gunicorn
EXPOSE 8000

# --- TEMPORARY DEBUGGING STEPS ---
# Run pip show to find the installation location of fastapi-mcp
RUN pip show fastapi-mcp
# List the contents of the installed package to see its internal structure
RUN ls -R $(python -c "import site; print(site.getsitepackages()[0])")/fastapi_mcp/
# --- END TEMPORARY DEBUGGING STEPS ---

# Comando para ejecutar la aplicación con Gunicorn y Uvicorn workers
# 'main:app' funciona porque main.py está directamente en /app/
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "main:app"]
