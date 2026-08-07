# FaceForge serving image -- CPU only.
#
# Sized for a CPU App Service plan: DDIM sampling runs ~45s for 12 images on 4 threads, which is
# usable for a demo. Full 1000-step sampling takes ~15 min on CPU and is realistically GPU-only.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Gradio otherwise tries to write to a home dir that may not be writable in the container.
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the application code,
# so editing app.py does not force a ~1GB torch reinstall on every rebuild.
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Only the modules serving actually imports. Training code, notebooks, datasets and the 341MB
# training checkpoints stay out of the image (see .dockerignore).
COPY model.py dataset.py export_model.py app.py ./
COPY models/faceforge_serving.pt ./models/faceforge_serving.pt

ENV FACEFORGE_CHECKPOINT=/app/models/faceforge_serving.pt \
    PORT=7860
EXPOSE 7860

# Run as a non-root user: the default root user is a needless privilege for a web process, and
# some Azure hosts refuse or flag containers that run as root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "app.py"]
