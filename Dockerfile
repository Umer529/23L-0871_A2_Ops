# VisionGuard web app (Task 6.2)
# Slim Debian base: has glibc (TensorFlow wheels need it, Alpine's musl does not) but
# none of the compilers/docs of the full python image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_PATH=/app/models/champion.keras \
    PORT=5000

WORKDIR /app

# 1) dependencies first: this slow layer (~TensorFlow) is reused from cache
#    as long as requirements.txt does not change, even when the code does
COPY app/requirements.txt .
RUN pip install -r requirements.txt

# 2) a non-root user, so a compromised app cannot modify the image or the host volume
RUN useradd --create-home --uid 10001 appuser

# 3) source code last: editing a template only rebuilds these small layers
COPY --chown=appuser:appuser src/preprocess.py ./
COPY --chown=appuser:appuser app/ ./

USER appuser
EXPOSE 5000

# the model is NOT baked in: docker-compose mounts ./models read-only at /app/models
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)" || exit 1

CMD ["python", "main.py"]
