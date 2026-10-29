FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Inference is CPU-only, so use the small CPU wheel instead of the ROCm one pinned in pyproject.toml
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY docker/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY ML_layer ./ML_layer
COPY data_pipeline ./data_pipeline
# artifacts/ is mounted at runtime (see docker-compose.yml). The checkpoint must carry its serving bundle
# (python -m ML_layer.prepare_checkpoint), so the training data is not needed here.

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
