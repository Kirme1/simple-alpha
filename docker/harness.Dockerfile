FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY docker/requirements-harness.txt .
RUN pip install --no-cache-dir -r requirements-harness.txt

COPY llm_harness ./llm_harness

EXPOSE 8002
CMD ["uvicorn", "llm_harness.ws_server:app", "--host", "0.0.0.0", "--port", "8002"]
