FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY docker/requirements-mcp.txt .
RUN pip install --no-cache-dir -r requirements-mcp.txt

COPY mcp_server ./mcp_server

EXPOSE 8001
CMD ["python", "-m", "mcp_server.server"]
