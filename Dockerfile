FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[server,agents,video]"

# Kalıcı veri Railway volume'una yazılır (DUALODDS_DATA_DIR, varsayılan /data)
ENV DUALODDS_DATA_DIR=/data
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn dualodds.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
