# AXIOM backend — production image.
#
# The schema is owned by the Alembic migration chain; the app applies
# `alembic upgrade head` at boot (RUN_MIGRATIONS_ON_STARTUP=true, the
# default) before serving traffic. One worker per container: scale with
# replicas — W2 moved trace and eval-job state to Postgres so any number
# of workers can share one Postgres/Redis pair.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libmagic backs python-magic (upload content sniffing).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

COPY . .

# Non-root: the API needs no elevated privileges.
RUN useradd --create-home axiom \
    && chown -R axiom:axiom /app
USER axiom

EXPOSE 8000

# /api/health is public and dependency-light; start-period covers model
# loading (reranker) and first-boot migrations.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "1800"]
