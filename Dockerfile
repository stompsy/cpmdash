# ── Stage 1: Build Tailwind CSS ──────────────────────────────────
FROM node:22-alpine AS css-builder
WORKDIR /build
COPY package.json ./
RUN npm install
COPY assets/css/ assets/css/
COPY src/static/ src/static/
COPY src/templates/ src/templates/
COPY src/apps/ src/apps/
RUN npm run build

# ── Stage 2: Python application ─────────────────────────────────
FROM python:3.12-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# System deps for psycopg, geopandas, scipy
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev gcc libgeos-dev libproj-dev gdal-bin && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python deps (cache-friendly: copy lock + metadata first)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen

# Copy application code
COPY src/ src/
COPY Procfile runtime.txt ./

# Copy built Tailwind CSS from stage 1
COPY --from=css-builder /build/src/static/css/output.css src/static/css/output.css

# Copy static assets that ship with the repo
COPY staticfiles/ staticfiles/
COPY media/ media/

# Collect static files (WhiteNoise serves from STATIC_ROOT)
RUN SECRET_KEY=build-placeholder \
    DATABASE_URL=sqlite:///tmp/build.db \
    uv run python src/manage.py collectstatic --noinput

EXPOSE 8000

CMD ["uv", "run", "gunicorn", "config.wsgi", "--workers", "4", "--timeout", "180", "--graceful-timeout", "30", "--log-level", "info", "--chdir", "src", "--bind", "0.0.0.0:8000"]
