# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

FROM base AS runtime
WORKDIR /app
COPY --from=base /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"
COPY src ./src
ENV DJANGO_SETTINGS_MODULE=cpmdash.settings
RUN python src/manage.py collectstatic --noinput
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8000
ENV PORT=8000 WEB_CONCURRENCY=4
ENTRYPOINT ["/entrypoint.sh"]
