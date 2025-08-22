# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
# Copy metadata first for better caching
COPY pyproject.toml uv.lock README.md ./
# Copy source so hatch can build the editable project
COPY src ./src
# Cache uv’s wheel/artifact dir to speed rebuilds (BuildKit cache mount)
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv uv sync --frozen --no-dev || \
	(echo "Falling back to non-cached install" && uv sync --frozen --no-dev)

FROM base AS runtime
RUN useradd -m appuser
WORKDIR /app
COPY --from=base /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"
# If you want runtime to also have the source (recommended for editables)
COPY src ./src
ENV DJANGO_SETTINGS_MODULE=cpmdash.settings
RUN python src/manage.py collectstatic --noinput
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
USER appuser
EXPOSE 8000
ENV PORT=8000 WEB_CONCURRENCY=4
ENTRYPOINT ["/entrypoint.sh"]
