#!/usr/bin/env sh
set -e

# Apply database migrations
python src/manage.py migrate --noinput

# Optionally collect static at runtime (usually done at build, but safe if repeated)
if [ "${COLLECTSTATIC_AT_RUNTIME:-0}" = "1" ]; then
  python src/manage.py collectstatic --noinput
fi

# Start gunicorn / uvicorn hybrid workers binding to provided PORT (Railway sets $PORT)
exec gunicorn -w ${WEB_CONCURRENCY:-4} -k uvicorn.workers.UvicornWorker cpmdash.asgi:application --bind 0.0.0.0:${PORT:-8000}
