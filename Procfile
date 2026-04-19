web: gunicorn config.wsgi --workers 2 --threads 4 --worker-class gthread --timeout 600 --graceful-timeout 30 --log-level info --chdir src
