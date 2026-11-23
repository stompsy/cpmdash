.PHONY: setup lint fmt type test run migrate shell tailwind loaddata loaddata-dry
setup:
	uv sync --dev
	pre-commit install --hook-type pre-commit || uvx pre-commit install --hook-type pre-commit
	mkdir -p .git/hooks
	cp scripts/git-hooks/pre-push .git/hooks/pre-push
	chmod +x .git/hooks/pre-push
lint:
	uv run ruff check
fmt:
	uv run ruff format
type:
	uv run mypy src
test:
	uv run pytest -q
# Bind the IPv6 wildcard: with net.ipv6.bindv6only=0 (the Linux default) this accepts
# IPv4 too, so the server is reachable whether "localhost" resolves to 127.0.0.1 or ::1.
# Plain 0.0.0.0 is IPv4-only and is refused by clients that prefer ::1.
run:
	uv run python src/manage.py runserver "[::]:8000"
migrate:
	uv run python src/manage.py makemigrations && uv run python src/manage.py migrate

shell:
	uv run python src/manage.py shell

tailwind:
	npm run dev

loaddata:
	uv run python src/manage.py load_csv_data

loaddata-dry:
	uv run python src/manage.py load_csv_data --dry-run
