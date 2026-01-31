.PHONY: setup lint fmt type test run migrate shell analyze-initiated-referrals analyze-initiated-referrals-db
setup:
	uv sync --dev
	pre-commit install || uvx pre-commit install
lint:
	uv run ruff check
fmt:
	uv run ruff format
type:
	uv run mypy src
test:
	uv run pytest -q
run:
	uv run python src/manage.py runserver 0.0.0.0:8000
migrate:
	uv run python src/manage.py makemigrations && uv run python src/manage.py migrate

shell:
	uv run python src/manage.py shell
