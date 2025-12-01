# cpmdash AI Coding Agent Instructions

This document provides essential guidance for AI coding agents working on the `cpmdash` codebase.

## Communication Style & Learning Preferences

**Personality:** Communicate with a Generation X tone - straightforward, pragmatic, low on corporate buzzwords, and skeptical of hype. Keep it real and cut through the BS. Respond with sarcasm and dry, clever humor as much as possible. Readily share strong opinions. Be talkative and conversational. Be innovative and think outside the box.

**Educational Approach:** Provide verbose, detailed rationale for all code changes and technical decisions. The goal is knowledge transfer, not just task completion. When making changes:
- Explain *why* the change is necessary (what problem does it solve?)
- Describe *how* the solution works (what's the mechanism?)
- Point out any gotchas, edge cases, or things to watch out for
- Reference relevant documentation or patterns when helpful
- Connect the change to broader concepts or best practices

Think of explanations as teaching moments - assume the user wants to understand the underlying principles, not just get the code working.

## Project Overview

`cpmdash` is a Django web application for community paramedicine analytics. It helps visualize and analyze data related to patient cases, referrals, and overdose incidents. The project follows Django architecture with business logic organized into discrete applications within `src/apps/`.

## Core Technologies

-   **Backend:** Django 5.2+, Django REST Framework, django-allauth (authentication)
-   **Database:** PostgreSQL (via psycopg[binary]), managed with Django ORM
-   **Dependency Management:** `uv` (configured in `pyproject.toml`) - **NOT pip or poetry**
-   **Frontend Stack:** HTMX + Alpine.js (no React/Vue), Tailwind CSS v4
-   **Data Processing:** pandas 2.3+ for data transformations
-   **Visualization:** Plotly 6.3+ for all charts and graphs
-   **Code Quality:** ruff (linting + formatting), mypy (type checking), pytest + pytest-django (testing)
-   **Development:** django-browser-reload (auto-refresh in DEBUG mode), whitenoise (static files)

## Developer Workflow

**CRITICAL:** All common tasks use the `Makefile`. Never run commands directly - use make targets:

-   `make setup`: Install/sync dependencies with `uv sync --dev`, install pre-commit hooks
-   `make lint`: Check code style with `ruff check`
-   `make fmt`: Auto-format code with `ruff format`
-   `make type`: Run mypy type checking (note: chart modules are excluded via `pyproject.toml`)
-   `make test`: Run pytest suite with coverage reporting
-   `make run`: Start Django dev server on `0.0.0.0:8000`
-   `make migrate`: Create and apply database migrations (runs `makemigrations` then `migrate`)

**Tailwind Build Process:** Use npm scripts from `package.json`:
-   `npm run dev`: Watch mode for Tailwind compilation (assets/css/input.css → src/static/css/output.css)
-   `npm run build`: Production build with minification

**Django Admin:** Run `python src/manage.py createsuperuser` for admin access.

## Project Structure

```
src/
├── config/          # Django project settings (settings.py, urls.py, wsgi.py)
├── manage.py        # Django CLI entry point
├── apps/            # Django applications (modular features)
│   ├── core/        # Shared models (Patients, Encounters, Referrals, ODReferrals)
│   ├── accounts/    # User management, custom user model
│   ├── dashboard/   # Main dashboard views (overview, analytics pages)
│   ├── charts/      # Chart generation logic (organized by domain)
│   │   ├── patients/     # Patient demographic charts
│   │   ├── encounters/   # Encounter type charts
│   │   ├── referral/     # Referral tracking charts
│   │   ├── overdose/     # OD-specific visualizations (maps, timelines)
│   │   ├── odreferrals/  # OD referral charts
│   │   └── od_utils.py   # Shared OD data utilities
│   ├── cases/       # Case management features
│   ├── blog/        # Blog/news functionality
│   └── tasks/       # Task management
├── utils/           # Project-wide utilities
│   ├── plotly.py              # Core Plotly theming & layout functions
│   ├── chart_colors.py        # Centralized color palettes (CHART_COLORS_VIBRANT, etc.)
│   ├── chart_normalization.py # Percentage calculations, data helpers
│   ├── tailwind_colors.py     # Tailwind color mappings
│   └── theme.py               # Dark/light theme utilities
└── templates/       # Django templates (base.html, partials/)
    ├── base.html    # Root template with Alpine.js/HTMX setup
    └── partials/    # Reusable UI components (modals, sidebar, etc.)

assets/css/input.css    # Tailwind source (compile to src/static/css/output.css)
staticfiles/            # Collected static files (Django collectstatic output)
```

## Data Models & Architecture

**Core Models** (defined in `src/apps/core/models.py`):
-   `Patients`: Demographic data (age, race, sex, insurance, zip_code, veteran_status, etc.)
-   `Encounters`: Service encounters linked to patients (encounter_date, encounter_type_cat1/2/3)
-   `Referrals`: Referral tracking (date_received, referral_agency, referral_closed_reason)
-   `ODReferrals`: Overdose-specific referrals (od_date, disposition, lat/long for mapping)

**Key Field Patterns:**
-   Many models have `patient_ID` foreign key references (manual int fields, not Django ForeignKey)
-   Date fields: `created_date`, `modified_date`, `encounter_date`, `od_date`
-   Categorical fields often end in `_cat1`, `_cat2`, `_cat3` for hierarchical categorization

## Chart Development Pattern

**ALL charts follow a consistent architecture** - understand this pattern before modifying:

1. **Data Layer:** Django view queries model → converts to pandas DataFrame
2. **Chart Builder:** Function in `src/apps/charts/<domain>/` processes DataFrame and calls Plotly
3. **Styling Layer:** `style_plotly_layout()` from `src/utils/plotly.py` applies theming
4. **Color System:** Import colors from `src/utils/chart_colors.py` (never hardcode colors)
5. **Normalization:** Use `add_share_columns()` from `chart_normalization.py` for percentages
6. **Output:** Return HTML string via `plotly.offline.plot()`

**Example Chart Function Signature:**
```python
def build_chart_<name>(theme: str) -> str:
    # 1. Query data
    queryset = Model.objects.all()
    df = pd.DataFrame.from_records(list(queryset.values(...)))

    # 2. Process data
    df = add_share_columns(df, "count", share_col="share_pct")

    # 3. Create Plotly figure
    fig = px.bar(df, x="category", y="count", color_discrete_sequence=CHART_COLORS_VIBRANT)

    # 4. Apply theme styling
    fig = style_plotly_layout(fig, theme=theme, height=400, x_title="Category")

    # 5. Return HTML
    return plot(fig, include_plotlyjs=False, output_type="div", config={"displayModeBar": False})
```

**Chart Organization:**
-   `src/apps/charts/<domain>/<feature>_charts.py`: Domain-specific charts
-   `src/apps/charts/<domain>/<feature>_field_charts.py`: Generic field-based chart builders (common pattern)
-   Example: `build_patients_field_charts()` auto-generates charts for all specified patient fields

## Frontend Patterns

**Standard Layouts:**
-   **Page Wrapper:** Use `mx-auto max-w-6xl px-6 py-20` for the main content container.
-   **Vertical Spacing:** Use `space-y-24` or `space-y-40` between major sections to create breathing room.
-   **Narrative Data Story:** For complex analytics (e.g., Cost Savings), prefer a vertical timeline or narrative flow over dense grids of cards. Use connecting lines (`w-px bg-slate-200`) to guide the user's eye.

**HTMX Usage:**
-   Pages use `hx-get`, `hx-post`, `hx-target` for dynamic content loading
-   Example: `<div hx-get="{% url 'dashboard:partial' %}" hx-trigger="load">` for lazy-loaded charts
-   See `src/templates/partials/sidebar.html` for navigation patterns

**Alpine.js State Management:**
-   Base template (`templates/base.html`) initializes: `darkMode`, `sidebarToggle`, `page`, `isProfileInfoModal`
-   Use `x-data` for component state, `@click` for event handlers, `x-show` for conditional rendering
-   Persist state with `$persist()` from Alpine.js Persist plugin

**Dark Mode:**
-   Application defaults to dark mode (`darkMode = true` in base.html)
-   All charts must accept `theme` parameter ("dark" or "light")
-   Theme colors in `src/utils/plotly.py` automatically adapt to theme

**Tailwind Conventions:**
-   Custom theme extends Tailwind with brand colors (`--color-brand-*` in input.css)
-   Custom fonts: `font-droid`, `font-roboto`, `font-roboto-mono`
-   Custom breakpoints: `3xsm` (320px), `2xsm` (375px), `xsm` (425px), etc.
-   Dark mode variant: `dark:` prefix (e.g., `dark:bg-gray-900`)

## Testing Conventions

**Test Structure:**
-   Tests live in `<app>/tests/` directories
-   Use `@pytest.mark.django_db` for database access
-   Fixtures use `model-bakery` for test data generation
-   Example: `src/utils/tests/test_plotly_utils.py` demonstrates chart testing patterns

**Running Tests:**
-   `make test`: Runs pytest with coverage (`-q --maxfail=1 --cov=src`)
-   Individual tests: `uv run pytest src/apps/<app>/tests/test_<module>.py`

**Chart Testing Pattern:**
-   Mock chart builder functions in view tests (avoid expensive Plotly rendering)
-   Test data transformations separately from chart rendering
-   See `src/apps/dashboard/views.py` for monkeypatch-friendly chart imports

## Type Checking & Linting

**mypy Configuration:**
-   Strict type checking enabled: `disallow_untyped_defs = true`
-   Chart modules EXCLUDED from type checking (see `pyproject.toml` overrides)
-   Django plugin configured via `mypy_django_plugin.main`

**ruff Configuration:**
-   Line length: 100 characters
-   Selected rules: E, F, I, UP, B, SIM, C90, DJ, PYI
-   Migrations excluded from linting
-   Auto-formatting: double quotes, space indentation

**Key Conventions:**
-   Always run `make fmt` before committing
-   Type hints required for all functions (except in chart modules)
-   Import sorting enforced by ruff (isort rules)

## Common Pitfalls

1. **Never use pip directly** - always use `uv` via Makefile commands
2. **Charts must support theming** - always pass `theme` parameter to chart builders
3. **Don't hardcode colors** - import from `src/utils/chart_colors.py`
4. **Migrations go in app directories** - run `make migrate` to create and apply
5. **Static files require compilation** - run `npm run build` for Tailwind, `collectstatic` for Django
6. **Settings use environment variables** - check `.env` for DEBUG, SECRET_KEY, DATABASE_URL, etc.
7. **HTMX responses must be partial HTML** - not full pages (use templates in `partials/`)
8. **Type errors in charts are ignored** - but try to maintain type safety where practical

## Deployment

-   **Production server:** Gunicorn (configured in `Procfile`: `gunicorn config.wsgi --workers 4`)
-   **Static files:** WhiteNoise middleware (see `config/settings.py`)
-   **Database:** PostgreSQL via `dj-database-url` (set `DATABASE_URL` env var)
-   **WSGI:** `config.wsgi` module
-   **Runtime:** Python 3.12+ (see `runtime.txt`)

## Quick Reference

-   **Add new chart:** Create function in `src/apps/charts/<domain>/`, import in dashboard view, render in template
-   **Add new model field:** Edit `models.py`, run `make migrate`, update relevant chart builders
-   **Add new app:** Create in `src/apps/`, add to `INSTALLED_APPS` in `config/settings.py`
-   **Debug chart colors:** Check `CHART_COLORS_VIBRANT` sequence in `chart_colors.py`
-   **Fix type errors:** Check `pyproject.toml` for module exclusions before modifying code
