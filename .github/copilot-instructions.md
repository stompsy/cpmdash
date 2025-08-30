# cpmdash AI Coding Agent Instructions

This document provides essential guidance for AI coding agents working on the `cpmdash` codebase.

## Project Overview

`cpmdash` is a Django web application for community paramedicine analytics. It helps visualize and analyze data related to patient cases, referrals, and overdose incidents.

The project follows a standard Django architecture, with business logic organized into discrete applications within the `src/apps` directory.

## Core Technologies

-   **Backend:** Django
-   **Dependency Management:** `uv` (configured in `pyproject.toml`)
-   **Frontend Interactivity:** HTMX, Alpine.js
-   **Styling:** Tailwind CSS
-   **Data Visualization:** Plotly
-   **Linting/Formatting:** `ruff`
-   **Type Checking:** `mypy`

## Developer Workflow

All common development tasks are managed through the `Makefile`. Use these commands as your primary workflow:

-   `make setup`: Install/sync dependencies using `uv`.
-   `make lint`: Check for code style issues with `ruff`.
-   `make fmt`: Format the code with `ruff`.
-   `make type`: Run static type analysis with `mypy`.
-   `make test`: Execute the test suite using `pytest`.
-   `make run`: Start the Django development server.
-   `make migrate`: Create and apply database migrations.

## Project Structure

-   `src/cpmdash`: The main Django project configuration.
-   `src/manage.py`: Django's command-line utility.
-   `src/apps/`: Contains the individual Django applications.
    -   `core`: Shared models, views, and business logic.
    -   `cases`: Manages patient case information.
    -   `dashboard`: The main user-facing dashboard views.
    -   `charts`: Handles data processing and generation of visualizations using Plotly. Logic for specific charts (e.g., overdose, referral) is located here.
-   `src/utils/`: Project-wide utility modules, including `plotly.py` for chart creation.
-   `templates/`: Global Django templates. Each app also has its own `templates` directory.
-   `assets/`: Source frontend assets (e.g., `css/input.css` for Tailwind).
-   `staticfiles/`: Compiled static files served by the application.

## Frontend Development

-   **Styling:** Styles are written using Tailwind CSS in `assets/css/input.css`. A build process (managed outside this instruction set, via `npm`) compiles it to `static/css/output.css`. When adding or modifying styles, edit the source file.
-   **Interactivity:** The project uses HTMX and Alpine.js to enhance frontend interactivity without a large JavaScript framework. Look for attributes in the Django templates (`.html` files) to understand dynamic behavior.

## Data Visualization with Plotly

-   Charts are generated using the Plotly library.
-   The core chart generation logic is in `src/utils/plotly.py`.
-   Data preparation and view-specific chart logic are located in the `src/apps/charts` folder. For example, `src/apps/charts/overdose/` contains logic specific to overdose-related visualizations.
-   When asked to create or modify a chart, identify the relevant view in `src/apps/cases/views.py`, `src/apps/core/views.py`, and `src/apps/dashboard/views.py` (or other app views) and trace the data pipeline back to the models.

## Conventions

-   All code should be formatted with `ruff` (`make fmt`) and pass type checks (`make type`).
-   Follow the existing Django app structure when adding new features. Create a new app if the feature is a distinct functional domain.
-   Models should be defined in the `models.py` of their respective apps.
-   Views should handle the request/response cycle, delegating complex business or data logic to other modules.
