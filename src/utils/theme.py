from django.http import HttpRequest, HttpResponse


def get_theme_from_request(request: HttpRequest) -> str:
    """Return the active theme based on request (currently forced to dark)."""
    return "dark"


def set_theme_cookie_response(response: HttpResponse, theme: str) -> HttpResponse:
    """Set a persistent theme cookie if valid and return the response."""
    if theme in {"dark", "light"}:
        response.set_cookie("theme", theme, max_age=365 * 24 * 60 * 60)
    return response
