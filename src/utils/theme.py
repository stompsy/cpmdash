from django.conf import settings


def get_theme_from_request(request):
    """
    Get theme preference - temporarily hardcoded to dark mode
    """
    # Temporarily force dark mode for all charts
    return "dark"


def set_theme_cookie_response(response, theme):
    """Helper to set theme cookie on response"""
    if theme in ["dark", "light"]:
        response.set_cookie("theme", theme, max_age=365*24*60*60)  # 1 year
    return response
