from django.conf import settings


def get_theme_from_request(request):
    return request.GET.get("theme") or getattr(settings, "PLOTLY_THEME", "light")
