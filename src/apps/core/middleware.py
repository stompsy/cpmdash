from collections.abc import Callable

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.urls import reverse


class LoginRequiredForDashboardMiddleware:
    """Optionally force login for /dashboard/* routes when LOGIN_REQUIRED=True in settings.

    This is off by default to preserve current behavior and tests. Enable by setting LOGIN_REQUIRED=True.
    """

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response: Callable[[HttpRequest], HttpResponse] = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        if getattr(settings, "LOGIN_REQUIRED", False):
            path = request.path
            if path.startswith("/dashboard/") and not request.user.is_authenticated:
                login_url = settings.LOGIN_URL or reverse("login")
                return redirect(f"{login_url}?next={path}")
        return self.get_response(request)
