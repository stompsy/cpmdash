from collections.abc import Callable

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse


class GlobalLoginRequiredMiddleware:
    """
    Middleware that requires users to be authenticated for all views
    except login, logout, and password reset flows.
    """

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response: Callable[[HttpRequest], HttpResponse] = get_response
        # URLs that don't require authentication
        self.exempt_paths = [
            "/accounts/login/",
            "/accounts/logout/",
            "/accounts/password/reset/",
            "/accounts/password/reset/done/",
            "/accounts/password/reset/key/",
            "/accounts/password/reset/key/done/",
            "/admin/login/",  # Admin login should work
            "/__reload__/",  # django-browser-reload (dev only)
        ]

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Check if user is authenticated
        if not request.user.is_authenticated:
            # Check if the path is exempt from authentication
            path = request.path_info
            is_exempt = any(path.startswith(exempt) for exempt in self.exempt_paths)

            if not is_exempt:
                # Return access denied page with ocean background
                return render(
                    request,
                    "access_denied.html",
                    {
                        "current_year": 2025,
                        "request": request,
                    },
                    status=403,
                )

        response = self.get_response(request)
        return response


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
