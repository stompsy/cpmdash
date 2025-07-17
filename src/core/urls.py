from django.contrib import admin
from django.urls import include, path
from . import views


urlpatterns = [
    path("", views.opshield, name="opshield"),
    path("cases/", include("cases.urls")),
    path("dashboard/", include("dashboard.urls")),
    path("timeline/", include("timeline.urls")),
    path("profile/", views.user_profile, name="user_profile"),
    path("health/", views.health_check, name="health_check"),
    path("admin/", admin.site.urls),
    path("__reload__/", include("django_browser_reload.urls")),
]
