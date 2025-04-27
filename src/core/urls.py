from django.contrib import admin
from django.urls import include, path
from . import views


urlpatterns = [
    path("", views.hello_world, name="hello_world"),
    path("health/", views.health_check, name="health_check"),
    path("admin/", admin.site.urls),
    path("__reload__/", include("django_browser_reload.urls")),
]
