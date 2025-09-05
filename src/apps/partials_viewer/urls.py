from django.urls import path

from . import views

app_name = "partials_viewer"

urlpatterns = [
    path("", views.list_partials, name="list"),
    path("<path:partial_path>", views.view_partial, name="view"),
]
