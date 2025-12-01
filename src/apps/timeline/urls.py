from django.urls import path

from .views import (
    timeline_create,
    timeline_delete,
    timeline_get,
    timeline_list_entries,
    timeline_update,
    timeline_view,
)

app_name = "timeline"

urlpatterns = [
    path("", timeline_view, name="index"),
    path("api/entries/", timeline_create, name="create"),
    path("api/entries/<int:entry_id>/", timeline_get, name="get"),
    path("api/entries/<int:entry_id>/update/", timeline_update, name="update"),
    path("api/entries/<int:entry_id>/delete/", timeline_delete, name="delete"),
    path("htmx/entries/", timeline_list_entries, name="list_entries"),
]
