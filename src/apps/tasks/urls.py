from django.urls import path

from .views import (
    TaskDeleteView,
    TaskListCreateView,
    TaskUpdateView,
    delete_htmx,
    export_csv,
    task_row_cancel,
    task_row_edit,
    task_row_update,
    toggle_complete,
)

app_name = "tasks"

urlpatterns = [
    path("", TaskListCreateView.as_view(), name="list"),
    path("<int:pk>/edit/", TaskUpdateView.as_view(), name="edit"),
    path("<int:pk>/delete/", TaskDeleteView.as_view(), name="delete"),
    path("<int:pk>/toggle/", toggle_complete, name="toggle"),
    path("<int:pk>/delete-htmx/", delete_htmx, name="delete_htmx"),
    path("<int:pk>/row/edit/", task_row_edit, name="row_edit"),
    path("<int:pk>/row/cancel/", task_row_cancel, name="row_cancel"),
    path("<int:pk>/row/update/", task_row_update, name="row_update"),
    path("export/csv/", export_csv, name="export_csv"),
]
