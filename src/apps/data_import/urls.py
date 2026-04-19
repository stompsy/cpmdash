from django.urls import path

from . import views

app_name = "data_import"

urlpatterns = [
    # History
    path("", views.batch_list, name="batch_list"),
    # Upload
    path("upload/", views.upload, name="upload"),
    path("<int:batch_id>/upload/", views.upload_to_batch, name="upload_to_batch"),
    # Processing
    path("<int:batch_id>/process/", views.process_view, name="process"),
    path("<int:batch_id>/stream/", views.process_stream, name="process_stream"),
    # Review
    path("<int:batch_id>/review/", views.review, name="review"),
    path("<int:batch_id>/review/<str:dataset>/table/", views.review_table, name="review_table"),
    # Inline editing
    path("<int:batch_id>/review/<str:dataset>/<int:row_pk>/edit/", views.row_edit, name="row_edit"),
    path(
        "<int:batch_id>/review/<str:dataset>/<int:row_pk>/update/",
        views.row_update,
        name="row_update",
    ),
    path(
        "<int:batch_id>/review/<str:dataset>/<int:row_pk>/cancel/",
        views.row_cancel,
        name="row_cancel",
    ),
    # Single-cell inline editing
    path(
        "<int:batch_id>/review/<str:dataset>/<int:row_pk>/cell/<str:field_name>/",
        views.cell_update,
        name="cell_update",
    ),
    # Production reference
    path("<int:batch_id>/production-ref/", views.production_reference, name="production_ref"),
    # Batch operations
    path(
        "<int:batch_id>/review/<str:dataset>/batch-update/",
        views.batch_update_field,
        name="batch_update_field",
    ),
    # Commit
    path("<int:batch_id>/commit/", views.commit_view, name="commit"),
    path("<int:batch_id>/post-commit/", views.post_commit, name="post_commit"),
    path("<int:batch_id>/purge/", views.purge_staging, name="purge_staging"),
    path("<int:batch_id>/delete/", views.batch_delete, name="batch_delete"),
    # Logs
    path("logs/", views.log_list, name="log_list"),
    path("logs/<int:log_id>/delete/", views.log_delete, name="log_delete"),
]
