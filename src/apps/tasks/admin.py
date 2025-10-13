from django.contrib import admin

from .models import Task


@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "user",
        "assignee",
        "priority",
        "is_completed",
        "due_date",
        "created_at",
        "completed_at",
    )
    list_filter = ("is_completed", "created_at", "priority", "due_date", "user", "assignee")
    search_fields = ("title", "description", "user__username", "assignee__username")
