from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import QuerySet
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseForbidden,
    HttpResponseNotAllowed,
    StreamingHttpResponse,
)
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.utils.timezone import now
from django.views.generic import ListView, UpdateView

from .forms import TaskForm
from .models import Task

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apps.accounts.models import User as AuthUser


class TaskListCreateView(LoginRequiredMixin, ListView):
    template_name = "tasks/list.html"
    context_object_name = "tasks"
    paginate_by = 10

    def get_queryset(self) -> QuerySet[Task]:
        # Show all tasks ordered by completion status and creation date
        return Task.objects.all().order_by("is_completed", "-created_at")

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context.setdefault("form", TaskForm())
        # Keep page object/paginator already provided by ListView
        return context

    def post(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        form = TaskForm(request.POST)
        if form.is_valid():
            task = form.save(commit=False)
            # Narrow type for mypy: LoginRequiredMixin ensures an authenticated user
            user = cast("AuthUser", request.user)
            task.user_id = cast(int, user.pk)
            task.save()
            messages.success(request, "Task created.")
            if request.headers.get("HX-Request") == "true":
                # Return a single row to prepend and OOB badges
                from django.template.loader import render_to_string

                row_html = render_to_string(
                    "tasks/_task_item.html", {"task": task}, request=request
                )
                badges_html = _render_badges_oob(request)
                response = HttpResponse(row_html + badges_html)
                response["HX-Trigger"] = "task-created"
                return response
        else:
            messages.error(request, "Please correct the errors.")
        # If HTMX invalid, return the form fragment OOB to replace the form
        if request.headers.get("HX-Request") == "true":
            from django.template.loader import render_to_string

            form_html = render_to_string(
                "tasks/_task_create_form.html", {"form": form}, request=request
            )
            # Mark as out-of-band so it replaces the form while target is UL
            replaced_html: str = str(form_html).replace("<form", '<form hx-swap-oob="true"', 1)
            return HttpResponse(replaced_html)
        return redirect("tasks:list")


class TaskUpdateView(LoginRequiredMixin, UpdateView):
    model = Task
    form_class = TaskForm
    template_name = "tasks/edit.html"
    success_url = reverse_lazy("tasks:list")

    def get_queryset(self) -> QuerySet[Task]:
        return Task.objects.all()

    def form_valid(self, form: TaskForm) -> HttpResponse:
        messages.success(self.request, "Task updated.")
        return super().form_valid(form)


if TYPE_CHECKING:  # Use generic typing only for static type checkers
    from django.views.generic import DeleteView as _DeleteViewGeneric

    _DeleteViewBase = _DeleteViewGeneric[Task, TaskForm]
else:  # at runtime, use the non-generic class (Django classes are not subscriptable)
    from django.views.generic import DeleteView as _DeleteViewBase


class TaskDeleteView(LoginRequiredMixin, _DeleteViewBase):
    model = Task
    template_name = "tasks/confirm_delete.html"
    success_url = reverse_lazy("tasks:list")

    def get_queryset(self) -> QuerySet[Task]:
        return Task.objects.all()


@login_required
def toggle_complete(request: HttpRequest, pk: int) -> HttpResponse:
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])  # semantics: toggle should be POST
    task = get_object_or_404(Task, pk=pk)
    if not _can_modify(request.user, task):
        return HttpResponseForbidden("Not allowed")
    # Toggle based on current state (checkbox change triggers regardless of checked/unchecked)
    task.is_completed = not task.is_completed
    task.completed_at = now() if task.is_completed else None
    task.save(update_fields=["is_completed", "completed_at"])
    # Return updated row HTML plus OOB badges for sidebar and header
    from django.template.loader import render_to_string

    row_html = render_to_string("tasks/_task_item.html", {"task": task}, request=request)
    badges_html = _render_badges_oob(request)
    return HttpResponse(row_html + badges_html)


@login_required
def delete_htmx(request: HttpRequest, pk: int) -> HttpResponse:
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])  # POST-only delete for HTMX button
    task = get_object_or_404(Task, pk=pk)
    if not _can_modify(request.user, task):
        return HttpResponseForbidden("Not allowed")
    task.delete()
    # Return refreshed list HTML and OOB badge; client can target the list container
    from django.template.loader import render_to_string

    tasks = Task.objects.all().order_by("is_completed", "-created_at")
    list_html = render_to_string("tasks/_tasks_list.html", {"tasks": tasks}, request=request)
    badges_html = _render_badges_oob(request)
    return HttpResponse(list_html + badges_html)


@login_required
def task_row_edit(request: HttpRequest, pk: int) -> HttpResponse:
    task = get_object_or_404(Task, pk=pk)
    if not _can_modify(request.user, task):
        return HttpResponseForbidden("Not allowed")
    from django.template.loader import render_to_string

    form = TaskForm(instance=task)
    html = render_to_string(
        "tasks/_task_item_form.html", {"task": task, "form": form}, request=request
    )
    return HttpResponse(html)


@login_required
def task_row_cancel(request: HttpRequest, pk: int) -> HttpResponse:
    """Cancel editing and return to task item view"""
    task = get_object_or_404(Task, pk=pk)
    if not _can_modify(request.user, task):
        return HttpResponseForbidden("Not allowed")
    from django.template.loader import render_to_string

    html = render_to_string("tasks/_task_item.html", {"task": task}, request=request)
    return HttpResponse(html)


@login_required
def task_row_update(request: HttpRequest, pk: int) -> HttpResponse:
    task = get_object_or_404(Task, pk=pk)
    if not _can_modify(request.user, task):
        return HttpResponseForbidden("Not allowed")
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])  # POST for inline edit
    from django.template.loader import render_to_string

    form = TaskForm(request.POST, instance=task)
    if form.is_valid():
        form.save()
        messages.success(request, "Task updated.")
        row_html = render_to_string("tasks/_task_item.html", {"task": task}, request=request)
        badges_html = _render_badges_oob(request)
        return HttpResponse(row_html + badges_html)
    html = render_to_string(
        "tasks/_task_item_form.html", {"task": task, "form": form}, request=request
    )
    return HttpResponse(html)


@login_required
def export_csv(request: HttpRequest) -> HttpResponse:
    # Export all tasks ordered by completion status and creation date
    qs = Task.objects.all().order_by("is_completed", "-created_at")

    import csv

    def row_iter() -> Iterator[list[str]]:
        yield [
            "ID",
            "Title",
            "Description",
            "Priority",
            "Due Date",
            "Created By",
            "Assignee",
            "Completed",
            "Created At",
            "Completed At",
        ]
        for t in qs.iterator():
            yield [
                str(t.pk),
                str(t.title),
                str(t.description),
                str(Task.Priority(t.priority).label),
                str(t.due_date.isoformat() if t.due_date else ""),
                str(t.user.username),
                str(t.assignee.username if t.assignee else ""),
                "yes" if t.is_completed else "no",
                str(t.created_at.isoformat(timespec="seconds")),
                str(t.completed_at.isoformat(timespec="seconds") if t.completed_at else ""),
            ]

    class Echo:
        def write(self, value: str) -> str:
            return value

    pseudo_buffer = Echo()
    writer = csv.writer(pseudo_buffer)
    response = StreamingHttpResponse(
        (writer.writerow(r) for r in row_iter()), content_type="text/csv"
    )
    response["Content-Disposition"] = "attachment; filename=tasks.csv"
    # Cast to HttpResponse for typing purposes
    return cast(HttpResponse, response)


def _can_modify(user: Any, task: Task) -> bool:
    try:
        owner_id = cast(int, task.user.pk)
        assignee_id_opt = cast("int | None", (task.assignee.pk if task.assignee else None))
        return bool(getattr(user, "is_staff", False) or user.pk in {owner_id, assignee_id_opt})
    except Exception:  # pragma: no cover - defensive
        return False


def _render_badges_oob(request: HttpRequest) -> str:
    from django.template.loader import render_to_string

    count = Task.objects.filter(is_completed=False).count()
    badge_sidebar = render_to_string(
        "tasks/_tasks_badge.html",
        {"incomplete_tasks_count": count, "element_id": "tasks-badge"},
        request=request,
    )
    badge_header = render_to_string(
        "tasks/_tasks_badge.html",
        {"incomplete_tasks_count": count, "element_id": "tasks-badge-header"},
        request=request,
    )
    return badge_sidebar + badge_header
