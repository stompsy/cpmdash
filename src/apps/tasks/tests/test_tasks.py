import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

from apps.tasks.models import Task


@pytest.mark.django_db
def test_task_model_defaults():
    User = get_user_model()
    user = User.objects.create_user(username="alice", password="pass")  # type: ignore[attr-defined]
    t = Task.objects.create(user=user, title="A")
    assert t.is_completed is False
    assert t.description == ""
    assert t.completed_at is None


@pytest.mark.django_db
def test_tasks_list_create_and_toggle_htmx(client: Client):
    User = get_user_model()
    user = User.objects.create_user(username="bob", password="pass")  # type: ignore[attr-defined]
    assert client.login(username="bob", password="pass")

    # List empty
    resp = client.get(reverse("tasks:list"))
    assert resp.status_code == 200
    assert b"No tasks yet" in resp.content

    # Create via POST
    resp = client.post(reverse("tasks:list"), {"title": "First", "description": "d"})
    assert resp.status_code in (302, 303)
    t = Task.objects.get(user=user)
    assert t.title == "First"
    assert t.is_completed is False

    # Toggle via HTMX POST should return updated row HTML
    toggle_url = reverse("tasks:toggle", args=[t.pk])
    resp = client.post(toggle_url, HTTP_HX_REQUEST="true")
    assert resp.status_code == 200
    t.refresh_from_db()
    assert t.is_completed is True
    assert b"task-%d" % t.pk in resp.content


@pytest.mark.django_db
def test_tasks_update_and_delete(client: Client):
    User = get_user_model()
    user = User.objects.create_user(username="carol", password="pass")  # type: ignore[attr-defined]
    assert client.login(username="carol", password="pass")
    t = Task.objects.create(user=user, title="Edit me")

    # Update
    edit_url = reverse("tasks:edit", args=[t.pk])
    resp = client.get(edit_url)
    assert resp.status_code == 200
    resp = client.post(edit_url, {"title": "Edited", "description": "new"})
    assert resp.status_code in (302, 303)
    t.refresh_from_db()
    assert t.title == "Edited"
    assert t.description == "new"

    # Delete via HTMX
    del_url = reverse("tasks:delete_htmx", args=[t.pk])
    resp = client.post(del_url, HTTP_HX_REQUEST="true")
    assert resp.status_code == 200
    assert Task.objects.filter(pk=t.pk).exists() is False
    # List HTML returned (empty state) and badge present (OOB)
    assert b"No tasks yet" in resp.content
    assert b"tasks-badge" in resp.content
