import pytest
from django.contrib.auth import get_user_model
from django.test import Client


@pytest.mark.django_db
def test_healthz() -> None:
    User = get_user_model()
    user = User.objects.create_user("testuser", "test@example.com", "password123")
    client = Client()
    client.force_login(user)
    resp = client.get("/api/healthz/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
