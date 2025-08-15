import pytest
from django.test import Client


@pytest.mark.django_db
def test_healthz() -> None:
    client = Client()
    resp = client.get("/api/healthz/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
