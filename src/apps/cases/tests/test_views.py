import pytest
from django.urls import reverse
from django.utils import timezone

from apps.core.models import ODReferrals
from utils.theme import get_theme_from_request, set_theme_cookie_response

pytestmark = pytest.mark.django_db


@pytest.fixture()
def authenticated_client(client, django_user_model):
    user = django_user_model.objects.create_user("testuser", "test@example.com", "password123")
    client.force_login(user)
    return client


@pytest.fixture
def odreferral_factory():
    def make(**kwargs):
        defaults = {"od_date": timezone.now(), "patient_id": 1}
        defaults.update(kwargs)
        return ODReferrals.objects.create(**defaults)

    return make


@pytest.fixture(autouse=True)
def minimal_odreferral(odreferral_factory):
    return odreferral_factory()


def test_opshield_page(authenticated_client):
    resp = authenticated_client.get(reverse("cases:opshield"))
    assert resp.status_code == 200
    assert b"Operation Shielding Hope" in resp.content


def test_costsavings_page(authenticated_client):
    resp = authenticated_client.get(reverse("cases:costsavings"))
    assert resp.status_code == 200
    assert b"Cost Savings" in resp.content or b"Cost" in resp.content


# Utility tests


def test_get_theme_from_request(rf):
    req = rf.get("/")
    assert get_theme_from_request(req) == "dark"


def test_set_theme_cookie_response(rf):
    from django.http import HttpResponse

    resp = HttpResponse("ok")
    set_theme_cookie_response(resp, "dark")
    assert resp.cookies["theme"].value == "dark"
