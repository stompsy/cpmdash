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


def test_shiftcoverage_redirects_to_dashboard(authenticated_client):
    resp = authenticated_client.get(reverse("cases:shiftcoverage"))
    assert resp.status_code == 302
    assert resp.url == reverse("dashboard:odreferrals_shift_coverage")


def test_costsavings_page(authenticated_client):
    resp = authenticated_client.get(reverse("cases:costsavings"))
    assert resp.status_code == 200
    assert b"Cost Savings" in resp.content or b"Cost" in resp.content


def test_htmx_heatmap_endpoint(authenticated_client, monkeypatch):
    from apps.cases import views

    monkeypatch.setattr(
        views,
        "build_chart_od_density_heatmap",
        lambda theme=None: ("<svg id='density'></svg>", None),
    )
    resp = authenticated_client.get(reverse("cases:htmx_heatmap"))
    assert resp.status_code == 200
    assert b"density" in resp.content


def test_htmx_hourly_breakdown_endpoint(authenticated_client, monkeypatch):
    from apps.cases import views

    monkeypatch.setattr(
        views, "build_chart_od_hourly_breakdown", lambda theme=None: "<svg id='hourly'></svg>"
    )
    resp = authenticated_client.get(reverse("cases:htmx_hourly_breakdown"))
    assert resp.status_code == 200
    assert b"hourly" in resp.content


def test_htmx_shift_scenarios_endpoint(authenticated_client, monkeypatch):
    from apps.cases import views

    monkeypatch.setattr(
        views, "build_chart_shift_scenarios", lambda theme=None: "<div id='shift'></div>"
    )
    resp = authenticated_client.get(reverse("cases:htmx_shift_scenarios"))
    assert resp.status_code == 200
    assert b"shift" in resp.content


def test_htmx_cost_benefit_endpoint(authenticated_client, monkeypatch):
    from apps.cases import views

    monkeypatch.setattr(
        views, "build_chart_cost_benefit_analysis", lambda theme=None: "<div id='cost'></div>"
    )
    resp = authenticated_client.get(reverse("cases:htmx_cost_benefit"))
    assert resp.status_code == 200
    assert b"cost" in resp.content


# Utility tests


def test_get_theme_from_request(rf):
    req = rf.get("/")
    assert get_theme_from_request(req) == "dark"


def test_set_theme_cookie_response(rf):
    from django.http import HttpResponse

    resp = HttpResponse("ok")
    set_theme_cookie_response(resp, "dark")
    assert resp.cookies["theme"].value == "dark"
