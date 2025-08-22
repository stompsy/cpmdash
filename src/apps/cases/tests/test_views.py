import pytest
from django.urls import reverse
from django.utils import timezone

from apps.core.models import ODReferrals
from utils.theme import get_theme_from_request, set_theme_cookie_response

pytestmark = pytest.mark.django_db


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


def test_opshield_page(client):
    resp = client.get(reverse("cases:opshield"))
    assert resp.status_code == 200
    assert b"Operation Shielding Hope" in resp.content


def test_shiftcoverage_page(client, monkeypatch):
    from apps.cases import views

    monkeypatch.setattr(
        views,
        "build_chart_od_density_heatmap",
        lambda theme=None: ("<div id='density'></div>", None),
    )
    monkeypatch.setattr(
        views, "build_chart_od_hourly_breakdown", lambda theme=None: "<div id='hourly'></div>"
    )
    monkeypatch.setattr(
        views, "build_chart_day_of_week_totals", lambda theme=None: "<div id='dow'></div>"
    )
    monkeypatch.setattr(
        views, "build_chart_shift_scenarios", lambda theme=None: "<div id='scenarios'></div>"
    )
    monkeypatch.setattr(
        views, "build_chart_cost_benefit_analysis", lambda theme=None: "<div id='cost'></div>"
    )
    monkeypatch.setattr(
        views, "calculate_coverage_scenarios", lambda: {"scenario": {"coverage": 10}}
    )
    resp = client.get(reverse("cases:shiftcoverage"))
    assert resp.status_code == 200
    assert b"density" in resp.content
    assert b"hourly" in resp.content


def test_shiftcoverage_zero_data(client, monkeypatch, db):
    from apps.cases import views

    ODReferrals.objects.all().delete()  # zero data path
    monkeypatch.setattr(
        views,
        "build_chart_od_density_heatmap",
        lambda theme=None: ("<div id='density'></div>", None),
    )
    monkeypatch.setattr(
        views, "build_chart_od_hourly_breakdown", lambda theme=None: "<div id='hourly'></div>"
    )
    monkeypatch.setattr(
        views, "build_chart_day_of_week_totals", lambda theme=None: "<div id='dow'></div>"
    )
    monkeypatch.setattr(
        views, "build_chart_shift_scenarios", lambda theme=None: "<div id='scenarios'></div>"
    )
    monkeypatch.setattr(
        views, "build_chart_cost_benefit_analysis", lambda theme=None: "<div id='cost'></div>"
    )
    monkeypatch.setattr(views, "calculate_coverage_scenarios", lambda: {})
    resp = client.get(reverse("cases:shiftcoverage"))
    assert resp.status_code == 200
    # Should still render even if no data
    assert b"shift_coverage" not in resp.content  # placeholder sanity check


def test_repeatods_page_multiple_years(client, monkeypatch, odreferral_factory):
    from apps.cases import views

    monkeypatch.setattr(
        views, "build_chart_repeats_scatter", lambda theme=None: "<div id='repeats'></div>"
    )
    # Create repeats across two years
    base = timezone.now()
    for year_offset in [0, 1]:
        for _ in range(2):  # two events for same patient to count as repeat
            odreferral_factory(
                patient_id=100 + year_offset,
                od_date=base.replace(year=base.year - year_offset),
            )
    resp = client.get(reverse("cases:repeatods"))
    assert resp.status_code == 200
    assert b"repeats" in resp.content


def test_costsavings_page(client):
    resp = client.get(reverse("cases:costsavings"))
    assert resp.status_code == 200
    assert b"Cost Savings" in resp.content or b"Cost" in resp.content


def test_htmx_heatmap_endpoint(client, monkeypatch):
    from apps.cases import views

    monkeypatch.setattr(
        views,
        "build_chart_od_density_heatmap",
        lambda theme=None: ("<svg id='density'></svg>", None),
    )
    resp = client.get(reverse("cases:htmx_heatmap"))
    assert resp.status_code == 200
    assert b"density" in resp.content


def test_htmx_hourly_breakdown_endpoint(client, monkeypatch):
    from apps.cases import views

    monkeypatch.setattr(
        views, "build_chart_od_hourly_breakdown", lambda theme=None: "<svg id='hourly'></svg>"
    )
    resp = client.get(reverse("cases:htmx_hourly_breakdown"))
    assert resp.status_code == 200
    assert b"hourly" in resp.content


def test_htmx_shift_scenarios_endpoint(client, monkeypatch):
    from apps.cases import views

    monkeypatch.setattr(
        views, "build_chart_shift_scenarios", lambda theme=None: "<div id='shift'></div>"
    )
    resp = client.get(reverse("cases:htmx_shift_scenarios"))
    assert resp.status_code == 200
    assert b"shift" in resp.content


def test_htmx_cost_benefit_endpoint(client, monkeypatch):
    from apps.cases import views

    monkeypatch.setattr(
        views, "build_chart_cost_benefit_analysis", lambda theme=None: "<div id='cost'></div>"
    )
    resp = client.get(reverse("cases:htmx_cost_benefit"))
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
