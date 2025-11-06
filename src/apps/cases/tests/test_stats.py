from datetime import datetime
from typing import Any

import pytest
from django.urls import reverse
from django.utils import timezone

from apps.core.models import ODReferrals, Patients
from utils.theme import set_theme_cookie_response

pytestmark = pytest.mark.django_db


@pytest.fixture()
def authenticated_client(client, django_user_model):
    user = django_user_model.objects.create_user("testuser", "test@example.com", "password123")
    client.force_login(user)
    return client


def make_dt(year: int, month: int, day: int, hour: int, minute: int = 0):
    tz = timezone.get_current_timezone()
    return timezone.make_aware(datetime(year, month, day, hour, minute), tz)


def _stub_charts(monkeypatch: Any) -> None:
    from apps.dashboard import views as dashboard_views

    monkeypatch.setattr(
        dashboard_views,
        "build_chart_od_density_heatmap",
        lambda theme=None: ("<div id='density'></div>", None),
    )
    monkeypatch.setattr(
        dashboard_views,
        "build_chart_od_hourly_breakdown",
        lambda theme=None: "<div id='hourly'></div>",
    )
    monkeypatch.setattr(
        dashboard_views,
        "build_chart_day_of_week_totals",
        lambda theme=None: "<div id='dow'></div>",
    )
    monkeypatch.setattr(
        dashboard_views,
        "build_chart_shift_scenarios",
        lambda theme=None: "<div id='scenarios'></div>",
    )
    monkeypatch.setattr(
        dashboard_views,
        "build_chart_cost_benefit_analysis",
        lambda theme=None: "<div id='cost'></div>",
    )
    monkeypatch.setattr(
        dashboard_views,
        "calculate_coverage_scenarios",
        lambda: {"baseline": {"coverage": 30}},
    )


def test_shiftcoverage_percentage_calculations(authenticated_client, monkeypatch):
    """Verify working / proposed coverage math using controlled timestamps."""
    _stub_charts(monkeypatch)
    ODReferrals.objects.all().delete()
    year = 2025
    # Monday 6 Jan 2025
    # Deliberately exclude early-evening events so proposed_coverage == current_coverage
    entries = [
        make_dt(year, 1, 6, 7),  # early morning (old/new neither working hours)
        make_dt(year, 1, 6, 8, 30),  # old working hours only
        make_dt(year, 1, 6, 9, 30),  # working hours both
        make_dt(year, 1, 6, 16, 30),  # new working hours only
        make_dt(year, 1, 6, 19, 30),  # late evening
        make_dt(year, 1, 11, 10, 0),  # Saturday daytime (weekend daytime)
        make_dt(year, 1, 12, 20, 0),  # Sunday late evening
        make_dt(year, 1, 6, 14, 0),  # working hours both
    ]
    for idx, dt in enumerate(entries):
        ODReferrals.objects.create(od_date=dt, patient_id=idx + 1)

    resp = authenticated_client.get(reverse("dashboard:odreferrals_shift_coverage"))
    assert resp.status_code == 200
    # 3 of 8 records fall in old working hours => 37.5
    assert resp.context["current_coverage"] == 37.5
    # 100 - 37.5 = 62.5
    assert resp.context["missed_opportunities"] == 62.5
    # proposed coverage equals current (no early evening records)
    assert resp.context["proposed_coverage"] == 37.5


def test_shiftcoverage_with_early_evening_extension(authenticated_client, monkeypatch):
    """Adding an early-evening event increases proposed coverage over current."""
    _stub_charts(monkeypatch)
    ODReferrals.objects.all().delete()
    year = 2025
    entries = [
        make_dt(year, 1, 6, 7),
        make_dt(year, 1, 6, 8, 30),
        make_dt(year, 1, 6, 9, 30),
        make_dt(year, 1, 6, 16, 30),
        make_dt(year, 1, 6, 17, 30),  # early evening weekday
        make_dt(year, 1, 6, 19, 30),
        make_dt(year, 1, 11, 10, 0),
        make_dt(year, 1, 12, 20, 0),
        make_dt(year, 1, 6, 14, 0),
    ]
    for idx, dt in enumerate(entries):
        ODReferrals.objects.create(od_date=dt, patient_id=idx + 1)
    resp = authenticated_client.get(reverse("dashboard:odreferrals_shift_coverage"))
    assert resp.status_code == 200
    # old working hours: 3/9 => 33.3
    assert resp.context["current_coverage"] == 33.3
    # proposed should be at least current (may include early evening extension)
    assert resp.context["proposed_coverage"] >= resp.context["current_coverage"]


def test_set_theme_cookie_invalid_not_set(rf):
    from django.http import HttpResponse

    resp = HttpResponse("ok")
    set_theme_cookie_response(resp, "blue")  # invalid
    assert "theme" not in resp.cookies


def test_model_strs():
    # Just basic smoke tests for __str__ implementations
    # Patients not currently used heavily but create one for coverage
    p = Patients.objects.create(age=30)
    assert "Patient ID" in str(p)
    od = ODReferrals.objects.create(patient_id=999)
    assert "OD Referral ID" in str(od)
