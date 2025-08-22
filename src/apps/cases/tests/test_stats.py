from datetime import datetime
from typing import Any

import pytest
from django.urls import reverse
from django.utils import timezone

from apps.core.models import ODReferrals, Patients
from utils.theme import set_theme_cookie_response

pytestmark = pytest.mark.django_db


def make_dt(year: int, month: int, day: int, hour: int, minute: int = 0):
    tz = timezone.get_current_timezone()
    return timezone.make_aware(datetime(year, month, day, hour, minute), tz)


def _stub_charts(monkeypatch: Any) -> None:
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
        views, "calculate_coverage_scenarios", lambda: {"baseline": {"coverage": 30}}
    )


def test_shiftcoverage_percentage_calculations(client, monkeypatch):
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

    resp = client.get(reverse("cases:shiftcoverage"))
    assert resp.status_code == 200
    # 3 of 8 records fall in old working hours => 37.5
    assert resp.context["current_coverage"] == 37.5
    # 100 - 37.5 = 62.5
    assert resp.context["missed_opportunities"] == 62.5
    # proposed coverage equals current (no early evening records)
    assert resp.context["proposed_coverage"] == 37.5


def test_shiftcoverage_with_early_evening_extension(client, monkeypatch):
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
    resp = client.get(reverse("cases:shiftcoverage"))
    assert resp.status_code == 200
    # old working hours: 3/9 => 33.3
    assert resp.context["current_coverage"] == 33.3
    # proposed should be at least current (may include early evening extension)
    assert resp.context["proposed_coverage"] >= resp.context["current_coverage"]


def test_repeatods_statistics_content(client, monkeypatch):
    """Ensure repeat overdose year-by-year stats computed correctly."""
    from apps.cases import views

    monkeypatch.setattr(
        views, "build_chart_repeats_scatter", lambda theme=None: "<div id='repeats'></div>"
    )
    ODReferrals.objects.all().delete()
    # Year 2024: patient 1 (3 overdoses), patient 2 (1) => repeat_overdoses=3, total=4 => 75.0%
    for _ in range(3):
        ODReferrals.objects.create(od_date=make_dt(2024, 5, 1, 10), patient_id=1)
    ODReferrals.objects.create(od_date=make_dt(2024, 6, 1, 12), patient_id=2)
    # Year 2025: patient 3 (2), patient 4 (2) => repeat_overdoses=4, total=4 => 100.0%
    for pid in (3, 4):
        ODReferrals.objects.create(od_date=make_dt(2025, 1, 6, 9), patient_id=pid)
        ODReferrals.objects.create(od_date=make_dt(2025, 1, 7, 9), patient_id=pid)

    resp = client.get(reverse("cases:repeatods"))
    assert resp.status_code == 200
    stats = resp.context["repeat_stats_by_year"]
    by_year = {s["year"]: s for s in stats}
    assert by_year[2024]["repeat_overdoses"] == 3
    assert by_year[2024]["repeat_patients"] == 1
    assert by_year[2024]["percent_repeat"] == 75.0
    assert by_year[2025]["repeat_overdoses"] == 4
    assert by_year[2025]["repeat_patients"] == 2
    assert by_year[2025]["percent_repeat"] == 100.0


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
