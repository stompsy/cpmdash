from datetime import datetime

import pytest
from django.urls import reverse
from django.utils import timezone

from apps.core.models import ODReferrals

pytestmark = pytest.mark.django_db


def test_basic_dashboard_pages(client):
    names = [
        "dashboard:dashboard_overview",
        "dashboard:patients",
        "dashboard:referrals",
        "dashboard:odreferrals",
        "dashboard:overdoses_by_case",
        "dashboard:encounters",
        "dashboard:user_profile",
        "dashboard:authentication",
    ]
    for name in names:
        resp = client.get(reverse(name))
        assert resp.status_code == 200, name


def make_dt(year, month, day, hour):
    tz = timezone.get_current_timezone()
    return timezone.make_aware(datetime(year, month, day, hour), tz)


def test_odreferrals_monthly_statistics(client, monkeypatch):
    from apps.dashboard import views

    monkeypatch.setattr(
        views, "build_chart_od_hist_monthly", lambda theme=None: "<div id='monthly'></div>"
    )
    monkeypatch.setattr(
        views, "build_chart_all_cases_scatter", lambda theme=None: "<div id='cases'></div>"
    )
    ODReferrals.objects.all().delete()
    dates = [
        make_dt(2025, 1, 1, 7),
        make_dt(2025, 1, 1, 8),
        make_dt(2025, 1, 2, 9),
        make_dt(2025, 1, 3, 10),
        make_dt(2025, 1, 4, 16),
        make_dt(2025, 1, 5, 20),
    ]
    patient_ids = [1, 1, 1, 2, 3, 3]
    dispositions = ["Stable", "Stable", "DOA", "Stable", "CPR attempted", "Stable"]
    referral_sources = ["EMS", "EMS", "Other", "Community", "EMS", "Community"]
    referral_to_sud = [True, False, False, True, False, True]

    for pid, dt, disp, rsrc, sud in zip(
        patient_ids, dates, dispositions, referral_sources, referral_to_sud, strict=False
    ):
        ODReferrals.objects.create(
            patient_id=pid,
            od_date=dt,
            disposition=disp,
            referral_source=rsrc,
            referral_to_sud_agency=sud,
            living_situation="House",
            suspected_drug="Fentanyl",
        )

    resp = client.get(reverse("dashboard:odreferrals_monthly"))
    assert resp.status_code == 200
    ctx = resp.context
    assert ctx["total_overdoses"] == 6
    assert ctx["fatal_overdoses"] == 2
    assert ctx["repeat_overdoses"] == 5
    assert ctx["repeat_patients"] == 2
    assert ctx["percent_repeat"] == 83.3
    assert ctx["referral_success_rate"] == 60.0
    assert set(ctx["density_stats"].keys()) == {
        "early_morning",
        "working_hours",
        "weekend_daytime",
        "early_evening",
        "weekend_early_evening",
        "late_evening",
    }
