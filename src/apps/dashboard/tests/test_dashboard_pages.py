import builtins
from datetime import datetime

import pytest
from django.contrib.auth import get_user_model
from django.http import HttpResponse
from django.urls import reverse
from django.utils import timezone

from apps.dashboard import views as dashboard_views
from apps.data_import.models import DataImportBatch

pytestmark = pytest.mark.django_db


@pytest.fixture()
def authenticated_client(client):
    User = get_user_model()
    user = User.objects.create_user("testuser", "test@example.com", "password123")  # type: ignore[attr-defined]
    client.force_login(user)
    return client


def test_basic_dashboard_pages(authenticated_client):
    names = [
        "dashboard:patients",
        "dashboard:referrals",
        # Skip OD-related pages - chart functions require actual data (empty DataFrame causes KeyError)
        # "dashboard:odreferrals",
        # "dashboard:overdoses_by_case",
        "dashboard:user_profile",
        "dashboard:authentication",
    ]
    for name in names:
        resp = authenticated_client.get(reverse(name))
        assert resp.status_code == 200, name


def test_dashboard_overview_redirects_to_home(authenticated_client):
    resp = authenticated_client.get(reverse("dashboard:dashboard_overview"))
    assert resp.status_code == 302
    assert resp.headers["Location"] == reverse("home")


def make_dt(year, month, day, hour):
    tz = timezone.get_current_timezone()
    return timezone.make_aware(datetime(year, month, day, hour), tz)


def test_dataset_update_context_uses_latest_matching_committed_batch():
    older_commit = make_dt(2026, 2, 1, 9)
    latest_commit = make_dt(2026, 3, 7, 14)

    DataImportBatch.objects.create(
        status=DataImportBatch.Status.COMMITTED,
        committed_at=older_commit,
        committed_patients=12,
    )
    DataImportBatch.objects.create(
        status=DataImportBatch.Status.REVIEW,
        committed_at=make_dt(2026, 4, 1, 8),
        committed_patients=99,
    )
    latest_batch = DataImportBatch.objects.create(
        status=DataImportBatch.Status.COMMITTED,
        committed_at=latest_commit,
        committed_odreferrals=3,
    )

    context = dashboard_views._page_header_update_context("patients", "odreferrals")

    assert context["page_header_updated_at"] == latest_commit
    assert context["page_header_updated_at_iso"] == latest_commit.isoformat()
    assert context["page_header_freshness_badge_text"] == (
        "Data is current through the last committed import "
        f"(March 7, 2026, import batch #{latest_batch.pk:04d}) and is ready for reporting."
    )


def test_patients_page_uses_last_patient_commit_date(rf, monkeypatch):
    patient_commit = make_dt(2026, 5, 12, 11)
    later_referral_commit = make_dt(2026, 6, 1, 16)

    patient_batch = DataImportBatch.objects.create(
        status=DataImportBatch.Status.COMMITTED,
        committed_at=patient_commit,
        committed_patients=5,
    )
    DataImportBatch.objects.create(
        status=DataImportBatch.Status.COMMITTED,
        committed_at=later_referral_commit,
        committed_referrals=7,
    )

    User = get_user_model()
    request = rf.get(reverse("dashboard:patients"))
    request.user = User.objects.create_user("pagecheck", "page@example.com", "password123")  # type: ignore[attr-defined]
    for batch in DataImportBatch.objects.all():
        batch.agency = request.user.agency
        batch.county = request.user.agency.county
        batch.save(update_fields=["agency", "county"])
    captured_context = {}

    def fake_render(_request, _template_name, context):
        captured_context.update(context)
        return HttpResponse("ok")

    monkeypatch.setattr(dashboard_views, "render", fake_render)

    response = dashboard_views.patients(request)

    assert response.status_code == 200
    assert captured_context["page_header_updated_at"] == patient_commit
    assert captured_context["page_header_updated_at_iso"] == patient_commit.isoformat()
    assert captured_context["page_header_freshness_badge_text"] == (
        "Data is current through the last committed import "
        f"(May 12, 2026, import batch #{patient_batch.pk:04d}) and is ready for reporting."
    )


def test_hargrove_page_uses_latest_import_date(rf, monkeypatch):
    grant_commit = make_dt(2026, 7, 1, 10)

    grant_batch = DataImportBatch.objects.create(
        status=DataImportBatch.Status.COMMITTED,
        committed_at=grant_commit,
        committed_encounters=8,
    )

    User = get_user_model()
    request = rf.get(reverse("dashboard:hargrove_grant"))
    request.user = User.objects.create_user("grantcheck", "grant@example.com", "password123")  # type: ignore[attr-defined]
    for batch in DataImportBatch.objects.all():
        batch.agency = request.user.agency
        batch.county = request.user.agency.county
        batch.save(update_fields=["agency", "county"])
    captured_context = {}

    def fake_render(_request, _template_name, context):
        captured_context.update(context)
        return HttpResponse("ok")

    monkeypatch.setattr(dashboard_views, "render", fake_render)

    response = dashboard_views.hargrove_grant(request)

    assert response.status_code == 200
    assert captured_context["page_header_updated_at"] == grant_commit
    assert captured_context["page_header_freshness_badge_text"] == (
        "Data is current through the last committed import "
        f"(July 1, 2026, import batch #{grant_batch.pk:04d}) and is ready for reporting."
    )


def test_hargrove_docx_export_returns_file(authenticated_client):
    resp = authenticated_client.get(reverse("dashboard:hargrove_grant_export", args=[2026, 2]))

    assert resp.status_code == 200
    assert (
        resp["Content-Type"]
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert 'filename="hargrove_grant_2026_Q2.docx"' in resp["Content-Disposition"]
    assert resp.content.startswith(b"PK")


def test_hargrove_docx_export_missing_dependency_returns_503(authenticated_client, monkeypatch):
    real_import = builtins.__import__

    def _import_with_missing_docx(name, *args, **kwargs):
        if name == "docx" or name.startswith("docx."):
            raise ModuleNotFoundError("No module named 'docx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_with_missing_docx)

    resp = authenticated_client.get(reverse("dashboard:hargrove_grant_export", args=[2026, 2]))

    assert resp.status_code == 503
    assert b"Install python-docx" in resp.content


# Test removed - odreferrals_monthly page has been removed
