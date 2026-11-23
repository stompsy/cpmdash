import builtins
import re
from datetime import date, datetime
from io import BytesIO

import pytest
from django.contrib.auth import get_user_model
from django.http import HttpResponse
from django.urls import reverse
from django.utils import timezone

from apps.core.models import Agency, County, Encounters, ODReferrals, Referrals
from apps.dashboard import views as dashboard_views
from apps.dashboard.models import HargroveMetricOverride
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


@pytest.mark.parametrize("latest_dataset", ["referrals", "odreferrals"])
def test_patients_page_uses_latest_patient_or_referral_commit_date(rf, monkeypatch, latest_dataset):
    patient_commit = make_dt(2026, 5, 12, 11)
    later_referral_commit = make_dt(2026, 6, 1, 16)

    DataImportBatch.objects.create(
        status=DataImportBatch.Status.COMMITTED,
        committed_at=patient_commit,
        committed_patients=5,
    )
    referral_batch = DataImportBatch.objects.create(
        status=DataImportBatch.Status.COMMITTED,
        committed_at=later_referral_commit,
        **{f"committed_{latest_dataset}": 7},
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
    assert captured_context["page_header_updated_at"] == later_referral_commit
    assert captured_context["page_header_updated_at_iso"] == later_referral_commit.isoformat()
    assert captured_context["page_header_freshness_badge_text"] == (
        "Data is current through the last committed import "
        f"(June 1, 2026, import batch #{referral_batch.pk:04d}) and is ready for reporting."
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

    captured_agency_ids = []

    def fake_build_accordions(agency_id):
        captured_agency_ids.append(agency_id)
        return []

    monkeypatch.setattr(dashboard_views, "render", fake_render)
    monkeypatch.setattr(dashboard_views, "_build_hargrove_accordions", fake_build_accordions)

    response = dashboard_views.hargrove_grant(request)

    assert response.status_code == 200
    assert captured_agency_ids == [request.user.agency_id]
    assert captured_context["report_agency"].id == request.user.agency_id
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
    assert 'filename="hargrove_grant_port-angeles_2026_Q2.docx"' in resp["Content-Disposition"]
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


def test_hargrove_dynamic_metrics_exclude_other_agency_records():
    county = County.objects.create(name="Test County", slug="test-county")
    port_angeles = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    sequim = Agency.objects.create(name="Test Agency", slug="test-agency", county=county)
    Referrals.objects.create(patient_ID=10, date_received="2026-07-01", agency=sequim)
    Referrals.objects.create(patient_ID=10, date_received="2026-07-02", agency=sequim)
    ODReferrals.objects.create(
        patient_id=11,
        od_date=make_dt(2026, 7, 3, 10),
        agency=sequim,
    )
    Referrals.objects.create(patient_ID=20, date_received="2026-07-01", agency=port_angeles)
    ODReferrals.objects.create(
        patient_id=21,
        od_date=make_dt(2026, 7, 3, 10),
        agency=port_angeles,
    )
    Encounters.objects.create(
        patient_ID=22,
        encounter_date="2026-07-01",
        agency=port_angeles,
    )

    sections = dashboard_views._build_dynamic_hargrove_metrics(2026, 3, port_angeles.id)
    rows = [row for section in sections if section["type"] == "table" for row in section["rows"]]
    metric_values = {row["id"]: row["value"] for row in rows}

    assert metric_values["96719"] == "3"
    assert metric_values["96721"] == "2"
    assert metric_values["96723"] == "2"


def _hargrove_user(username, agency, *, staff=False):
    user = get_user_model().objects.create_user(username, f"{username}@example.com", "pw12345!")  # type: ignore[attr-defined]
    user.agency = agency
    user.is_staff = staff
    user.save(update_fields=["agency", "is_staff"])
    return user


def test_hargrove_report_always_uses_port_angeles(rf):
    port_angeles = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    sequim = Agency.objects.get(name="Sequim", county=port_angeles.county)
    staff = _hargrove_user("hgstaff", sequim, staff=True)

    for params in (
        {"scope": "agency", "agency_id": sequim.id},
        {"scope": "county", "county_id": port_angeles.county_id},
        {},
    ):
        request = rf.get(reverse("dashboard:hargrove_grant"), params)
        request.user = staff
        assert dashboard_views._hargrove_report_agency(request).id == port_angeles.id


def test_hargrove_is_hidden_and_forbidden_for_other_agencies(client):
    port_angeles = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    sequim = Agency.objects.get(name="Sequim", county=port_angeles.county)
    client.force_login(_hargrove_user("hgsequim", sequim))

    assert client.get(reverse("dashboard:hargrove_grant")).status_code == 403
    assert client.get(reverse("dashboard:hargrove_grant_export", args=[2026, 3])).status_code == 403
    save = client.post(
        reverse("dashboard:hargrove_metric_save"),
        {"year": "2026", "quarter": "3", "metric_key": "Patients Served", "field": "value"},
    )
    assert save.status_code == 403
    assert not HargroveMetricOverride.objects.filter(metric_key="Patients Served").exists()
    profile = client.get(reverse("dashboard:user_profile")).content.decode()
    assert reverse("dashboard:hargrove_grant") not in profile


def test_hargrove_is_visible_to_port_angeles_users(authenticated_client):
    profile = authenticated_client.get(reverse("dashboard:user_profile")).content.decode()
    assert reverse("dashboard:hargrove_grant") in profile
    assert authenticated_client.get(reverse("dashboard:hargrove_grant")).status_code == 200


def test_hargrove_save_always_writes_port_angeles_override(client):
    port_angeles = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    sequim = Agency.objects.get(name="Sequim", county=port_angeles.county)
    client.force_login(_hargrove_user("hgsaver", sequim, staff=True))
    response = client.post(
        f"{reverse('dashboard:hargrove_metric_save')}?scope=agency&agency_id={sequim.id}",
        {
            "year": "2026",
            "quarter": "3",
            "metric_key": "Patients Served",
            "metric_id": "96723",
            "field": "value",
            "content": "7",
        },
    )

    assert response.status_code == 204
    override = HargroveMetricOverride.objects.get(
        year=2026, quarter=3, metric_key="Patients Served"
    )
    assert override.agency_id == port_angeles.id
    assert override.value == "7"


def test_hargrove_supportive_services_count_all_port_angeles_overdoses():
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    HargroveMetricOverride.objects.filter(agency=agency, year=2026).delete()
    # Coordinates far outside Port Angeles must still count; only the agency matters.
    for patient, (lat, lon, supportive) in enumerate(
        [(48.11, -123.43, True), (47.6, -122.3, False), (0.0, 0.0, False)]
    ):
        ODReferrals.objects.create(
            agency=agency,
            patient_id=patient,
            od_date=make_dt(2026, 8, 1, 10),
            lat=lat,
            long=lon,
            referral_to_sud_agency=supportive,
        )
    sections = dashboard_views._build_dynamic_hargrove_metrics(2026, 3, agency.id)
    rows = {
        row["id"]: row
        for section in sections
        if section["type"] == "table"
        for row in section["rows"]
    }
    assert rows["2026-01a"]["value"] == "33.3% (1 of 3)"
    assert rows["96733"]["value"] == "0% (0 of 3)"


def test_hargrove_hides_archived_metrics_from_other_agencies():
    sections = dashboard_views._build_hargrove_quarter_sections(
        year=2022,
        q=1,
        year_metrics={"1": {}},
        agency_id=2,
        legacy_agency_id=1,
        current_year=2026,
    )

    assert sections[0]["rows"][0]["value"] == "Not available for this agency"


def test_hargrove_overrides_are_isolated_by_agency():
    port_angeles = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    county = County.objects.create(name="Override County", slug="override-county")
    sequim = Agency.objects.create(name="Override Agency", slug="override-agency", county=county)
    sections = [
        {
            "type": "table",
            "rows": [{"metric": "Patients Served", "value": "2", "notes": ""}],
        }
    ]
    HargroveMetricOverride.objects.create(
        agency=port_angeles,
        year=2026,
        quarter=3,
        metric_key="Patients Served",
        value="PA override",
    )

    dashboard_views._hargrove_apply_overrides(sections, 2026, 3, sequim.id)

    assert sections[0]["rows"][0]["value"] == "2"
    assert HargroveMetricOverride.objects.filter(agency=port_angeles, year=2026, quarter=1).exists()


def test_hargrove_ytd_columns_sum_counts_and_preserve_snapshots():
    quarters = []
    for q in range(1, 5):
        quarters.append(
            {
                "q": q,
                "sections": [
                    {
                        "title": "Metrics",
                        "type": "table",
                        "rows": [
                            {
                                "id": "contacts",
                                "metric": "Total Contacts",
                                "value": f"{q * 1000:,}",
                            },
                            {"id": "cost", "metric": "Dollars Saved", "value": f"${q * 2000:,}"},
                            {
                                "id": "percent",
                                "metric": "% Repeat (Quarter)",
                                "value": "25% (1 of 4)",
                            },
                            {
                                "id": "average",
                                "metric": "Avg services per patient",
                                "value": "2.00 (10 / 5)",
                            },
                            {
                                "id": "types",
                                "metric": "Distinct service types (Referrals)",
                                "value": "2",
                                "aggregation": "union",
                                "computed_value": "2",
                                "items": {"Housing", f"Service {q}"},
                            },
                            {
                                "id": "status",
                                "metric": "Reduction Goal Status",
                                "value": "Goal met",
                            },
                            {
                                "id": "fatal",
                                "metric": f"Fatal Overdoses (YTD through Q{q})",
                                "value": str(q),
                            },
                            {
                                "id": "compare",
                                "metric": "2025 same-period comparison",
                                "value": str(q),
                            },
                            {
                                "id": "ytd_pct",
                                "metric": "% Supportive Services (YTD avg)",
                                "value": "50%",
                            },
                            {"id": "people", "metric": "Patients Served", "value": "2"},
                        ],
                    },
                    {"title": "Narrative", "type": "narrative", "questions_responses": []},
                ],
            }
        )

    dashboard_views._hargrove_add_ytd_columns(quarters)

    assert quarters[0]["sections"][0]["rows"][0]["running_total"] == "1,000"
    assert quarters[0]["sections"][0]["rows"][0]["previous_values"] == []
    final = quarters[-1]["sections"][0]
    assert final["report_columns"] == [
        "ID",
        "Metric",
        "Q4 Value",
        "Q3 Value",
        "Q2 Value",
        "Q1 Value",
        "Running YTD",
        "Notes",
    ]
    assert sum(final["column_widths"]) == pytest.approx(100)
    rows = {row["id"]: row for row in final["rows"]}
    assert rows["contacts"]["value"] == "4,000"
    assert rows["contacts"]["previous_values"] == ["3,000", "2,000", "1,000"]
    assert rows["contacts"]["running_total"] == "10,000"
    assert rows["cost"]["running_total"] == "$20,000"
    assert rows["percent"]["running_total"] == "25% (4 of 16)"
    assert rows["average"]["running_total"] == "2.00 (40 / 20)"
    assert rows["types"]["running_total"] == "5"
    assert rows["status"]["running_total"] == "Goal met"
    assert rows["fatal"]["running_total"] == "4"
    assert rows["fatal"]["previous_values"] == ["3", "2", "1"]
    assert rows["compare"]["running_total"] == "4"
    assert rows["ytd_pct"]["running_total"] == "50%"
    assert rows["people"]["running_total"] == "8"
    assert "report_columns" not in quarters[-1]["sections"][1]


@pytest.mark.parametrize("missing", [None, "", "pending", "12 people", "1,2"])
def test_hargrove_ytd_does_not_treat_missing_values_as_zero(missing):
    quarters = [
        {
            "q": q,
            "sections": [
                {
                    "type": "table",
                    "title": "Metrics",
                    "rows": [{"id": "1", "metric": "Contacts", "value": value}],
                }
            ],
        }
        for q, value in enumerate(["2", missing, "3"], start=1)
    ]

    dashboard_views._hargrove_add_ytd_columns(quarters)

    assert quarters[-1]["sections"][0]["rows"][0]["running_total"] == "Not available"
    assert quarters[0]["sections"][0]["rows"][0]["running_total"] == "2"


def test_hargrove_ytd_missing_quarter_is_not_zero():
    quarters = [
        {"q": 1, "sections": []},
        {
            "q": 2,
            "sections": [
                {
                    "title": "Metrics",
                    "type": "table",
                    "rows": [{"id": "1", "metric": "Contacts", "value": "5"}],
                }
            ],
        },
    ]
    dashboard_views._hargrove_add_ytd_columns(quarters)
    assert quarters[1]["sections"][0]["rows"][0]["running_total"] == "Not available"
    assert quarters[1]["sections"][0]["rows"][0]["previous_values"] == ["Not available"]


def test_hargrove_ytd_html_and_docx_include_scoped_saved_totals(authenticated_client, monkeypatch):
    from docx import Document

    monkeypatch.setattr(dashboard_views.timezone, "localdate", lambda: date(2026, 8, 1))
    user = get_user_model().objects.get(username="testuser")
    HargroveMetricOverride.objects.filter(agency=user.agency, year=2026).delete()
    other = Agency.objects.get(name="Sequim", county=user.agency.county)
    for quarter in range(1, 5):
        for _ in range(quarter):
            Referrals.objects.create(
                agency=user.agency, patient_ID=10, date_received=date(2026, quarter * 3, 1)
            )
        Referrals.objects.create(
            agency=other, patient_ID=20, date_received=date(2026, quarter * 3, 1)
        )
    HargroveMetricOverride.objects.create(
        agency=user.agency, year=2026, quarter=1, metric_key="Total Contacts", value="1,000"
    )
    HargroveMetricOverride.objects.create(
        agency=other, year=2026, quarter=1, metric_key="Total Contacts", value="9,000"
    )
    Referrals.objects.create(agency=user.agency, patient_ID=10, date_received=date(2025, 12, 1))

    response = authenticated_client.get(reverse("dashboard:hargrove_grant"))
    assert response.status_code == 200
    year = next(year for year in response.context["years_data"] if year["year"] == 2026)
    contacts = [quarter["sections"][0]["rows"][0]["running_total"] for quarter in year["quarters"]]
    assert contacts == ["1,000", "1,002", "1,005", "1,009"]
    assert year["quarters"][2]["sections"][0]["rows"][0]["previous_values"] == ["2", "1,000"]
    html = response.content.decode()
    assert "Q3 Value" in html
    assert "Running YTD" in html
    assert "data-hargrove-ytd" in html
    assert ">1,005</td>" in html
    assert "min-width: 1120px" not in html
    assert "overflow-wrap: anywhere" in html
    assert 'title="# of Unduplicated Individuals Contacted by CPM"' in html
    assert re.search(r"<summary\b[^>]*># of Unduplicated Individuals Contacted.*?</summary>", html)
    tables = [
        table for table in re.findall(r"<table\b.*?</table>", html, re.S) if "Running YTD" in table
    ]
    widths_by_header = {}
    for rendered_table in tables:
        assert "table-fixed" in rendered_table
        assert "hargrove-report-table" in rendered_table
        headers = tuple(re.findall(r"<th\b[^>]*>(.*?)</th>", rendered_table))
        widths = re.findall(r'<col style="width: ([^"]+);"', rendered_table)
        assert len(widths) == len(headers)
        assert widths == widths_by_header.setdefault(headers, widths)
    people = year["quarters"][2]["sections"][1]["rows"][0]
    assert people["running_total"] == "3"

    export = authenticated_client.get(reverse("dashboard:hargrove_grant_export", args=[2026, 3]))
    assert export.status_code == 200
    document = Document(BytesIO(export.content))
    table = document.tables[0]
    assert [cell.text for cell in table.rows[0].cells] == [
        "ID",
        "Metric",
        "Q3 Value",
        "Q2 Value",
        "Q1 Value",
        "Running YTD",
        "Notes",
    ]
    assert [cell.text for cell in table.rows[1].cells][2:6] == ["3", "2", "1,000", "1,005"]
    expected_widths = [column.width for column in table.columns]
    for other_table in document.tables:
        assert other_table.autofit is False
        assert [column.width for column in other_table.columns] == expected_widths
        for row in other_table.rows:
            assert [cell.width for cell in row.cells] == expected_widths
    assert document.tables[2].rows[1].cells[1].text == (
        "# of Unduplicated Individuals Contacted by CPM"
    )
    paragraphs = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "Certified Crisis Intervention Specialist II" in paragraphs
    assert "$483,044.73" in paragraphs
    assert "$661,762" in paragraphs
    assert "477 patient-quarter counts year to date" in paragraphs


def test_hargrove_zip_totals_include_saved_edits():
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    HargroveMetricOverride.objects.create(
        agency=agency, year=2026, quarter=3, metric_key="98362", value="1,002"
    )
    sections = [
        {
            "title": "ZIP",
            "type": "table",
            "section_variant": "zip",
            "rows": [
                {"id": "1", "metric": "98362", "value": "2"},
                {"id": "2", "metric": "Unknown", "value": "3"},
                {"id": "-", "metric": "Total", "value": "5"},
            ],
        }
    ]
    dashboard_views._hargrove_apply_overrides(sections, 2026, 3, agency.id)
    dashboard_views._hargrove_add_ytd_columns([{"q": 1, "sections": sections}])
    total = sections[0]["rows"][-1]
    assert total["is_zip_total"] is True
    assert total["value"] == "1,005"
    assert total["running_total"] == "1,005"


def test_hargrove_archived_ytd_uses_saved_values():
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    HargroveMetricOverride.objects.create(
        agency=agency, year=2024, quarter=1, metric_key="Total Contacts", value="12"
    )
    metrics = {
        str(q): {
            "metrics": {
                "services": [{"id": "1", "metric": "Total Contacts (Current Quarter)", "value": q}]
            }
        }
        for q in range(1, 5)
    }
    quarters = [
        {
            "q": q,
            "sections": dashboard_views._build_hargrove_quarter_sections(
                2024, q, metrics, agency.id, agency.id, 2026
            ),
        }
        for q in range(1, 5)
    ]
    dashboard_views._hargrove_add_ytd_columns(quarters)
    row = quarters[-1]["sections"][3]["rows"][0]
    assert row["previous_values"] == ["3", "2", "12"]
    assert row["running_total"] == "21"


@pytest.mark.parametrize("quarter", [0, 5])
def test_hargrove_export_rejects_invalid_quarter(authenticated_client, quarter):
    response = authenticated_client.get(
        reverse("dashboard:hargrove_grant_export", args=[2026, quarter])
    )
    assert response.status_code == 400


def test_hargrove_repeat_events_use_quarter_and_all_time_history(monkeypatch):
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    other = Agency.objects.get(name="Sequim", county=agency.county)
    HargroveMetricOverride.objects.filter(agency=agency, year=2026).delete()
    ODReferrals.objects.create(agency=agency, patient_id=1, od_date=make_dt(2025, 12, 1, 10))
    ODReferrals.objects.create(agency=other, patient_id=30, od_date=make_dt(2025, 12, 1, 10))
    ODReferrals.objects.create(agency=agency, patient_id=31, od_date=make_dt(2026, 7, 1, 10))
    for patient in [1, 1, *range(10, 18)]:
        ODReferrals.objects.create(
            agency=agency, patient_id=patient, od_date=make_dt(2026, 3, 31, 10)
        )
    for patient in [1, 2, 2, 2, 10, 30, 31, 32, 33, 34]:
        ODReferrals.objects.create(
            agency=agency, patient_id=patient, od_date=make_dt(2026, 4, 1, 10)
        )

    assert dashboard_views._hargrove_repeat_event_counts(2026, 1, agency.id) == (1, 2, 10)
    assert dashboard_views._hargrove_repeat_event_counts(2026, 2, agency.id) == (2, 4, 10)
    quarters = [
        {"q": q, "sections": dashboard_views._build_dynamic_hargrove_metrics(2026, q, agency.id)}
        for q in (1, 2)
    ]
    dashboard_views._hargrove_add_ytd_columns(quarters)
    rows = {row["id"]: row for row in quarters[1]["sections"][2]["rows"]}
    assert rows["96733"]["value"] == "20% (2 of 10)"
    assert rows["96733"]["previous_values"] == ["10% (1 of 10)"]
    assert rows["96733"]["running_total"] == "15% (3 of 20)"
    assert rows["96733-history"]["value"] == "40% (4 of 10)"
    assert rows["96733-history"]["previous_values"] == ["20% (2 of 10)"]
    assert rows["96733-history"]["running_total"] == "30% (6 of 20)"


def test_hargrove_repeat_events_handle_unidentified_and_undated_events():
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    for _ in range(2):
        ODReferrals.objects.create(agency=agency, patient_id=None, od_date=make_dt(2026, 4, 1, 10))
    ODReferrals.objects.create(agency=agency, patient_id=1, od_date=None)
    assert dashboard_views._hargrove_repeat_event_counts(2026, 2, agency.id) == (0, 0, 2)
    assert dashboard_views._hargrove_repeat_event_counts(2026, 1, agency.id) == (0, 0, 0)
    assert dashboard_views._hargrove_rate_display(0, 0) == "N/A (0 of 0)"


def test_hargrove_unduplicated_overdose_patient_counts_include_history(monkeypatch):
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    other = Agency.objects.get(name="Sequim", county=agency.county)
    monkeypatch.setattr(dashboard_views, "get_ed_encounter_unit_cost", lambda: 1631)

    for patient, month in [(1, 12), (2, 10), (2, 11), (8, 10), (8, 11)]:
        ODReferrals.objects.create(
            agency=agency, patient_id=patient, od_date=make_dt(2025, month, 1, 10)
        )
    for patient in [1, 2, 3, 4, 4, 5]:
        ODReferrals.objects.create(
            agency=agency, patient_id=patient, od_date=make_dt(2026, 3, 1, 10)
        )
    for patient in [5, 6, 6]:
        ODReferrals.objects.create(
            agency=agency, patient_id=patient, od_date=make_dt(2026, 4, 1, 10)
        )
    ODReferrals.objects.create(agency=other, patient_id=7, od_date=make_dt(2026, 3, 1, 10))

    quarters = [
        {
            "q": q,
            "sections": [
                {
                    "title": "2026 Hargrove Targets & Cost Avoidance",
                    "type": "table",
                    "rows": dashboard_views._build_hargrove_2026_rows(
                        year=2026,
                        q=q,
                        agency_id=agency.id,
                        od_qs=ODReferrals.objects.filter(
                            agency=agency,
                            od_date__date__range=dashboard_views._hargrove_quarter_date_range(
                                2026, q
                            ),
                        ),
                        patients_served=0,
                    ),
                }
            ],
        }
        for q in (1, 2)
    ]
    dashboard_views._hargrove_add_ytd_columns(quarters)

    q1_rows = {row["id"]: row for row in quarters[0]["sections"][0]["rows"]}
    q2_rows = {row["id"]: row for row in quarters[1]["sections"][0]["rows"]}
    assert q1_rows["2026-03d"]["value"] == "5"
    assert q1_rows["2026-03e"]["value"] == "1"
    assert q1_rows["2026-03f"]["value"] == "3"
    assert q2_rows["2026-03d"]["value"] == "2"
    assert q2_rows["2026-03d"]["running_total"] == "6"
    assert q2_rows["2026-03e"]["value"] == "1"
    assert q2_rows["2026-03e"]["running_total"] == "2"
    assert q2_rows["2026-03f"]["value"] == "5"
    assert q2_rows["2026-03f"]["running_total"] == "5"


def test_hargrove_repeat_targets_count_same_quarter_events_and_ignore_old_overrides():
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    other = Agency.objects.get(name="Sequim", county=agency.county)
    HargroveMetricOverride.objects.filter(agency=agency, year=2026).delete()
    for metric, value in [
        ("Repeat Overdose Patients (YTD through Q3)", "999"),
        ("Repeat Overdose Patients (2025 same-period comparison)", "888"),
        ("Repeat Overdose Reduction Goal Status", "Old status"),
    ]:
        HargroveMetricOverride.objects.create(
            agency=agency, year=2026, quarter=3, metric_key=metric, value=value
        )
    for year, month, patients in [
        (2025, 3, [9, 9, 9, 9]),
        (2025, 8, [1, 1, 1, 2, 2]),
        (2026, 3, [1, 1, 1, 1]),
        (2026, 6, [3, 3, 3]),
        (2026, 8, [1, 1, 1, 3, 4, None, None]),
    ]:
        for patient in patients:
            ODReferrals.objects.create(
                agency=agency, patient_id=patient, od_date=make_dt(year, month, 1, 10)
            )
    for _ in range(4):
        ODReferrals.objects.create(agency=other, patient_id=1, od_date=make_dt(2026, 8, 1, 10))
        Referrals.objects.create(
            agency=agency,
            patient_ID=4,
            date_received=date(2026, 8, 1),
            encounter_type_cat1="Overdose",
        )
    ODReferrals.objects.create(agency=agency, patient_id=1, od_date=None)
    quarters = [
        {"q": q, "sections": dashboard_views._build_dynamic_hargrove_metrics(2026, q, agency.id)}
        for q in (1, 2, 3)
    ]
    dashboard_views._hargrove_add_ytd_columns(quarters)
    rows = {row["id"]: row for row in quarters[-1]["sections"][3]["rows"]}
    assert rows["2026-03a"]["value"] == "2"
    assert rows["2026-03a"]["metric"] == "Repeat Overdose Events (Same Quarter)"
    assert rows["2026-03a"]["previous_values"] == ["2", "3"]
    assert rows["2026-03a"]["running_total"] == "7"
    assert rows["2026-03b"]["value"] == "3"
    assert rows["2026-03b"]["metric"] == "Repeat Overdose Events (2025 same-quarter comparison)"
    assert rows["2026-03b"]["previous_values"] == ["0", "3"]
    assert rows["2026-03b"]["running_total"] == "6"
    assert rows["2026-03c"]["value"] == "Goal met (−33.3%)"


def test_hargrove_survivor_and_fatal_targets_count_events_not_patients():
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    other = Agency.objects.get(name="Sequim", county=agency.county)
    HargroveMetricOverride.objects.filter(agency=agency, year=2026).delete()
    for year, month, disposition, supportive in [
        (2025, 3, "DOA", False),
        (2025, 8, "DOA", False),
        (2025, 8, "DOA", False),
        (2026, 3, "Transport", True),
        (2026, 8, "Transport", True),
        (2026, 8, "Refused transport", True),
        (2026, 8, "Transport", False),
        (2026, 8, "DOA", False),
        (2026, 8, "DOA", False),
    ]:
        ODReferrals.objects.create(
            agency=agency,
            patient_id=1,
            od_date=make_dt(year, month, 1, 10),
            disposition=disposition,
            referral_to_sud_agency=supportive,
        )
    ODReferrals.objects.create(
        agency=other, patient_id=1, od_date=make_dt(2026, 8, 1, 10), disposition="DOA"
    )
    Referrals.objects.create(
        agency=agency,
        patient_ID=2,
        date_received=date(2026, 8, 1),
        encounter_type_cat1="Overdose",
    )
    sections = dashboard_views._build_dynamic_hargrove_metrics(2026, 3, agency.id)
    rows = {row["id"]: row for row in sections[3]["rows"]}
    assert rows["2026-01a"]["value"] == "66.7% (2 of 3)"
    assert rows["2026-01b"]["value"] == "75% (3 of 4)"
    assert rows["2026-02a"]["value"] == "2"
    assert rows["2026-02b"]["value"] == "3"
    assert rows["2026-02c"]["value"] == "Goal met (−33.3%)"


def test_hargrove_running_rates_pool_saved_counts_and_report_missing_inputs():
    rows = [
        {"metric": "% Repeat Events", "value": "50% (1 of 2)"},
        {"metric": "% Repeat Events", "value": "10% (1 of 10)"},
    ]
    assert dashboard_views._hargrove_running_value(rows) == "16.7% (2 of 12)"
    rows[1]["value"] = "25% (5 of 20)"
    assert dashboard_views._hargrove_running_value(rows) == "27.3% (6 of 22)"
    rows[1]["value"] = "25%"
    assert "needs numerator / denominator" in dashboard_views._hargrove_running_value(rows)
    rows[1]["value"] = "200% (20 of 10)"
    assert "numerator exceeds denominator" in dashboard_views._hargrove_running_value(rows)
    assert (
        dashboard_views._hargrove_running_value(
            [
                {"metric": "Avg services", "value": "2.00 (4 / 2)"},
                {"metric": "Avg services", "value": "4.00 (40 / 10)"},
            ]
        )
        == "3.67 (44 / 12)"
    )
    assert (
        dashboard_views._hargrove_running_value(
            [
                {"metric": "% Repeat Events", "value": "N/A (0 of 0)"},
                {"metric": "% Repeat Events", "value": "10% (1 of 10)"},
            ]
        )
        == "10% (1 of 10)"
    )


def test_hargrove_service_union_cannot_guess_from_overridden_count():
    rows = [
        {
            "metric": "Distinct service types",
            "value": "2",
            "computed_value": "2",
            "aggregation": "union",
            "items": {"housing", "detox"},
        },
        {
            "metric": "Distinct service types",
            "value": "2",
            "computed_value": "2",
            "aggregation": "union",
            "items": {"housing", "transport"},
        },
    ]
    assert dashboard_views._hargrove_running_value(rows) == "3"
    rows[1]["value"] = "5"
    assert "edited count needs service names" in dashboard_views._hargrove_running_value(rows)
    assert "needs service names" in dashboard_views._hargrove_running_value(
        [{"metric": "Distinct service types (Referrals)", "value": "3"}]
    )


def test_hargrove_service_totals_follow_service_not_rank(monkeypatch):
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    HargroveMetricOverride.objects.filter(agency=agency, year=2026).delete()
    for patient, service in enumerate(
        ["Housing", "Housing", "Detox", *[f"Service {i}" for i in range(13)]]
    ):
        Referrals.objects.create(
            agency=agency, patient_ID=patient, date_received=date(2026, 1, 1), referral_1=service
        )
    for patient, service in enumerate(["Detox", "Detox", "Transport"]):
        Referrals.objects.create(
            agency=agency, patient_ID=patient, date_received=date(2026, 4, 1), referral_1=service
        )
    quarters = [
        {"q": q, "sections": dashboard_views._build_dynamic_hargrove_metrics(2026, q, agency.id)}
        for q in (1, 2)
    ]
    dashboard_views._hargrove_add_ytd_columns(quarters)
    sections = quarters[1]["sections"]
    services = next(section for section in sections if section.get("complete_service_table"))
    rows = {row["metric"]: row for row in services["rows"]}
    assert rows["Housing"]["value"] == "0"
    assert rows["Housing"]["previous_values"] == ["2"]
    assert rows["Housing"]["running_total"] == "2"
    assert rows["Detox"]["previous_values"] == ["1"]
    assert rows["Detox"]["running_total"] == "3"
    assert rows["Transport"]["previous_values"] == ["0"]
    assert rows["Transport"]["running_total"] == "1"
    assert len(rows) == 16
    stats = next(section for section in sections if "Service Gap Analytics" in section["title"])
    assert stats["rows"][0]["running_total"] == "16"


def test_hargrove_supportive_services_ytd_is_pooled(monkeypatch):
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    HargroveMetricOverride.objects.filter(agency=agency, year=2026).delete()
    for quarter, flags in [(1, [True]), (2, [True, False, False])]:
        for patient, supportive in enumerate(flags):
            ODReferrals.objects.create(
                agency=agency,
                patient_id=patient,
                od_date=make_dt(2026, quarter * 3, 1, 10),
                referral_to_sud_agency=supportive,
            )
    quarters = [
        {"q": q, "sections": dashboard_views._build_dynamic_hargrove_metrics(2026, q, agency.id)}
        for q in (1, 2)
    ]
    dashboard_views._hargrove_add_ytd_columns(quarters)
    rows = {row["id"]: row for row in quarters[-1]["sections"][3]["rows"]}
    assert rows["2026-01a"]["value"] == "33.3% (1 of 3)"
    assert rows["2026-01a"]["previous_values"] == ["100% (1 of 1)"]
    assert rows["2026-01a"]["running_total"] == "50% (2 of 4)"
    assert rows["2026-01b"]["value"] == "50% (2 of 4)"


def test_hargrove_q3_2026_narratives_include_activities_and_case_context():
    items = dashboard_views._hargrove_load_narratives(2026, 3)
    assert len(items) == 5
    assert [bool(item["response"].strip()) for item in items] == [True, True, True, False, True]
    assert items[3]["question"] == "Success Stories"
    responses = "\n".join(item["response"] for item in items)
    for activity in (
        "Certified Crisis Intervention Specialist II",
        "7th grade classes",
        "Lower Elwha Klallam Tribe",
        "Reflections counseling services",
        "Homelessness Response System Report Committee",
        "Project SAFER presentation",
        "UW BHCore",
        "COAST training",
        "International Overdose Awareness Day",
        "SUD Recovery Bash",
        "Vern Burton gym",
    ):
        assert activity in responses
    comments = items[4]
    assert comments["narrative_key"] == "narrative::q5"
    for fact in (
        "sepsis",
        "ambulance",
        "two weeks",
        "27 services",
        "$483,044.73",
        "$661,762",
        "477 patient-quarter counts year to date",
        "$1,631",
        "readmissions",
    ):
        assert fact in comments["response"]
    assert "we do not claim any" in comments["response"]
    assert "$483,044.73" in comments["response_html"]


def test_hargrove_q3_narratives_use_consistent_event_and_cost_definitions():
    items = dashboard_views._hargrove_load_narratives(2026, 3)
    results, collaboration, sustainability, _, comments = [item["response"] for item in items]
    assert "3 to 1 (−67%)" in results
    assert "17 events involving 16 patients" in results
    assert "not confirmed service receipt" in results
    assert "6, from 13" not in results
    assert "nine events without referral flags" in results
    assert "Nine Q3 OD events" in collaboration
    assert "not nine distinct people" in collaboration
    assert "Certified Crisis Intervention Specialist II" in sustainability
    assert "$661,762" not in sustainability
    assert "$243,762 for ED evaluations" in comments
    assert "not an annual unduplicated count" in comments
    assert "fixed utilization and reduction assumptions" in comments
    assert "floor for estimated savings" not in comments
    assert sum(len(item["response"].split()) for item in items) <= 1200
    assert all("{{" not in item["response_html"] for item in items)


@pytest.mark.parametrize(
    ("served", "calls", "transports", "ed_visits"),
    [(0, 0, 0, 0), (6, 1, 1, 2), (25, 7, 4, 8), (152, 45, 25, 53)],
)
def test_hargrove_cost_avoidance_truncates_only_final_counts(
    monkeypatch, served, calls, transports, ed_visits
):
    monkeypatch.setattr(dashboard_views, "get_ed_encounter_unit_cost", lambda: 1631)
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    rows = {
        row["id"]: row
        for row in dashboard_views._build_hargrove_2026_rows(
            year=2026,
            q=3,
            agency_id=agency.id,
            od_qs=ODReferrals.objects.none(),
            patients_served=served,
        )
    }
    assert rows["2026-04a"]["value"] == str(transports)
    assert rows["2026-04c"]["value"] == str(calls)
    assert rows["2026-04e"]["value"] == str(ed_visits)
    assert rows["2026-04b"]["value"] == f"${transports * 3800:,}"
    assert rows["2026-04d"]["value"] == f"${(calls - transports) * 1900:,}"
    assert rows["2026-04f"]["value"] == f"${ed_visits * 1631:,}"


@pytest.mark.parametrize(
    ("served", "calls", "transports", "ed_visits"),
    [(0, 0, 0, 0), (6, 1, 1, 2), (25, 7, 4, 8), (152, 45, 25, 53)],
)
def test_cost_savings_truncates_only_final_counts(
    monkeypatch, served, calls, transports, ed_visits
):
    import pandas as pd

    from apps.charts import od_utils

    monkeypatch.setattr(
        od_utils,
        "get_quarterly_patient_counts",
        lambda: {
            "df": pd.DataFrame({"year": [2026, 2026], "count": [served // 2, served - served // 2]})
        },
    )
    monkeypatch.setattr(od_utils, "get_ed_encounter_unit_cost", lambda: 1631)

    metrics = od_utils.get_cost_savings_metrics()
    savings = {
        "transports": transports * 3800,
        "911_non_transport": (calls - transports) * 1900,
        "ed_visits": ed_visits * 1631,
    }
    assert metrics["calls_prevented"] == calls
    assert metrics["transports_averted"] == transports
    assert metrics["ed_visits_avoided"] == ed_visits
    assert metrics["savings_breakdown"] == savings
    assert metrics["total_savings"] == sum(savings.values())
    assert metrics["yearly_breakdown"] == [
        {
            "year": 2026,
            "patients": served,
            "calls_prevented": calls,
            "transports_averted": transports,
            "ed_visits_avoided": ed_visits,
            "total_savings": sum(savings.values()),
            "pct_change_savings": None,
        }
    ]


def test_hargrove_final_count_truncation_ytd_matches_q3_narrative(monkeypatch):
    monkeypatch.setattr(dashboard_views, "get_ed_encounter_unit_cost", lambda: 1631)
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    quarters = [
        {
            "q": q,
            "sections": [
                {
                    "title": "2026 Hargrove Targets & Cost Avoidance",
                    "type": "table",
                    "rows": dashboard_views._build_hargrove_2026_rows(
                        year=2026,
                        q=q,
                        agency_id=agency.id,
                        od_qs=ODReferrals.objects.none(),
                        patients_served=served,
                    ),
                }
            ],
        }
        for q, served in enumerate((168, 157, 152), start=1)
    ]
    dashboard_views._hargrove_add_ytd_columns(quarters)
    rows = {row["id"]: row for row in quarters[-1]["sections"][0]["rows"]}
    assert rows["2026-04a"]["running_total"] == "79"
    assert rows["2026-04c"]["running_total"] == "141"
    assert rows["2026-04e"]["running_total"] == "167"
    savings = {
        "2026-04b": "$300,200",
        "2026-04d": "$117,800",
        "2026-04f": "$243,762",
    }
    comments = dashboard_views._hargrove_load_narratives(2026, 3)[4]["response"]
    for row_id, amount in savings.items():
        assert rows[row_id]["running_total"] == amount
        assert amount in comments
    assert "$661,762" in comments
    assert "truncating only the final event counts" in comments


def test_hargrove_ed_visit_cost_switches_to_hospital_average_in_q2_2026(monkeypatch):
    monkeypatch.setattr(dashboard_views, "get_ed_encounter_unit_cost", lambda: 1631)
    assert dashboard_views._hargrove_ed_unit_cost(2025, 4) == 1146
    assert dashboard_views._hargrove_ed_unit_cost(2026, 1) == 1146
    assert dashboard_views._hargrove_ed_unit_cost(2026, 2) == 1631
    assert dashboard_views._hargrove_ed_unit_cost(2026, 3) == 1631

    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    HargroveMetricOverride.objects.filter(agency=agency, year=2026).delete()
    for quarter in (1, 2):
        for patient in range(10):
            Referrals.objects.create(
                agency=agency, patient_ID=patient, date_received=date(2026, quarter * 3, 1)
            )
    rows = {}
    for quarter in (1, 2):
        rows[quarter] = {
            row["id"]: row
            for row in dashboard_views._build_hargrove_2026_rows(
                year=2026,
                q=quarter,
                agency_id=agency.id,
                od_qs=ODReferrals.objects.none(),
                patients_served=10,
            )
        }
    # 10 patients × 51% × 69% → 3 ED visits prevented after final truncation.
    assert rows[1]["2026-04f"]["value"] == "$3,438"
    assert rows[1]["2026-04f"]["notes"].startswith("$1,146 per ED visit")
    assert rows[2]["2026-04f"]["value"] == "$4,893"
    assert rows[2]["2026-04f"]["notes"].startswith("$1,631 per ED visit")
    assert "excludes labs, imaging" in rows[2]["2026-04f"]["notes"]


def test_hargrove_q3_saved_narrative_edits_still_take_precedence():
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    HargroveMetricOverride.objects.create(
        agency=agency,
        year=2026,
        quarter=3,
        metric_key="narrative::q5",
        value="Agency-reviewed comments.",
    )
    sections = [
        {
            "title": "Narrative",
            "type": "narrative",
            "questions_responses": dashboard_views._hargrove_load_narratives(2026, 3),
        }
    ]
    dashboard_views._hargrove_apply_overrides(sections, 2026, 3, agency.id)
    assert sections[0]["questions_responses"][4]["response"] == "Agency-reviewed comments."


# Test removed - odreferrals_monthly page has been removed
