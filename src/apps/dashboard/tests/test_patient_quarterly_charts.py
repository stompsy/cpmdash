from datetime import date, datetime

import pandas as pd
import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone

from apps.charts.patients import patient_field_charts as patient_charts
from apps.core.models import Agency, County, ODReferrals, Patients, Referrals
from apps.data_import.models import DataImportBatch
from utils.plotly import get_theme_colors

pytestmark = pytest.mark.django_db

CHART_FIELD = "unique_referred_patients_quarterly"


def od_date(year, month, day, hour=12):
    return timezone.make_aware(datetime(year, month, day, hour))


@pytest.fixture()
def agencies():
    county = County.objects.create(name="Chart County", slug="chart-county")
    other_county = County.objects.create(name="Other Chart County", slug="other-chart-county")
    return [
        Agency.objects.create(name="Chart A", slug="chart-a", county=county),
        Agency.objects.create(name="Chart B", slug="chart-b", county=county),
        Agency.objects.create(name="Chart C", slug="chart-c", county=other_county),
    ]


@pytest.fixture()
def captured_figures(monkeypatch):
    figures = []
    original_plot = patient_charts.plot

    def capture(fig, **kwargs):
        figures.append(fig)
        return original_plot(fig, **kwargs)

    monkeypatch.setattr(patient_charts, "plot", capture)
    return figures


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_distinct_referral_patients_per_quarter(agencies, captured_figures, theme):
    agency = agencies[0]
    patient = Patients.objects.create(agency=agency, created_date=date(2020, 1, 1))
    Referrals.objects.bulk_create(
        [
            Referrals(agency=agency, patient_ID=patient.pk, date_received=date(2025, 12, 31)),
            Referrals(agency=agency, patient_ID=patient.pk, date_received=date(2026, 1, 1)),
            Referrals(agency=agency, patient_ID=patient.pk, date_received=date(2026, 3, 31)),
            Referrals(agency=agency, patient_ID=99999, date_received=date(2026, 2, 1)),
            Referrals(agency=agency, patient_ID=patient.pk, date_received=date(2026, 7, 1)),
            Referrals(agency=agency, patient_ID=None, date_received=date(2024, 1, 1)),
            Referrals(agency=agency, patient_ID=patient.pk, date_received=None),
        ]
    )

    charts = patient_charts.build_patients_field_charts(
        theme, fields=[CHART_FIELD], scope_filters={"agency_id": agency.pk}
    )

    assert set(charts) == {CHART_FIELD}
    assert "Plotly.newPlot" in charts[CHART_FIELD]
    assert "cdn.plot.ly" not in charts[CHART_FIELD]
    fig = captured_figures[0]
    assert list(fig.data[0].x) == ["2025 Q4", "2026 Q1", "2026 Q2", "2026 Q3"]
    assert list(fig.data[0].y) == [1, 2, 0, 1]
    assert list(fig.data[0].text) == ["1", "2", "0", "1"]
    assert fig.layout.yaxis.title.text == "Unique Patients"
    assert fig.layout.yaxis.range is None
    assert fig.layout.font.color == get_theme_colors(theme)["font_color"]


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("agency", 1),
        ("county", 2),
        ("multi-agency", 2),
        ("all", 3),
    ],
)
@pytest.mark.parametrize("source", ["regular", "od", "both"])
def test_quarterly_referral_patients_respect_scope(
    agencies, captured_figures, scope, expected, source
):
    if source in {"regular", "both"}:
        Referrals.objects.bulk_create(
            [
                Referrals(agency=agency, patient_ID=100 + i, date_received=date(2026, 4, 1))
                for i, agency in enumerate(agencies)
            ]
        )
    if source in {"od", "both"}:
        ODReferrals.objects.bulk_create(
            [
                ODReferrals(agency=agency, patient_id=100 + i, od_date=od_date(2026, 4, 1))
                for i, agency in enumerate(agencies)
            ]
        )
    filters = {
        "agency": {"agency_id": agencies[0].pk},
        "county": {"agency__county_id": agencies[0].county_id},
        "multi-agency": {"agency_id__in": [agencies[0].pk, agencies[1].pk]},
        "all": None,
    }[scope]

    patient_charts._build_unique_referred_patients_quarterly_bar("dark", scope_filters=filters)

    assert list(captured_figures[0].data[0].x) == ["2026 Q2"]
    assert list(captured_figures[0].data[0].y) == [expected]


@pytest.mark.parametrize("invalid_referrals", [False, True])
def test_empty_quarterly_referral_chart(agencies, invalid_referrals):
    if invalid_referrals:
        Referrals.objects.create(agency=agencies[0], patient_ID=123)
        Referrals.objects.create(agency=agencies[0], date_received=date(2026, 1, 1))
        ODReferrals.objects.create(agency=agencies[0], patient_id=123)
        ODReferrals.objects.create(agency=agencies[0], od_date=od_date(2026, 1, 1))

    assert patient_charts.build_patients_field_charts("dark", fields=[CHART_FIELD]) == {}


def test_combined_sources_deduplicate_patients_within_each_quarter(agencies, captured_figures):
    agency = agencies[0]
    Referrals.objects.bulk_create(
        [
            Referrals(agency=agency, patient_ID=111, date_received=date(2026, 1, 1)),
            Referrals(agency=agency, patient_ID=111, date_received=date(2026, 3, 31)),
            Referrals(agency=agency, patient_ID=222, date_received=date(2026, 2, 1)),
        ]
    )
    ODReferrals.objects.bulk_create(
        [
            ODReferrals(agency=agency, patient_id=111, od_date=od_date(2026, 3, 31, 23)),
            ODReferrals(agency=agency, patient_id=333, od_date=od_date(2026, 1, 1)),
            ODReferrals(agency=agency, patient_id=333, od_date=od_date(2026, 2, 1)),
            ODReferrals(agency=agency, patient_id=111, od_date=od_date(2026, 4, 1, 0)),
            ODReferrals(agency=agency, patient_id=444, od_date=od_date(2026, 10, 1)),
        ]
    )

    charts = patient_charts.build_patients_field_charts(
        "dark", fields=[CHART_FIELD], scope_filters={"agency_id": agency.pk}
    )

    assert set(charts) == {CHART_FIELD}
    assert list(captured_figures[0].data[0].x) == ["2026 Q1", "2026 Q2", "2026 Q3", "2026 Q4"]
    assert list(captured_figures[0].data[0].y) == [3, 1, 0, 1]


def test_quarterly_chart_errors_are_not_silenced(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Chart generation failed")

    monkeypatch.setattr(patient_charts, "_build_unique_referred_patients_quarterly_bar", fail)

    with pytest.raises(RuntimeError, match="Chart generation failed"):
        patient_charts.build_patients_field_charts("dark", fields=[CHART_FIELD])


def test_unrequested_quarterly_chart_is_not_built(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Unrequested referral chart was built")

    monkeypatch.setattr(patient_charts, "_build_unique_referred_patients_quarterly_bar", fail)

    assert patient_charts.build_patients_field_charts("dark", fields=[]) == {}


def test_existing_enrollment_chart_still_uses_creation_quarter(agencies, captured_figures):
    source_df = pd.DataFrame({"created_date": [date(2026, 1, 1), date(2026, 4, 1)]})
    patient_charts._build_quarterly_patient_bar(
        "dark", source_df=source_df, scope_filters={"agency_id": agencies[0].pk}
    )

    assert list(captured_figures[0].data[0].y) == [1, 1]


def test_patients_page_orders_full_width_quarterly_cards(client, agencies):
    user = get_user_model().objects.create_user(username="quarter-page", agency=agencies[0])
    client.force_login(user)

    response = client.get(reverse("dashboard:patients"))

    assert response.status_code == 200
    cards = response.context["chart_cards"]
    assert [card["field"] for card in cards[:2]] == ["patient_counts_quarterly", CHART_FIELD]
    assert [card["col_span"] for card in cards[:2]] == [2, 2]
    assert cards[0]["label"] == "New Patients by Quarter (All-Time Unique)"
    assert cards[1]["label"] == "Unique Referred Patients by Quarter"
    assert reverse("dashboard:patients_chart_fragment", args=[CHART_FIELD]) in response.text


@pytest.mark.parametrize("has_referrals", [False, True])
@pytest.mark.parametrize("source", ["regular", "od"])
def test_quarterly_fragment_renders_scoped_referrals_and_freshness(
    client, agencies, captured_figures, has_referrals, source
):
    agency = agencies[0]
    user = get_user_model().objects.create_user(username="quarter-fragment", agency=agency)
    client.force_login(user)
    committed_at = timezone.now()
    DataImportBatch.objects.create(
        agency=agency,
        county=agency.county,
        status=DataImportBatch.Status.COMMITTED,
        committed_at=committed_at,
        committed_referrals=1 if source == "regular" else 0,
        committed_odreferrals=1 if source == "od" else 0,
    )
    if has_referrals:
        if source == "regular":
            Referrals.objects.create(agency=agency, patient_ID=111, date_received=date(2026, 1, 1))
        else:
            ODReferrals.objects.create(agency=agency, patient_id=111, od_date=od_date(2026, 1, 1))
    Referrals.objects.create(agency=agencies[1], patient_ID=222, date_received=date(2026, 4, 1))
    ODReferrals.objects.create(agency=agencies[1], patient_id=333, od_date=od_date(2026, 7, 1))

    response = client.get(reverse("dashboard:patients_chart_fragment", args=[CHART_FIELD]))

    assert response.status_code == 200
    item = response.context["item"]
    assert item["has_chart"] == has_referrals
    assert item["label"] == "Unique Referred Patients by Quarter"
    assert item["updated_at"] == committed_at
    assert "same patient can count again" in item["rationale"]
    assert "<!DOCTYPE" not in response.text
    if has_referrals:
        assert list(captured_figures[0].data[0].x) == ["2026 Q1"]
        assert list(captured_figures[0].data[0].y) == [1]
        assert "Plotly.newPlot" in response.text
    else:
        assert not captured_figures
        assert "Not enough data yet" in response.text
