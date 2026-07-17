import pytest
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.urls import reverse

from apps.core.models import Agency, County, ODReferrals, Patients, Referrals
from apps.dashboard import views as dashboard_views

pytestmark = pytest.mark.django_db


@pytest.fixture()
def tenant_setup(client):
    county = County.objects.create(name="Clallam County X", slug="clallam-county-x")
    agency_a = Agency.objects.create(name="Port Angeles X", slug="port-angeles-x", county=county)
    agency_b = Agency.objects.create(name="Sequim X", slug="sequim-x", county=county)

    User = get_user_model()
    user = User.objects.create_user(
        username="tenant-user",
        email="tenant@example.com",
        password="password123",
        agency=agency_a,
    )
    client.force_login(user)

    return {
        "client": client,
        "user": user,
        "county": county,
        "agency_a": agency_a,
        "agency_b": agency_b,
    }


def _patch_fragment_render(monkeypatch):
    def fake_render(_request, _template_name, context):
        return JsonResponse(context["item"])

    monkeypatch.setattr(dashboard_views, "render", fake_render)


def test_patients_fragment_agency_scope_honors_selected_agency(tenant_setup, monkeypatch):
    agency_a = tenant_setup["agency_a"]
    agency_b = tenant_setup["agency_b"]
    client = tenant_setup["client"]

    Patients.objects.create(id=1001, insurance="A_ONLY", agency=agency_a)
    Patients.objects.create(id=1002, insurance="B_ONLY", agency=agency_b)

    _patch_fragment_render(monkeypatch)
    monkeypatch.setattr(
        dashboard_views,
        "build_patients_field_charts",
        lambda **_kwargs: {"insurance": "<div>chart</div>"},
    )
    monkeypatch.setattr(dashboard_views, "_build_patient_quick_stats", lambda *_args, **_kwargs: {})

    def fake_patient_insights(df_all, _age_insights, _scope_filters=None):
        labels = sorted({str(v) for v in df_all["insurance"].dropna().tolist()})
        return {"insurance": [{"label": "Visible", "value": "|".join(labels)}]}

    monkeypatch.setattr(dashboard_views, "_build_patients_chart_insights", fake_patient_insights)

    response = client.get(
        reverse("dashboard:patients_chart_fragment", args=["insurance"]),
        {"scope": "agency", "agency_id": agency_b.id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["insights"][0]["value"] == "B_ONLY"


def test_patients_fragment_county_scope_includes_county_agencies(tenant_setup, monkeypatch):
    agency_a = tenant_setup["agency_a"]
    agency_b = tenant_setup["agency_b"]
    county = tenant_setup["county"]
    client = tenant_setup["client"]

    Patients.objects.create(id=1011, insurance="A_ONLY", agency=agency_a)
    Patients.objects.create(id=1012, insurance="B_ONLY", agency=agency_b)

    _patch_fragment_render(monkeypatch)
    monkeypatch.setattr(
        dashboard_views,
        "build_patients_field_charts",
        lambda **_kwargs: {"insurance": "<div>chart</div>"},
    )
    monkeypatch.setattr(dashboard_views, "_build_patient_quick_stats", lambda *_args, **_kwargs: {})

    def fake_patient_insights(df_all, _age_insights, _scope_filters=None):
        labels = sorted({str(v) for v in df_all["insurance"].dropna().tolist()})
        return {"insurance": [{"label": "Visible", "value": "|".join(labels)}]}

    monkeypatch.setattr(dashboard_views, "_build_patients_chart_insights", fake_patient_insights)

    response = client.get(
        reverse("dashboard:patients_chart_fragment", args=["insurance"]),
        {"scope": "county", "county_id": county.id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["insights"][0]["value"] == "A_ONLY|B_ONLY"


def test_referrals_and_od_fragments_respect_agency_scope(tenant_setup, monkeypatch):
    agency_a = tenant_setup["agency_a"]
    agency_b = tenant_setup["agency_b"]
    client = tenant_setup["client"]

    Referrals.objects.create(ID=2011, referral_agency="A_REF", agency=agency_a)
    Referrals.objects.create(ID=2012, referral_agency="B_REF", agency=agency_b)
    ODReferrals.objects.create(ID=3011, suspected_drug="A_SRC", agency=agency_a)
    ODReferrals.objects.create(ID=3012, suspected_drug="B_SRC", agency=agency_b)

    _patch_fragment_render(monkeypatch)

    monkeypatch.setattr(
        dashboard_views,
        "build_referrals_field_charts",
        lambda **_kwargs: {"referral_agency": "<div>chart</div>"},
    )
    monkeypatch.setattr(
        dashboard_views,
        "_build_referrals_chart_insights",
        lambda df_all: {
            "referral_agency": [
                {
                    "label": "Visible",
                    "value": "|".join(
                        sorted({str(v) for v in df_all["referral_agency"].dropna().tolist()})
                    ),
                }
            ]
        },
    )
    monkeypatch.setattr(
        dashboard_views, "_build_referral_quick_stats", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        dashboard_views,
        "_build_encounters_quarterly_quick_stats",
        lambda *_args, **_kwargs: {},
    )

    referral_response = client.get(
        reverse("dashboard:referrals_chart_fragment", args=["referral_agency"]),
        {"scope": "agency", "agency_id": agency_b.id},
    )
    assert referral_response.status_code == 200
    referral_payload = referral_response.json()
    assert referral_payload["insights"][0]["value"] == "B_REF"

    monkeypatch.setattr(
        dashboard_views,
        "build_odreferrals_field_charts",
        lambda **_kwargs: {"suspected_drug": "<div>chart</div>"},
    )
    monkeypatch.setattr(
        dashboard_views,
        "_build_odreferrals_chart_insights",
        lambda df_all: {
            "suspected_drug": [
                {
                    "label": "Visible",
                    "value": "|".join(
                        sorted({str(v) for v in df_all["suspected_drug"].dropna().tolist()})
                    ),
                }
            ]
        },
    )
    monkeypatch.setattr(
        dashboard_views,
        "_build_odreferrals_quick_stats",
        lambda *_args, **_kwargs: {},
    )

    od_response = client.get(
        reverse("dashboard:odreferrals_chart_fragment", args=["suspected_drug"]),
        {"scope": "agency", "agency_id": agency_b.id},
    )
    assert od_response.status_code == 200
    od_payload = od_response.json()
    assert od_payload["insights"][0]["value"] == "B_SRC"
