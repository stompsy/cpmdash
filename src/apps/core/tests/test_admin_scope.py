import pytest
from django.contrib import admin
from django.contrib.auth import get_user_model
from django.test import RequestFactory

from apps.core.admin import AgencyAdmin, CountyAdmin, PatientsAdmin
from apps.core.models import Agency, County, Patients

pytestmark = pytest.mark.django_db


def _build_request(user):
    request = RequestFactory().get("/admin/")
    request.user = user
    return request


def test_core_admin_querysets_are_scoped_for_non_superusers():
    county_a = County.objects.create(name="Clallam A", slug="clallam-a")
    county_b = County.objects.create(name="Jefferson B", slug="jefferson-b")
    agency_a = Agency.objects.create(name="Port Angeles A", slug="pa-a", county=county_a)
    agency_b = Agency.objects.create(name="Port Townsend B", slug="pt-b", county=county_b)

    Patients.objects.create(id=1001, insurance="A", agency=agency_a)
    Patients.objects.create(id=1002, insurance="B", agency=agency_b)

    User = get_user_model()
    user = User.objects.create_user(
        username="agency-admin",
        email="agency-admin@example.com",
        password="password123",
        is_staff=True,
        agency=agency_a,
    )

    request = _build_request(user)

    patients_admin = PatientsAdmin(Patients, admin.site)
    visible_patient_ids = set(patients_admin.get_queryset(request).values_list("id", flat=True))
    assert visible_patient_ids == {1001}

    agency_admin = AgencyAdmin(Agency, admin.site)
    visible_agency_ids = set(agency_admin.get_queryset(request).values_list("id", flat=True))
    assert visible_agency_ids == {agency_a.id}

    county_admin = CountyAdmin(County, admin.site)
    visible_county_ids = set(county_admin.get_queryset(request).values_list("id", flat=True))
    assert visible_county_ids == {county_a.id}
