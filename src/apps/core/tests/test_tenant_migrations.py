from importlib import import_module

import pytest
from django.apps import apps as django_apps
from django.contrib.auth import get_user_model
from django.db import IntegrityError

from apps.core.models import Agency, County, Encounters, ODReferrals, Patients, Referrals

pytestmark = pytest.mark.django_db


def test_default_tenants_are_seeded() -> None:
    county = County.objects.get(slug="clallam-county")
    agencies = set(Agency.objects.filter(county=county).values_list("slug", flat=True))

    assert county.name == "Clallam County"
    assert {"port-angeles", "sequim"}.issubset(agencies)


def test_agency_slug_is_unique_per_county() -> None:
    county = County.objects.get(slug="clallam-county")

    with pytest.raises(IntegrityError):
        Agency.objects.create(name="Duplicate Sequim", slug="sequim", county=county)


def test_same_agency_slug_allowed_in_different_county() -> None:
    other_county = County.objects.create(name="Jefferson County", slug="jefferson-county")

    agency = Agency.objects.create(name="Sequim", slug="sequim", county=other_county)

    assert agency.county_id == other_county.id


def test_backfill_default_agency_updates_null_tenant_rows_and_users() -> None:
    migration = import_module("apps.core.migrations.0009_backfill_default_agency")
    backfill_default_agency = migration.backfill_default_agency

    county = County.objects.get(slug="clallam-county")
    port_angeles = Agency.objects.get(county=county, slug="port-angeles")
    sequim = Agency.objects.get(county=county, slug="sequim")

    patient_null = Patients.objects.create(id=901001, insurance="A")
    patient_existing = Patients.objects.create(id=901002, insurance="B", agency=sequim)
    referral_null = Referrals.objects.create(ID=902001)
    referral_existing = Referrals.objects.create(ID=902002, agency=sequim)
    encounter_null = Encounters.objects.create(ID=903001)
    encounter_existing = Encounters.objects.create(ID=903002, agency=sequim)
    od_null = ODReferrals.objects.create(ID=904001)
    od_existing = ODReferrals.objects.create(ID=904002, agency=sequim)

    User = get_user_model()
    user_null = User.objects.create_user(
        username="backfill-null-user",
        email="backfill-null@example.com",
        password="password123",
    )
    user_existing = User.objects.create_user(
        username="backfill-existing-user",
        email="backfill-existing@example.com",
        password="password123",
        agency=sequim,
    )

    backfill_default_agency(django_apps, None)

    patient_null.refresh_from_db()
    patient_existing.refresh_from_db()
    referral_null.refresh_from_db()
    referral_existing.refresh_from_db()
    encounter_null.refresh_from_db()
    encounter_existing.refresh_from_db()
    od_null.refresh_from_db()
    od_existing.refresh_from_db()
    user_null.refresh_from_db()
    user_existing.refresh_from_db()

    assert patient_null.agency_id == port_angeles.id
    assert referral_null.agency_id == port_angeles.id
    assert encounter_null.agency_id == port_angeles.id
    assert od_null.agency_id == port_angeles.id
    assert user_null.agency_id == port_angeles.id

    assert patient_existing.agency_id == sequim.id
    assert referral_existing.agency_id == sequim.id
    assert encounter_existing.agency_id == sequim.id
    assert od_existing.agency_id == sequim.id
    assert user_existing.agency_id == sequim.id


def test_non_null_agency_defaults_apply_to_new_core_rows_and_users() -> None:
    county = County.objects.get(slug="clallam-county")
    port_angeles = Agency.objects.get(county=county, slug="port-angeles")

    patient = Patients.objects.create(id=905001, insurance="Defaulted")
    referral = Referrals.objects.create(ID=905002)
    encounter = Encounters.objects.create(ID=905003)
    od = ODReferrals.objects.create(ID=905004)

    User = get_user_model()
    user = User.objects.create_user(
        username="default-agency-user",
        email="default-agency@example.com",
        password="password123",
    )

    assert patient.agency_id == port_angeles.id
    assert referral.agency_id == port_angeles.id
    assert encounter.agency_id == port_angeles.id
    assert od.agency_id == port_angeles.id
    assert user.agency_id == port_angeles.id
