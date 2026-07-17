from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from django.contrib.messages import get_messages
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from apps.core.models import Agency, County
from apps.data_import.models import DataImportBatch

pytestmark = pytest.mark.django_db


@pytest.fixture()
def upload_setup(client):
    county_a = County.objects.create(name="Clallam Upload", slug="clallam-upload")
    county_b = County.objects.create(name="Jefferson Upload", slug="jefferson-upload")
    agency_a = Agency.objects.create(name="Port Angeles Upload", slug="pa-upload", county=county_a)
    agency_b = Agency.objects.create(name="Port Townsend Upload", slug="pt-upload", county=county_b)

    User = get_user_model()
    user = User.objects.create_user(
        username="upload-user",
        email="upload-user@example.com",
        password="password123",
        is_staff=True,
        agency=agency_a,
    )
    superuser = User.objects.create_superuser(
        username="upload-superuser",
        email="upload-superuser@example.com",
        password="password123",
    )
    client.force_login(user)

    return {
        "client": client,
        "county_a": county_a,
        "county_b": county_b,
        "agency_a": agency_a,
        "agency_b": agency_b,
        "user": user,
        "superuser": superuser,
    }


def _patients_csv() -> SimpleUploadedFile:
    content = b"id,age,insurance\n1,45,Medicaid\n"
    return SimpleUploadedFile("patients.csv", content, content_type="text/csv")


def test_upload_requires_county_and_agency(upload_setup):
    client = upload_setup["client"]

    response = client.post(reverse("data_import:upload"), {"file": _patients_csv()})

    assert response.status_code == 200
    assert "form" in response.context
    assert response.context["form"].errors
    assert DataImportBatch.objects.count() == 0


def test_upload_rejects_agency_from_different_county(upload_setup):
    client = upload_setup["client"]
    county_a = upload_setup["county_a"]
    agency_b = upload_setup["agency_b"]

    response = client.post(
        reverse("data_import:upload"),
        {
            "county": county_a.id,
            "agency": agency_b.id,
            "file": _patients_csv(),
        },
    )

    assert response.status_code == 200
    assert "agency" in response.context["form"].errors
    assert DataImportBatch.objects.count() == 0


def test_upload_creates_batch_with_selected_tenant(upload_setup):
    client = upload_setup["client"]
    county_a = upload_setup["county_a"]
    agency_a = upload_setup["agency_a"]

    response = client.post(
        reverse("data_import:upload"),
        {
            "county": county_a.id,
            "agency": agency_a.id,
            "file": _patients_csv(),
        },
    )

    assert response.status_code == 302
    batch = DataImportBatch.objects.get()
    assert batch.county_id == county_a.id
    assert batch.agency_id == agency_a.id


def test_upload_to_batch_rejects_batch_missing_tenant_metadata(upload_setup):
    client = upload_setup["client"]
    superuser = upload_setup["superuser"]

    batch = DataImportBatch.objects.create(
        created_by=superuser,
        county=None,
        agency=None,
        status=DataImportBatch.Status.UPLOADING,
    )

    client.force_login(superuser)

    response = client.get(reverse("data_import:upload_to_batch", args=[batch.id]), follow=True)

    assert response.redirect_chain
    messages = [str(m) for m in get_messages(response.wsgi_request)]
    assert any("missing county/agency metadata" in message for message in messages)
