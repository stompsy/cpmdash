from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from django.contrib.messages import get_messages
from django.urls import reverse

from apps.core.models import Agency, County
from apps.data_import.models import DataImportBatch

pytestmark = pytest.mark.django_db


@pytest.fixture()
def tenant_import_setup(client):
    county = County.objects.create(name="Clallam Scope", slug="clallam-scope")
    agency_a = Agency.objects.create(name="Agency Scope A", slug="agency-scope-a", county=county)
    agency_b = Agency.objects.create(name="Agency Scope B", slug="agency-scope-b", county=county)

    User = get_user_model()
    user_a = User.objects.create_user(
        username="import-scope-a",
        email="import-scope-a@example.com",
        password="password123",
        is_staff=True,
        agency=agency_a,
    )
    superuser = User.objects.create_superuser(
        username="import-root",
        email="import-root@example.com",
        password="password123",
    )

    batch_a = DataImportBatch.objects.create(
        created_by=user_a,
        county=county,
        agency=agency_a,
        status=DataImportBatch.Status.REVIEW,
    )
    batch_b = DataImportBatch.objects.create(
        created_by=user_a,
        county=county,
        agency=agency_b,
        status=DataImportBatch.Status.REVIEW,
    )

    return {
        "client": client,
        "user_a": user_a,
        "superuser": superuser,
        "batch_a": batch_a,
        "batch_b": batch_b,
    }


def test_review_view_blocks_cross_agency_batch_access(tenant_import_setup):
    client = tenant_import_setup["client"]
    user_a = tenant_import_setup["user_a"]
    batch_b = tenant_import_setup["batch_b"]

    client.force_login(user_a)
    response = client.get(reverse("data_import:review", args=[batch_b.id]))

    assert response.status_code == 404


def test_batch_list_shows_only_current_agency_batches(tenant_import_setup):
    client = tenant_import_setup["client"]
    user_a = tenant_import_setup["user_a"]
    batch_a = tenant_import_setup["batch_a"]
    batch_b = tenant_import_setup["batch_b"]

    client.force_login(user_a)
    response = client.get(reverse("data_import:batch_list"))

    assert response.status_code == 200
    batch_ids = {batch.id for batch in response.context["batches"]}
    assert batch_a.id in batch_ids
    assert batch_b.id not in batch_ids


def test_commit_view_rejects_missing_tenant_metadata_for_superuser(tenant_import_setup):
    client = tenant_import_setup["client"]
    superuser = tenant_import_setup["superuser"]

    missing_tenant_batch = DataImportBatch.objects.create(
        created_by=superuser,
        status=DataImportBatch.Status.REVIEW,
        county=None,
        agency=None,
    )

    client.force_login(superuser)
    response = client.get(
        reverse("data_import:commit", args=[missing_tenant_batch.id]), follow=True
    )

    assert response.redirect_chain
    messages = [str(m) for m in get_messages(response.wsgi_request)]
    assert any("missing county/agency metadata" in message for message in messages)
