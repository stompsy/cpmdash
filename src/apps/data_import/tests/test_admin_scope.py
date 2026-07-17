import pytest
from django.contrib import admin
from django.contrib.auth import get_user_model
from django.test import RequestFactory

from apps.core.models import Agency, County
from apps.data_import.admin import DataImportBatchAdmin, ProcessingLogAdmin
from apps.data_import.models import DataImportBatch, ProcessingLog

pytestmark = pytest.mark.django_db


def _build_request(user):
    request = RequestFactory().get("/admin/")
    request.user = user
    return request


def test_data_import_admin_querysets_are_scoped_for_non_superusers():
    county = County.objects.create(name="Clallam Tenant", slug="clallam-tenant")
    agency_a = Agency.objects.create(name="Agency A", slug="agency-a", county=county)
    agency_b = Agency.objects.create(name="Agency B", slug="agency-b", county=county)

    User = get_user_model()
    user = User.objects.create_user(
        username="import-admin",
        email="import-admin@example.com",
        password="password123",
        is_staff=True,
        agency=agency_a,
    )

    batch_a = DataImportBatch.objects.create(created_by=user, county=county, agency=agency_a)
    batch_b = DataImportBatch.objects.create(created_by=user, county=county, agency=agency_b)

    ProcessingLog.objects.create(batch=batch_a, content="A log")
    ProcessingLog.objects.create(batch=batch_b, content="B log")

    request = _build_request(user)

    batch_admin = DataImportBatchAdmin(DataImportBatch, admin.site)
    visible_batch_ids = set(batch_admin.get_queryset(request).values_list("id", flat=True))
    assert visible_batch_ids == {batch_a.id}

    log_admin = ProcessingLogAdmin(ProcessingLog, admin.site)
    visible_log_batch_ids = set(log_admin.get_queryset(request).values_list("batch_id", flat=True))
    assert visible_log_batch_ids == {batch_a.id}
