from datetime import datetime

import pytest
from django.utils import timezone

from apps.core.models import ODReferrals, Patients
from utils.theme import set_theme_cookie_response

pytestmark = pytest.mark.django_db


@pytest.fixture()
def authenticated_client(client, django_user_model):
    user = django_user_model.objects.create_user("testuser", "test@example.com", "password123")
    client.force_login(user)
    return client


def make_dt(year: int, month: int, day: int, hour: int, minute: int = 0):
    tz = timezone.get_current_timezone()
    return timezone.make_aware(datetime(year, month, day, hour, minute), tz)


def test_set_theme_cookie_invalid_not_set(rf):
    from django.http import HttpResponse

    resp = HttpResponse("ok")
    set_theme_cookie_response(resp, "blue")  # invalid
    assert "theme" not in resp.cookies


def test_model_strs():
    # Just basic smoke tests for __str__ implementations
    # Patients not currently used heavily but create one for coverage
    p = Patients.objects.create(age=30)
    assert "Patient ID" in str(p)
    od = ODReferrals.objects.create(patient_id=999)
    assert "OD Referral ID" in str(od)
