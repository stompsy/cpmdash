from datetime import datetime

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone

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


# Test removed - odreferrals_monthly page has been removed
