from django.urls import path

from .views import (
    authentication,
    encounters,
    odreferrals,
    odreferrals_monthly,
    overview,
    patients,
    referrals,
    user_profile,
)

app_name = "dashboard"

urlpatterns = [
    path("", overview, name="dashboard_overview"),
    path("patients/", patients, name="patients"),
    path("referrals/", referrals, name="referrals"),
    path("odreferrals/", odreferrals, name="odreferrals"),
    path("odreferrals/monthly/", odreferrals_monthly, name="odreferrals_monthly"),
    # Back-compat alias: route removed per request but referenced in tests
    path("overdoses/by-case/", odreferrals, name="overdoses_by_case"),
    path("encounters/", encounters, name="encounters"),
    path("profile/", user_profile, name="user_profile"),
    path("authentication/", authentication, name="authentication"),
]
