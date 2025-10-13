from django.urls import path

from .views import (
    authentication,
    encounters,
    encounters_insights,
    odreferrals,
    odreferrals_insights,
    odreferrals_monthly,
    odreferrals_repeat_overdoses,
    odreferrals_shift_coverage,
    overview,
    patients,
    patients_insights,
    referrals,
    referrals_insights,
    user_profile,
)

app_name = "dashboard"

urlpatterns = [
    path("", overview, name="dashboard_overview"),
    path("patients/", patients, name="patients"),
    path("patients/insights/", patients_insights, name="patients_insights"),
    path("referrals/", referrals, name="referrals"),
    path("referrals/insights/", referrals_insights, name="referrals_insights"),
    path("odreferrals/", odreferrals, name="odreferrals"),
    path("odreferrals/insights/", odreferrals_insights, name="odreferrals_insights"),
    path(
        "odreferrals/shift-coverage/",
        odreferrals_shift_coverage,
        name="odreferrals_shift_coverage",
    ),
    path(
        "odreferrals/repeat-overdoses/",
        odreferrals_repeat_overdoses,
        name="odreferrals_repeat_overdoses",
    ),
    path("odreferrals/monthly/", odreferrals_monthly, name="odreferrals_monthly"),
    # Back-compat alias: route removed per request but referenced in tests
    path("overdoses/by-case/", odreferrals, name="overdoses_by_case"),
    path("encounters/", encounters, name="encounters"),
    path("encounters/insights/", encounters_insights, name="encounters_insights"),
    path("profile/", user_profile, name="user_profile"),
    path("authentication/", authentication, name="authentication"),
]
