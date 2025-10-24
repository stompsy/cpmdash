from django.urls import path

from .views import (
    age_chart_variations_demo,
    authentication,
    encounters,
    encounters_chart_fragment,
    encounters_insights,
    encounters_insights_fragment,
    odreferrals,
    odreferrals_chart_fragment,
    odreferrals_hotspots,
    odreferrals_insights,
    odreferrals_insights_fragment,
    odreferrals_monthly,
    odreferrals_repeat_overdoses,
    odreferrals_shift_coverage,
    overview,
    patients,
    patients_chart_fragment,
    referrals,
    referrals_chart_fragment,
    referrals_insights,
    referrals_insights_fragment,
    top_engaged_patients,
    user_profile,
)

app_name = "dashboard"

urlpatterns = [
    path("", overview, name="dashboard_overview"),
    path("patients/", patients, name="patients"),
    path("patients/charts/<slug:field>/", patients_chart_fragment, name="patients_chart_fragment"),
    path("patients/top-engaged/", top_engaged_patients, name="top_engaged_patients"),
    path("referrals/", referrals, name="referrals"),
    path(
        "referrals/charts/<slug:field>/", referrals_chart_fragment, name="referrals_chart_fragment"
    ),
    path("referrals/insights/", referrals_insights, name="referrals_insights"),
    path(
        "referrals/insights/fragment/",
        referrals_insights_fragment,
        name="referrals_insights_fragment",
    ),
    path("odreferrals/", odreferrals, name="odreferrals"),
    path(
        "odreferrals/charts/<slug:field>/",
        odreferrals_chart_fragment,
        name="odreferrals_chart_fragment",
    ),
    path("odreferrals/insights/", odreferrals_insights, name="odreferrals_insights"),
    path(
        "odreferrals/insights/fragment/",
        odreferrals_insights_fragment,
        name="odreferrals_insights_fragment",
    ),
    path(
        "odreferrals/shift-coverage/",
        odreferrals_shift_coverage,
        name="odreferrals_shift_coverage",
    ),
    path(
        "odreferrals/hotspots/",
        odreferrals_hotspots,
        name="odreferrals_hotspots",
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
    path(
        "encounters/charts/<slug:field>/",
        encounters_chart_fragment,
        name="encounters_chart_fragment",
    ),
    path("encounters/insights/", encounters_insights, name="encounters_insights"),
    path(
        "encounters/insights/fragment/",
        encounters_insights_fragment,
        name="encounters_insights_fragment",
    ),
    path("profile/", user_profile, name="user_profile"),
    path("authentication/", authentication, name="authentication"),
    path("demo/age-chart-variations/", age_chart_variations_demo, name="age_chart_variations_demo"),
]
