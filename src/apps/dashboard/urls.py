from django.urls import path

from .views import (
    age_chart_variations_demo,
    authentication,
    od_cpm_disposition_detail,
    od_cpm_notification_detail,
    od_cpr_administered_detail,
    od_disposition_detail,
    od_engagement_location_detail,
    od_living_situation_detail,
    od_narcan_detail,
    od_police_ita_detail,
    od_referral_delay_detail,
    od_referral_source_detail,
    od_scene_responders_detail,
    od_sud_referral_detail,
    od_suspected_drug_detail,
    od_transport_detail,
    od_weekday_detail,
    odreferrals,
    odreferrals_chart_fragment,
    odreferrals_hotspots,
    odreferrals_insights,
    odreferrals_insights_fragment,
    odreferrals_shift_coverage,
    overview,
    patients,
    patients_chart_fragment,
    referral_types_table,
    referrals,
    referrals_chart_fragment,
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
    path("referrals/types-table/", referral_types_table, name="referral_types_table"),
    path("odreferrals/", odreferrals, name="odreferrals"),
    path(
        "odreferrals/charts/<slug:field>/",
        odreferrals_chart_fragment,
        name="odreferrals_chart_fragment",
    ),
    path("odreferrals/weekday-detail/", od_weekday_detail, name="od_weekday_detail"),
    path(
        "odreferrals/referral-source-detail/",
        od_referral_source_detail,
        name="od_referral_source_detail",
    ),
    path(
        "odreferrals/suspected-drug-detail/",
        od_suspected_drug_detail,
        name="od_suspected_drug_detail",
    ),
    path(
        "odreferrals/cpm-disposition-detail/",
        od_cpm_disposition_detail,
        name="od_cpm_disposition_detail",
    ),
    path(
        "odreferrals/living-situation-detail/",
        od_living_situation_detail,
        name="od_living_situation_detail",
    ),
    path(
        "odreferrals/engagement-location-detail/",
        od_engagement_location_detail,
        name="od_engagement_location_detail",
    ),
    path(
        "odreferrals/sud-referral-detail/",
        od_sud_referral_detail,
        name="od_sud_referral_detail",
    ),
    path(
        "odreferrals/narcan-detail/",
        od_narcan_detail,
        name="od_narcan_detail",
    ),
    path(
        "odreferrals/referral-delay-detail/",
        od_referral_delay_detail,
        name="od_referral_delay_detail",
    ),
    path(
        "odreferrals/cpm-notification-detail/",
        od_cpm_notification_detail,
        name="od_cpm_notification_detail",
    ),
    path(
        "odreferrals/scene-responders-detail/",
        od_scene_responders_detail,
        name="od_scene_responders_detail",
    ),
    path(
        "odreferrals/cpr-administered-detail/",
        od_cpr_administered_detail,
        name="od_cpr_administered_detail",
    ),
    path(
        "odreferrals/police-ita-detail/",
        od_police_ita_detail,
        name="od_police_ita_detail",
    ),
    path(
        "odreferrals/disposition-detail/",
        od_disposition_detail,
        name="od_disposition_detail",
    ),
    path(
        "odreferrals/transport-detail/",
        od_transport_detail,
        name="od_transport_detail",
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
    # Back-compat alias: route removed per request but referenced in tests
    path("overdoses/by-case/", odreferrals, name="overdoses_by_case"),
    path("profile/", user_profile, name="user_profile"),
    path("authentication/", authentication, name="authentication"),
    path("demo/age-chart-variations/", age_chart_variations_demo, name="age_chart_variations_demo"),
]
