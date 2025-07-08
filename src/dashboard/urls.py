from django.urls import path
from . import views

urlpatterns = [
    path("", views.odreferrals, name="index"),
    path("patients/", views.patients, name="patients"),
    path("encounters/", views.encounters, name="encounters"),
    path("referrals/", views.referrals, name="referrals"),
    path("odreferrals/", views.odreferrals, name="odreferrals"),
    path("odreferrals/trends/", views.odreferrals_trends, name="odreferrals_trends"),
    path("odreferrals/geographic/", views.odreferrals_geographic, name="odreferrals_geographic"),
    path("odreferrals/substances/", views.odreferrals_substances, name="odreferrals_substances"),
    path("odreferrals/operations/", views.odreferrals_operations, name="odreferrals_operations"),
    path("odreferrals/response/", views.odreferrals_response, name="odreferrals_response"),
    path("odreferrals/cpm/", views.odreferrals_cpm, name="odreferrals_cpm"),
    path("odreferrals/socioeconomic/", views.odreferrals_socioeconomic, name="odreferrals_socioeconomic"),
    
    # HTMX Chart Update Endpoints
    path("charts/heatmap/", views.htmx_heatmap_chart, name="htmx_heatmap"),
    path("charts/time-region-bars/", views.htmx_time_region_bars_chart, name="htmx_time_region_bars"),
    path("charts/hourly-breakdown/", views.htmx_hourly_breakdown_chart, name="htmx_hourly_breakdown"),
    path("charts/day-of-week/", views.htmx_day_of_week_chart, name="htmx_day_of_week"),
    path("charts/shift-scenarios/", views.htmx_shift_scenarios_chart, name="htmx_shift_scenarios"),
    path("charts/cost-benefit/", views.htmx_cost_benefit_chart, name="htmx_cost_benefit"),
]
