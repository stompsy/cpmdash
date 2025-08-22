from django.urls import path

from .views import (
    cases,
    costsavings,
    htmx_cost_benefit_chart,
    htmx_heatmap_chart,
    htmx_hourly_breakdown_chart,
    htmx_shift_scenarios_chart,
    opshield,
    repeatods,
    shiftcoverage,
)

app_name = "cases"

urlpatterns = [
    path("opshield/", opshield, name="opshield"),
    path("cases/", cases, name="odreferrals"),
    path("repeats/", repeatods, name="repeatods"),
    path("shiftcoverage/", shiftcoverage, name="shiftcoverage"),
    path("costsavings/", costsavings, name="costsavings"),
    # HTMX Chart Update Endpoints
    path("charts/heatmap/", htmx_heatmap_chart, name="htmx_heatmap"),
    path("charts/hourly-breakdown/", htmx_hourly_breakdown_chart, name="htmx_hourly_breakdown"),
    path("charts/shift-scenarios/", htmx_shift_scenarios_chart, name="htmx_shift_scenarios"),
    path("charts/cost-benefit/", htmx_cost_benefit_chart, name="htmx_cost_benefit"),
]
