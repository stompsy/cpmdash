from django.urls import path
from . import views


urlpatterns = [
    path("cases/", views.cases, name="odreferrals"),
    path("repeats/", views.repeatods, name="repeatods"),
    path("shiftcoverage/", views.shiftcoverage, name="shiftcoverage"),
    path("opshieldinghope/", views.opshieldinghope, name="opshieldinghope"),
    path("costsavings/", views.costsavings, name="costsavings"),

    # HTMX Chart Update Endpoints
    path("charts/heatmap/", views.htmx_heatmap_chart, name="htmx_heatmap"),
    path("charts/hourly-breakdown/", views.htmx_hourly_breakdown_chart, name="htmx_hourly_breakdown"),
    path("charts/shift-scenarios/", views.htmx_shift_scenarios_chart, name="htmx_shift_scenarios"),
    path("charts/cost-benefit/", views.htmx_cost_benefit_chart, name="htmx_cost_benefit"),
]
