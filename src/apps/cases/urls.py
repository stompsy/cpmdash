from django.urls import path

from .views import (
    cases,
    costsavings,
    opshield,
)

app_name = "cases"

urlpatterns = [
    path("", opshield, name="index"),
    # Alias for tests and backward-compatibility
    path("opshield/", opshield, name="opshield"),
    path("cases/", cases, name="odreferrals"),
    path("costsavings/", costsavings, name="costsavings"),
]
