from django.urls import path
from . import views

urlpatterns = [
    path("", views.overview, name="dashboard_overview"),

    path("patients/", views.patients, name="patients"),

    path("referrals/", views.referrals, name="referrals"),

    path("odreferrals/", views.odreferrals, name="odreferrals"),
    path("odreferrals/monthly/", views.odreferrals_monthly, name="odreferrals_monthly"),

    path("encounters/", views.encounters, name="encounters"),

    path("profile/", views.user_profile, name="user_profile"),
    path("authentication/", views.authentication, name="authentication"),
]
