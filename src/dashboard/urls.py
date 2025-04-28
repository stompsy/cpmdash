from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("patients/", views.patients, name="patients"),
    path("encounters/", views.encounters, name="encounters"),
    path("referrals/", views.referrals, name="referrals"),
    path("odreferrals/", views.odreferrals, name="odreferrals"),
]
