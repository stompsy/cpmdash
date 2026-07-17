from typing import Any

from django.contrib import admin
from django.http import HttpRequest

from .models import Agency, ContactSubmission, County, Encounters, ODReferrals, Patients, Referrals


class AgencyScopedModelAdmin(admin.ModelAdmin):
    """Restrict agency-bound model rows for non-superusers in Django admin."""

    def get_queryset(self, request: HttpRequest) -> Any:
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        agency_id = getattr(request.user, "agency_id", None)
        if not agency_id:
            return qs.none()
        return qs.filter(agency_id=agency_id)

    def formfield_for_foreignkey(self, db_field: Any, request: HttpRequest, **kwargs: Any) -> Any:
        if db_field.name == "agency" and not request.user.is_superuser:
            agency_id = getattr(request.user, "agency_id", None)
            if agency_id:
                kwargs["queryset"] = Agency.objects.filter(pk=agency_id)
            else:
                kwargs["queryset"] = Agency.objects.none()
        return super().formfield_for_foreignkey(db_field, request, **kwargs)


@admin.register(County)
class CountyAdmin(admin.ModelAdmin):
    list_display = ("name", "slug")
    search_fields = ("name", "slug")

    def get_queryset(self, request: HttpRequest) -> Any:
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        agency_id = getattr(request.user, "agency_id", None)
        if not agency_id:
            return qs.none()
        return qs.filter(agencies__id=agency_id).distinct()


@admin.register(Agency)
class AgencyAdmin(admin.ModelAdmin):
    list_display = ("name", "county", "slug")
    search_fields = ("name", "slug", "county__name")
    list_filter = ("county",)

    def get_queryset(self, request: HttpRequest) -> Any:
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        agency_id = getattr(request.user, "agency_id", None)
        if not agency_id:
            return qs.none()
        return qs.filter(pk=agency_id)


@admin.register(Patients)
class PatientsAdmin(AgencyScopedModelAdmin):
    list_display = (
        "id",
        "agency",
        "age",
        "sex",
        "race",
        "zip_code",
        "aud",
        "three_c_client",
        "created_date",
    )
    search_fields = ("id", "zip_code")
    list_filter = ("agency", "sex", "race", "zip_code", "aud", "three_c_client")


@admin.register(Encounters)
class EncountersAdmin(AgencyScopedModelAdmin):
    list_display = ("ID", "agency", "patient_ID", "encounter_date", "encounter_type_cat1")
    search_fields = ("ID", "patient_ID")
    list_filter = ("agency", "encounter_type_cat1", "encounter_date")


@admin.register(Referrals)
class ReferralsAdmin(AgencyScopedModelAdmin):
    list_display = (
        "ID",
        "agency",
        "patient_ID",
        "date_received",
        "referral_agency",
        "referral_closed_reason",
    )
    search_fields = ("ID", "patient_ID", "referral_agency")
    list_filter = ("agency", "referral_agency", "referral_closed_reason")


@admin.register(ODReferrals)
class ODReferralsAdmin(AgencyScopedModelAdmin):
    list_display = ("ID", "agency", "od_date", "patient_age", "patient_sex", "disposition")
    search_fields = ("ID", "od_address")
    list_filter = ("agency", "disposition", "patient_sex")


@admin.register(ContactSubmission)
class ContactSubmissionAdmin(admin.ModelAdmin):
    list_display = ("created_at", "first_name", "last_name", "email", "organization", "is_read")
    list_filter = ("is_read", "created_at")
    search_fields = ("first_name", "last_name", "email", "organization", "message")
    readonly_fields = ("created_at",)
    ordering = ("-created_at",)
