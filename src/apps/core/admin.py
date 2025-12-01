from django.contrib import admin

from .models import ContactSubmission, Encounters, ODReferrals, Patients, Referrals


@admin.register(Patients)
class PatientsAdmin(admin.ModelAdmin):
    list_display = ("id", "age", "sex", "race", "zip_code", "created_date")
    search_fields = ("id", "zip_code")
    list_filter = ("sex", "race", "zip_code")


@admin.register(Encounters)
class EncountersAdmin(admin.ModelAdmin):
    list_display = ("ID", "patient_ID", "encounter_date", "encounter_type_cat1")
    search_fields = ("ID", "patient_ID")
    list_filter = ("encounter_type_cat1", "encounter_date")


@admin.register(Referrals)
class ReferralsAdmin(admin.ModelAdmin):
    list_display = (
        "ID",
        "patient_ID",
        "date_received",
        "referral_agency",
        "referral_closed_reason",
    )
    search_fields = ("ID", "patient_ID", "referral_agency")
    list_filter = ("referral_agency", "referral_closed_reason")


@admin.register(ODReferrals)
class ODReferralsAdmin(admin.ModelAdmin):
    list_display = ("ID", "od_date", "patient_age", "patient_sex", "disposition")
    search_fields = ("ID", "od_address")
    list_filter = ("disposition", "patient_sex")


@admin.register(ContactSubmission)
class ContactSubmissionAdmin(admin.ModelAdmin):
    list_display = ("created_at", "first_name", "last_name", "email", "organization", "is_read")
    list_filter = ("is_read", "created_at")
    search_fields = ("first_name", "last_name", "email", "organization", "message")
    readonly_fields = ("created_at",)
    ordering = ("-created_at",)
