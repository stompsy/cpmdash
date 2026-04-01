from __future__ import annotations

from django.conf import settings
from django.db import models


class DataImportBatch(models.Model):
    """Tracks a single import session — upload through commit."""

    class Status(models.TextChoices):
        UPLOADING = "uploading", "Uploading"
        PROCESSING = "processing", "Processing"
        REVIEW = "review", "Review"
        COMMITTED = "committed", "Committed"
        FAILED = "failed", "Failed"

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="data_import_batches",
    )
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.UPLOADING)
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    committed_at = models.DateTimeField(null=True, blank=True)

    # Row counts snapshot taken at commit time for audit
    committed_patients = models.IntegerField(null=True, blank=True)
    committed_referrals = models.IntegerField(null=True, blank=True)
    committed_odreferrals = models.IntegerField(null=True, blank=True)
    committed_encounters = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Import #{self.pk} — {self.get_status_display()} ({self.created_at:%Y-%m-%d %H:%M})"


class DataImportFile(models.Model):
    """An uploaded CSV file attached to a batch."""

    class FileType(models.TextChoices):
        PATIENTS = "patients", "Patients"
        REFERRALS = "referrals", "Referrals"
        ODREFERRALS = "odreferrals", "OD Referrals"
        ENCOUNTERS = "encounters", "Encounters"

    batch = models.ForeignKey(DataImportBatch, on_delete=models.CASCADE, related_name="files")
    file_type = models.CharField(max_length=20, choices=FileType.choices)
    file = models.FileField(upload_to="uploads/data_import/")
    original_filename = models.CharField(max_length=255)
    row_count = models.IntegerField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.original_filename} ({self.get_file_type_display()})"


class ProcessingLog(models.Model):
    """Persistent log of processing actions. Manually managed by the user."""

    batch = models.ForeignKey(
        DataImportBatch, on_delete=models.SET_NULL, null=True, related_name="logs"
    )
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        batch_label = f"Batch #{self.batch.pk}" if self.batch else "No batch"
        return f"Log {self.pk} — {batch_label} ({self.created_at:%Y-%m-%d %H:%M})"


# ======================================================================
# Staging models — mirror core models with batch FK + row metadata
# ======================================================================


class RowStatus(models.TextChoices):
    NEW = "new", "New"
    EXISTING = "existing", "Existing"
    WARNING = "warning", "Warning"
    ERROR = "error", "Error"


class StagingPatient(models.Model):
    batch = models.ForeignKey(
        DataImportBatch, on_delete=models.CASCADE, related_name="staging_patients"
    )
    row_status = models.CharField(max_length=10, choices=RowStatus.choices, default=RowStatus.NEW)
    validation_notes = models.TextField(blank=True, default="")
    source_id = models.IntegerField(help_text="Original ID from source system")

    # Mirror of core.Patients fields
    age = models.IntegerField(null=True, blank=True)
    insurance = models.CharField(max_length=50, default="", blank=True)
    pcp_agency = models.CharField(max_length=50, default="", blank=True)
    race = models.CharField(max_length=50, default="", blank=True)
    sex = models.CharField(max_length=10, default="", blank=True)
    sud = models.BooleanField(null=True, blank=True)
    behavioral_health = models.BooleanField(null=True, blank=True)
    zip_code = models.CharField(max_length=50, default="", blank=True)
    created_date = models.DateField(null=True, blank=True)
    modified_date = models.DateField(null=True, blank=True)
    marital_status = models.CharField(max_length=50, default="", blank=True)
    veteran_status = models.CharField(max_length=50, default="", blank=True)
    aud = models.BooleanField(null=True, blank=True)
    three_c_client = models.BooleanField(null=True, blank=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["source_id"]

    def __str__(self) -> str:
        return f"Staging Patient {self.source_id} (Batch #{self.batch.pk})"


class StagingReferral(models.Model):
    batch = models.ForeignKey(
        DataImportBatch, on_delete=models.CASCADE, related_name="staging_referrals"
    )
    row_status = models.CharField(max_length=10, choices=RowStatus.choices, default=RowStatus.NEW)
    validation_notes = models.TextField(blank=True, default="")
    source_id = models.IntegerField(help_text="Original ID from source system")

    # Mirror of core.Referrals fields
    patient_ID = models.IntegerField(null=True, blank=True)
    sex = models.CharField(max_length=10, default="", blank=True)
    age = models.IntegerField(null=True, blank=True)
    date_received = models.DateField(null=True, blank=True)
    referral_agency = models.CharField(max_length=50, default="", blank=True)
    encounter_type_cat1 = models.CharField(max_length=50, default="", blank=True)
    encounter_type_cat2 = models.CharField(max_length=50, default="", blank=True)
    encounter_type_cat3 = models.CharField(max_length=50, default="", blank=True)
    referral_closed_reason = models.CharField(max_length=50, default="", blank=True)
    zipcode = models.CharField(max_length=50, default="", blank=True)
    insurance = models.CharField(max_length=50, default="", blank=True)
    referral_1 = models.CharField(max_length=50, default="", blank=True)
    referral_2 = models.CharField(max_length=50, default="", blank=True)
    referral_3 = models.CharField(max_length=50, default="", blank=True)
    referral_4 = models.CharField(max_length=50, default="", blank=True)
    referral_5 = models.CharField(max_length=50, default="", blank=True)

    class Meta:
        ordering = ["source_id"]

    def __str__(self) -> str:
        return f"Staging Referral {self.source_id} (Batch #{self.batch.pk})"


class StagingODReferral(models.Model):
    batch = models.ForeignKey(
        DataImportBatch, on_delete=models.CASCADE, related_name="staging_odreferrals"
    )
    row_status = models.CharField(max_length=10, choices=RowStatus.choices, default=RowStatus.NEW)
    validation_notes = models.TextField(blank=True, default="")
    source_id = models.IntegerField(help_text="Original ID from source system")

    # Mirror of core.ODReferrals fields
    patient_id = models.IntegerField(null=True, blank=True)
    patient_sex = models.CharField(max_length=20, default="", blank=True)
    patient_race = models.CharField(max_length=30, default="", blank=True)
    patient_age = models.IntegerField(null=True, blank=True)
    patient_zipcode = models.CharField(max_length=20, default="", blank=True)
    patient_insurance = models.CharField(max_length=50, default="", blank=True)
    living_situation = models.CharField(max_length=20, default="", blank=True)
    od_date = models.DateTimeField(null=True, blank=True)
    delay_in_referral = models.CharField(max_length=50, default="", blank=True)
    cpm_notification = models.CharField(max_length=50, default="", blank=True)
    cpm_disposition = models.CharField(max_length=50, default="", blank=True)
    referral_agency = models.CharField(max_length=50, default="", blank=True)
    referral_source = models.CharField(max_length=50, default="", blank=True)
    od_address = models.CharField(max_length=250, default="", blank=True)
    lat = models.FloatField(null=True, blank=True)
    long = models.FloatField(null=True, blank=True)
    engagement_location = models.CharField(max_length=50, default="", blank=True)
    number_of_nonems_onscene = models.IntegerField(null=True, blank=True)
    number_of_ems_onscene = models.IntegerField(null=True, blank=True)
    number_of_peers_onscene = models.IntegerField(null=True, blank=True)
    number_of_police_onscene = models.IntegerField(null=True, blank=True)
    suspected_drug = models.CharField(max_length=50, default="", blank=True)
    cpr_administered = models.CharField(max_length=50, default="", blank=True)
    police_ita = models.CharField(max_length=50, default="", blank=True)
    disposition = models.CharField(max_length=50, default="", blank=True)
    transport_to_location = models.CharField(max_length=50, default="", blank=True)
    transported_by = models.CharField(max_length=50, default="", blank=True)
    narcan_given = models.BooleanField(null=True, blank=True)
    narcan_doses_prior_to_ems = models.IntegerField(null=True, blank=True)
    narcan_prior_to_ems_dosage = models.FloatField(null=True, blank=True)
    narcan_doses_by_ems = models.IntegerField(null=True, blank=True)
    narcan_by_ems_dosage = models.FloatField(null=True, blank=True)
    leave_behind_narcan_amount = models.IntegerField(null=True, blank=True)
    persons_trained = models.FloatField(null=True, blank=True)
    referral_to_sud_agency = models.BooleanField(null=True, blank=True)
    referral_rediscovery = models.IntegerField(null=True, blank=True)
    referral_reflections = models.IntegerField(null=True, blank=True)
    referral_pbh = models.IntegerField(null=True, blank=True)
    referral_other = models.IntegerField(null=True, blank=True)
    contact_level_rediscovery = models.CharField(max_length=50, default="", blank=True)
    contact_level_reflections = models.CharField(max_length=50, default="", blank=True)
    contact_level_pbh = models.CharField(max_length=50, default="", blank=True)
    contact_level_other = models.CharField(max_length=50, default="", blank=True)
    accepted_rediscovery = models.IntegerField(null=True, blank=True)
    accepted_reflections = models.IntegerField(null=True, blank=True)
    accepted_pbh = models.IntegerField(null=True, blank=True)
    accepted_other = models.IntegerField(null=True, blank=True)
    is_bup_indicated = models.IntegerField(null=True, blank=True)
    bup_not_indicated_reason = models.CharField(max_length=50, default="", blank=True)
    bup_already_prescribed = models.CharField(max_length=50, default="", blank=True)
    bup_admin = models.IntegerField(null=True, blank=True)
    client_agrees_to_mat = models.IntegerField(null=True, blank=True)
    overdose_recent = models.CharField(max_length=30, default="", blank=True)
    jail_start_1 = models.DateField(null=True, blank=True)
    jail_end_1 = models.DateField(null=True, blank=True)
    jail_start_2 = models.DateField(null=True, blank=True)
    jail_end_2 = models.DateField(null=True, blank=True)

    class Meta:
        ordering = ["source_id"]

    def __str__(self) -> str:
        return f"Staging OD Referral {self.source_id} (Batch #{self.batch.pk})"


class StagingEncounter(models.Model):
    batch = models.ForeignKey(
        DataImportBatch, on_delete=models.CASCADE, related_name="staging_encounters"
    )
    row_status = models.CharField(max_length=10, choices=RowStatus.choices, default=RowStatus.NEW)
    validation_notes = models.TextField(blank=True, default="")
    source_id = models.IntegerField(help_text="Original ID from source system")

    # Mirror of core.Encounters fields
    referral_ID = models.IntegerField(null=True, blank=True)
    port_referral_ID = models.IntegerField(null=True, blank=True)
    patient_ID = models.IntegerField(null=True, blank=True)
    encounter_date = models.DateField(null=True, blank=True)
    pcp_agency = models.CharField(max_length=50, default="", blank=True)
    encounter_type_cat1 = models.CharField(max_length=50, default="", blank=True)
    encounter_type_cat2 = models.CharField(max_length=50, default="", blank=True)
    encounter_type_cat3 = models.CharField(max_length=50, default="", blank=True)

    class Meta:
        ordering = ["source_id"]

    def __str__(self) -> str:
        return f"Staging Encounter {self.source_id} (Batch #{self.batch.pk})"
