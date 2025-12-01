from django.db import models


class Patients(models.Model):
    id = models.AutoField(primary_key=True)
    age = models.IntegerField(null=True, blank=True)
    insurance = models.CharField(max_length=50, default="", blank=True)
    pcp_agency = models.CharField(max_length=50, default="", blank=True)
    race = models.CharField(max_length=50, default="", blank=True)
    sex = models.CharField(max_length=10, default="", blank=True)
    sud = models.BooleanField(null=True, blank=True)
    behavioral_health = models.BooleanField(null=True, blank=True)
    zip_code = models.CharField(max_length=50, default="", blank=True)
    created_date = models.DateField(max_length=20, null=True, blank=True)
    modified_date = models.DateField(max_length=20, null=True, blank=True)
    marital_status = models.CharField(max_length=50, default="", blank=True)
    veteran_status = models.CharField(max_length=50, default="", blank=True)

    def __str__(self) -> str:
        return "Patient ID: " + str(self.id)


# Create your models here.
class Encounters(models.Model):
    ID = models.AutoField(primary_key=True)
    referral_ID = models.IntegerField(null=True, blank=True)
    port_referral_ID = models.IntegerField(null=True, blank=True)
    patient_ID = models.IntegerField(null=True, blank=True)
    encounter_date = models.DateField(max_length=20, null=True, blank=True)
    pcp_agency = models.CharField(max_length=50, default="", blank=True)
    encounter_type_cat1 = models.CharField(max_length=50, default="", blank=True)
    encounter_type_cat2 = models.CharField(max_length=50, default="", blank=True)
    encounter_type_cat3 = models.CharField(max_length=50, default="", blank=True)

    def __str__(self) -> str:
        return "Encounter ID: " + str(self.ID)


class Referrals(models.Model):
    ID = models.AutoField(primary_key=True)
    patient_ID = models.IntegerField(null=True, blank=True)
    sex = models.CharField(max_length=10, default="", blank=True)
    age = models.IntegerField(null=True, blank=True)
    date_received = models.DateField(max_length=20, null=True, blank=True)
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

    def __str__(self) -> str:
        return "Referral ID: " + str(self.ID)


class ODReferrals(models.Model):
    ID = models.AutoField(primary_key=True)
    patient_id = models.IntegerField(null=True, blank=True)
    patient_sex = models.CharField(max_length=20, default="", blank=True)
    patient_race = models.CharField(max_length=30, default="", blank=True)
    patient_age = models.IntegerField(null=True, blank=True)
    patient_zipcode = models.CharField(max_length=20, default="", blank=True)
    patient_insurance = models.CharField(max_length=50, default="", blank=True)
    living_situation = models.CharField(max_length=20, default="", blank=True)
    od_date = models.DateTimeField(auto_now=False, auto_now_add=False, null=True, default=None)
    delay_in_referral = models.CharField(max_length=50, default="", blank=True)
    cpm_notification = models.CharField(max_length=50, default="", blank=True)
    cpm_disposition = models.CharField(max_length=50, default="", blank=True)
    referral_agency = models.CharField(max_length=50, default="", blank=True)
    referral_source = models.CharField(max_length=50, default="", blank=True)
    od_address = models.CharField(max_length=250, default="", blank=True)
    lat = models.FloatField(max_length=50, null=True, blank=True)
    long = models.FloatField(max_length=50, null=True, blank=True)
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
    narcan_prior_to_ems_dosage = models.FloatField(max_length=50, null=True, blank=True)
    narcan_doses_by_ems = models.IntegerField(null=True, blank=True)
    narcan_by_ems_dosage = models.FloatField(max_length=50, null=True, blank=True)
    leave_behind_narcan_amount = models.IntegerField(null=True, blank=True)
    persons_trained = models.FloatField(max_length=50, null=True, blank=True)
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
    jail_start_1 = models.DateField(auto_now=False, auto_now_add=False, null=True, default=None)
    jail_end_1 = models.DateField(auto_now=False, auto_now_add=False, null=True, default=None)
    jail_start_2 = models.DateField(auto_now=False, auto_now_add=False, null=True, default=None)
    jail_end_2 = models.DateField(auto_now=False, auto_now_add=False, null=True, default=None)

    def __str__(self) -> str:
        return "OD Referral ID: " + str(self.ID)


class ContactSubmission(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    organization = models.CharField(max_length=200, blank=True)
    email = models.EmailField()
    phone_number = models.CharField(max_length=20, blank=True)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    def __str__(self) -> str:
        return f"{self.first_name} {self.last_name} - {self.created_at}"
