from django.db import models


class Patients(models.Model):
    id = models.AutoField(primary_key=True)
    client_3c = models.IntegerField()
    age = models.IntegerField()
    case_management = models.IntegerField()
    gender = models.CharField(max_length=20)
    insurance = models.CharField(max_length=50)
    pcp_agency = models.CharField(max_length=50)
    race = models.CharField(max_length=50)
    sex = models.CharField(max_length=20)
    sud = models.BooleanField()
    zip_code = models.CharField(max_length=50)
    created_date = models.DateField()
    modified_date = models.DateField()
    marital_status = models.CharField(max_length=30)
    veteran_status = models.CharField(max_length=30)

    def __str__(self):
        return "Patient ID: " + str(self.id)


# Create your models here.
class Encounters(models.Model):
    ID = models.AutoField(primary_key=True)
    referral_ID = models.IntegerField()
    port_referral_ID = models.IntegerField()
    patient_ID = models.IntegerField()
    encounter_date = models.DateField()
    pcp_agency = models.CharField(max_length=50)
    encounter_type_cat1 = models.CharField(max_length=50)
    encounter_type_cat2 = models.CharField(max_length=50)
    encounter_type_cat3 = models.CharField(max_length=50)
    encounter_stage = models.CharField(max_length=50)

    def __str__(self):
        return "Encounter ID: " + str(self.ID)
    

class Referrals(models.Model):
    ID = models.AutoField(primary_key=True)
    patient_ID = models.IntegerField()
    created = models.DateField()
    date_received = models.DateField()
    referral_agency = models.CharField(max_length=50)
    encounter_type_cat1 = models.CharField(max_length=50)
    encounter_type_cat2 = models.CharField(max_length=50)
    encounter_type_cat3 = models.CharField(max_length=50)
    referral_type_0 = models.CharField(max_length=50)
    referral_type_1 = models.CharField(max_length=50)
    referral_type_2 = models.CharField(max_length=50)
    referral_type_3 = models.CharField(max_length=50)
    referral_type_4 = models.CharField(max_length=50)

    def __str__(self):
        return "Referral ID: " + str(self.ID)
    
    
class ODReferrals(models.Model):
    ID = models.AutoField(primary_key=True)
    patient_id = models.IntegerField()
    patient_sex = models.CharField(max_length=20)
    patient_race = models.CharField(max_length=30, null=True, default=None)
    patient_age = models.IntegerField()
    patient_zipcode = models.CharField(max_length=20)
    patient_insurance = models.CharField(max_length=50)
    living_situation = models.CharField(max_length=20)
    od_date = models.DateTimeField(
        auto_now=False, auto_now_add=False, null=True, default=None
    )
    od_time = models.TimeField(
        auto_now=False, auto_now_add=False, null=True, default=None
    )
    delay_in_referral = models.CharField(max_length=50)
    cpm_notification = models.CharField(max_length=50)
    cpm_disposition = models.CharField(max_length=50)
    referral_agency = models.CharField(max_length=50, null=True)
    referral_source = models.CharField(max_length=20)
    od_address = models.CharField(max_length=100)
    lat = models.FloatField(max_length=50, null=True)
    long = models.FloatField(max_length=50, null=True)
    engagement_location = models.CharField(max_length=50)
    number_of_nonems_onscene = models.IntegerField()
    number_of_ems_onscene = models.IntegerField()
    number_of_peers_onscene = models.IntegerField()
    number_of_police_onscene = models.IntegerField()
    suspected_drug = models.CharField(max_length=50)
    cpr_administered = models.CharField(max_length=50)
    police_ita = models.CharField(max_length=50)
    disposition = models.CharField(max_length=50)
    transport_to_location = models.CharField(max_length=50)
    transported_by = models.CharField(max_length=50)
    narcan_given = models.BooleanField()
    narcan_doses_prior_to_ems = models.IntegerField()
    narcan_prior_to_ems_dosage = models.IntegerField()
    narcan_doses_by_ems = models.IntegerField()
    narcan_by_ems_dosage = models.FloatField()
    leave_behind_narcan_amount = models.IntegerField()
    persons_trained = models.FloatField(null=True)
    referral_to_sud_agency = models.BooleanField()
    no_referral_reason = models.CharField(max_length=50, default="No data", null=True)
    referral_rediscovery = models.IntegerField()
    referral_reflections = models.IntegerField()
    referral_pbh = models.IntegerField()
    referral_other = models.IntegerField()
    contact_level_rediscovery = models.CharField(max_length=50)
    contact_level_reflections = models.CharField(max_length=50)
    contact_level_pbh = models.CharField(max_length=50)
    contact_level_other = models.CharField(max_length=50)
    accepted_rediscovery = models.IntegerField()
    accepted_reflections = models.IntegerField()
    accepted_pbh = models.IntegerField()
    accepted_other = models.IntegerField()
    is_bup_indicated = models.IntegerField()
    bup_not_indicated_reason = models.CharField(max_length=50)
    bup_already_prescribed = models.CharField(max_length=50)
    bup_admin = models.IntegerField()
    client_agrees_to_mat = models.IntegerField()
    overdose_recent = models.CharField(max_length=30, null=True)

    def __str__(self):
        return "OD Referral ID: " + str(self.ID)