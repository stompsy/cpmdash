# import_data.py
# This script imports data from a CSV file and populates the database with the data.
import csv
from datetime import datetime
from dashboard.models import ODReferrals


def import_od_referrals(csv_file_path):
    with open(csv_file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ODReferrals.objects.create(
                ID=row["ID"],
                patient_id=row["patient_id"],
                patient_sex=row["patient_sex"],
                patient_age=row["patient_age"],
                patient_zipcode=row["patient_zipcode"],
                patient_insurance=row["patient_insurance"],
                living_situation=row["living_situation"],
                referral_date=datetime.strptime(row["referral_date"], "%Y-%m-%d"),
                overdose_date=datetime.strptime(row["overdose_date"], "%Y-%m-%d"),
                cpm_notification=row["cpm_notification"],
                cpm_disposition=row["cpm_disposition"],
                referral_source=row["referral_source"],
                od_address=row["od_address"],
                engagement_location=row["engagement_location"],
                number_of_nonems_onscene=row["number_of_nonems_onscene"],
                number_of_ems_onscene=row["number_of_ems_onscene"],
                number_of_peers_onscene=row["number_of_peers_onscene"],
                number_of_police_onscene=row["number_of_police_onscene"],
                suspected_drug=row["suspected_drug"],
                cpr_administered=row["cpr_administered"],
                police_ita=row["police_ita"],
                disposition=row["disposition"],
                transport_to_location=row["transport_to_location"],
                transported_by=row["transported_by"],
                narcan_given=row["narcan_given"],
                narcan_doses_prior_to_ems=row["narcan_doses_prior_to_ems"],
                narcan_prior_to_ems_dosage=row["narcan_prior_to_ems_dosage"],
                narcan_doses_by_ems=row["narcan_doses_by_ems"],
                narcan_by_ems_dosage=row["narcan_by_ems_dosage"],
                leave_behind_narcan_amount=row["leave_behind_narcan_amount"],
                referral_to_sud_agency=row["referral_to_sud_agency"],
                no_referral_reason=row["no_referral_reason"],
                referral_rediscovery=row["referral_rediscovery"],
                referral_reflections=row["referral_reflections"],
                referral_pbh=row["referral_pbh"],
                referral_other=row["referral_other"],
                contact_level_rediscovery=row["contact_level_rediscovery"],
                contact_level_reflections=row["contact_level_reflections"],
                contact_level_pbh=row["contact_level_pbh"],
                contact_level_other=row["contact_level_other"],
                accepted_rediscovery=row["accepted_rediscovery"],
                accepted_reflections=row["accepted_reflections"],
                accepted_pbh=row["accepted_pbh"],
                accepted_other=row["accepted_other"],
                is_bup_indicated=row["is_bup_indicated"],
                bup_not_indicated_reason=row["bup_not_indicated_reason"],
                bup_already_prescribed=row["bup_already_prescribed"],
                bup_admin=row["bup_admin"],
                client_agrees_to_mat=row["client_agrees_to_mat"],
                delay_in_referral=row["delay_in_referral"],
            )


if __name__ == "__main__":
    csv_file_path = "referrals_port_clean.csv"
    import_od_referrals(csv_file_path)
