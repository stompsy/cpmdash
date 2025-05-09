# Generated by Django 5.2 on 2025-05-01 05:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0006_remove_patients_case_management_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="encounters",
            name="patient_ID",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="encounters",
            name="port_referral_ID",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="encounters",
            name="referral_ID",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="accepted_other",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="accepted_pbh",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="accepted_rediscovery",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="accepted_reflections",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="bup_admin",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="client_agrees_to_mat",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="is_bup_indicated",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="leave_behind_narcan_amount",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="narcan_doses_by_ems",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="narcan_doses_prior_to_ems",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="number_of_ems_onscene",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="number_of_nonems_onscene",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="number_of_peers_onscene",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="number_of_police_onscene",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="patient_age",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="patient_id",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="referral_other",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="referral_pbh",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="referral_rediscovery",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="referral_reflections",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="patients",
            name="age",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="referrals",
            name="patient_ID",
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
