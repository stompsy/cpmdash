# Generated by Django 5.2.4 on 2025-07-20 08:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0007_alter_encounters_patient_id_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="odreferrals",
            name="jail_end_1",
            field=models.DateTimeField(default=None, null=True),
        ),
        migrations.AddField(
            model_name="odreferrals",
            name="jail_end_2",
            field=models.DateTimeField(default=None, null=True),
        ),
        migrations.AddField(
            model_name="odreferrals",
            name="jail_start_1",
            field=models.DateTimeField(default=None, null=True),
        ),
        migrations.AddField(
            model_name="odreferrals",
            name="jail_start_2",
            field=models.DateTimeField(default=None, null=True),
        ),
    ]
