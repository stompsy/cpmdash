# Generated by Django 5.2.4 on 2025-07-22 05:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0010_alter_odreferrals_jail_end_1_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="odreferrals",
            name="jail_end_1",
            field=models.DateField(default=None, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="jail_end_2",
            field=models.DateField(default=None, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="jail_start_1",
            field=models.DateField(default=None, null=True),
        ),
        migrations.AlterField(
            model_name="odreferrals",
            name="jail_start_2",
            field=models.DateField(default=None, null=True),
        ),
    ]
