from typing import Any

from django.db import migrations


def backfill_default_agency(apps: Any, schema_editor: Any) -> None:
    County = apps.get_model("core", "County")
    Agency = apps.get_model("core", "Agency")
    Patients = apps.get_model("core", "Patients")
    Referrals = apps.get_model("core", "Referrals")
    Encounters = apps.get_model("core", "Encounters")
    ODReferrals = apps.get_model("core", "ODReferrals")
    User = apps.get_model("accounts", "User")

    clallam, _ = County.objects.get_or_create(
        slug="clallam-county",
        defaults={"name": "Clallam County"},
    )
    port_angeles, _ = Agency.objects.get_or_create(
        county=clallam,
        slug="port-angeles",
        defaults={"name": "Port Angeles"},
    )

    for model in (Patients, Referrals, Encounters, ODReferrals):
        model.objects.filter(agency__isnull=True).update(agency=port_angeles)

    User.objects.filter(agency__isnull=True).update(agency=port_angeles)


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0008_seed_default_tenants"),
        ("accounts", "0005_user_agency"),
    ]

    operations = [
        migrations.RunPython(backfill_default_agency, migrations.RunPython.noop),
    ]
