from typing import Any

from django.db import migrations


def seed_default_tenants(apps: Any, schema_editor: Any) -> None:
    County = apps.get_model("core", "County")
    Agency = apps.get_model("core", "Agency")

    clallam, _ = County.objects.get_or_create(
        slug="clallam-county",
        defaults={"name": "Clallam County"},
    )

    Agency.objects.get_or_create(
        county=clallam,
        slug="port-angeles",
        defaults={"name": "Port Angeles"},
    )
    Agency.objects.get_or_create(
        county=clallam,
        slug="sequim",
        defaults={"name": "Sequim"},
    )


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0007_agency_county_encounters_agency_odreferrals_agency_and_more"),
    ]

    operations = [
        migrations.RunPython(seed_default_tenants, migrations.RunPython.noop),
    ]
