from typing import Any

import django.db.models.deletion
from django.db import migrations, models


def assign_legacy_agency(apps: Any, schema_editor: Any) -> None:
    Agency = apps.get_model("core", "Agency")
    HargroveMetricOverride = apps.get_model("dashboard", "HargroveMetricOverride")
    agency = Agency.objects.get(county__slug="clallam-county", slug="port-angeles")
    HargroveMetricOverride.objects.filter(agency__isnull=True).update(agency_id=agency.pk)


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0009_backfill_default_agency"),
        ("dashboard", "0003_alter_hargrovemetricoverride_value"),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name="hargrovemetricoverride",
            unique_together=set(),
        ),
        migrations.AddField(
            model_name="hargrovemetricoverride",
            name="agency",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                to="core.agency",
            ),
        ),
        migrations.RunPython(assign_legacy_agency, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="hargrovemetricoverride",
            name="agency",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.PROTECT,
                to="core.agency",
            ),
        ),
        migrations.AddConstraint(
            model_name="hargrovemetricoverride",
            constraint=models.UniqueConstraint(
                fields=("agency", "year", "quarter", "metric_key"),
                name="unique_hargrove_override_per_agency_period_metric",
            ),
        ),
    ]
