# Generated manually — seeds Q1 2026 HargroveMetricOverride notes into production.
# Only rows with meaningful notes content are included; accidental blank saves are omitted.
# Uses update_or_create on the natural key (year, quarter, metric_key) so this migration
# is fully idempotent and safe to run against a DB that already has some of these rows.

from typing import Any

from django.db import migrations


SEED_DATA = [
    # metric_id, metric_key, value, notes
    (
        "96719",
        "Total Contacts",
        "",
        "Encounters + Initial referrals + Initial OD referrals",
    ),
    (
        "—",
        "Avg distinct services per patient (Referrals)",
        "",
        "Avg 2 referrals / patient",
    ),
    (
        "1",
        "911/Walk-In",
        "",
        "42 patients",
    ),
    (
        "96729",
        "# Individuals Encountered Through Outreach",
        "",
        "Outreach referrals (agency contains 'Outreach') + encounters for those patients + 90×Tuesdays (60 afternoon + 30 morning)",
    ),
]


def seed_q1_2026(apps: Any, schema_editor: Any) -> None:
    HargroveMetricOverride = apps.get_model("dashboard", "HargroveMetricOverride")
    for metric_id, metric_key, value, notes in SEED_DATA:
        HargroveMetricOverride.objects.update_or_create(
            year=2026,
            quarter=1,
            metric_key=metric_key,
            defaults={
                "metric_id": metric_id,
                "value": value,
                "notes": notes,
            },
        )


def unseed_q1_2026(apps: Any, schema_editor: Any) -> None:
    HargroveMetricOverride = apps.get_model("dashboard", "HargroveMetricOverride")
    metric_keys = [row[1] for row in SEED_DATA]
    HargroveMetricOverride.objects.filter(
        year=2026,
        quarter=1,
        metric_key__in=metric_keys,
    ).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(seed_q1_2026, reverse_code=unseed_q1_2026),
    ]
