from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.core.cpimpact_export import build_export, write_zip
from apps.core.models import Agency


class Command(BaseCommand):
    help = (
        "Write a de-identified zip of patients, encounters, referrals and OD referrals "
        "for import into CP Impact. No addresses, coordinates, ZIPs or dates leave cpmdash."
    )

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--out", default="cpimpact_export.zip", help="Zip file to write.")
        parser.add_argument(
            "--agency",
            action="append",
            dest="agencies",
            help="Agency slug (repeatable). Default: all.",
        )
        parser.add_argument("--since-year", type=int, help="Only rows from this year on.")
        parser.add_argument(
            "--min-count",
            type=int,
            default=3,
            help='Text values seen fewer times than this in a column become "(rare)". 1 turns it off.',
        )

    def handle(self, *args: Any, **opts: Any) -> None:
        agencies = opts["agencies"]
        if agencies:
            known = set(Agency.objects.filter(slug__in=agencies).values_list("slug", flat=True))
            missing = sorted(set(agencies) - known)
            if missing:
                raise CommandError(f"Unknown agency slug(s): {', '.join(missing)}")
        tables, manifest = build_export(agencies, opts["since_year"], opts["min_count"])
        write_zip(opts["out"], tables, manifest)
        for name, n in manifest["rows"].items():
            self.stdout.write(f"  {name:<14} {n:>7,} rows")
        replaced = sum(sum(c.values()) for c in manifest["rare_values_replaced"].values())
        if replaced:
            self.stdout.write(f'  {replaced:,} rare text values replaced with "(rare)"')
        self.stdout.write(self.style.SUCCESS(f"Wrote {opts['out']}"))
