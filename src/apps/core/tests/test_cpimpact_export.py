import csv
import io
import json
import zipfile
from datetime import date, datetime

import pytest
from django.core.management import call_command

from apps.core.cpimpact_export import RARE, age_band, build_export, gap_bucket, year_quarter
from apps.core.models import Agency, County, Encounters, ODReferrals, Patients, Referrals


def test_age_band_matches_cp_impact():
    assert [age_band(a) for a in (None, 5, 18, 34, 35, 64, 65, 79, 80, 97)] == [
        "unknown",
        "lt18",
        "18_34",
        "18_34",
        "35_49",
        "50_64",
        "65_79",
        "65_79",
        "80p",
        "80p",
    ]


def test_year_quarter_and_gaps():
    assert year_quarter(date(2025, 3, 31)) == ("2025", "1")
    assert year_quarter(datetime(2025, 10, 1, 4)) == ("2025", "4")
    assert year_quarter(None) == ("", "")
    d = date(2025, 1, 1)
    assert gap_bucket(None, d) == "first"
    assert gap_bucket(d, d) == "same_day"
    assert gap_bucket(d, date(2025, 1, 8)) == "1_7d"
    assert gap_bucket(d, date(2025, 6, 1)) == "91d_plus"


@pytest.fixture
def data(db):
    county = County.objects.create(name="Test County", slug="test-county")
    ag = Agency.objects.create(name="Test FD", slug="test-fd", county=county)
    p = Patients.objects.create(
        agency=ag,
        age=93,
        address="123 Main St",
        latitude=48.1,
        longitude=-123.4,
        zip_code="98362",
        race="White",
        sex="M",
        insurance="Medicaid",
        behavioral_health=True,
    )
    for i, d in enumerate([date(2024, 12, 30), date(2024, 12, 30), date(2025, 1, 20)]):
        Encounters.objects.create(
            agency=ag,
            patient_ID=p.id,
            encounter_date=d,
            encounter_type_cat1="John Smith" if i == 2 else "Fall",
        )
    Referrals.objects.create(
        agency=ag, patient_ID=p.id, date_received=date(2024, 11, 2), zipcode="98362", age=40
    )
    ODReferrals.objects.create(
        agency=ag,
        patient_id=p.id,
        od_date=datetime(2025, 2, 3, 1, 0),
        od_address="1 Pier Rd",
        lat=48.0,
        long=-123.0,
        jail_start_1=date(2024, 1, 1),
        number_of_police_onscene=2,
    )
    return ag


def _read(zf, name):
    return list(csv.DictReader(io.StringIO(zf.read(name).decode())))


def test_export_scrubs_identifiers(data, tmp_path):
    out = tmp_path / "x.zip"
    call_command("export_cpimpact", out=str(out), stdout=io.StringIO())
    zf = zipfile.ZipFile(out)
    blob = b"".join(zf.read(n) for n in zf.namelist() if n.endswith(".csv"))
    for bad in (
        b"123 Main",
        b"98362",
        b"48.1",
        b"Pier",
        b"2024-",
        b"2025-",
        b"John Smith",
        b"White",
    ):
        assert bad not in blob, bad

    pats = _read(zf, "patients.csv")
    assert pats[0]["age_band"] == "80p" and pats[0]["behavioral_health"] == "1"
    assert pats[0]["patient_key"].startswith("P")

    enc = _read(zf, "encounters.csv")
    assert [(e["year"], e["quarter"], e["seq"], e["gap_from_prev"]) for e in enc] == [
        ("2024", "4", "1", "first"),
        ("2024", "4", "2", "same_day"),
        ("2025", "1", "3", "8_30d"),
    ]
    assert enc[2]["encounter_type_cat1"] == RARE  # seen once, min_count 3

    od = _read(zf, "od_referrals.csv")
    assert od[0]["number_of_law_enforcement_onscene"] == "2"
    assert "jail_start_1" not in od[0] and "od_address" not in od[0]

    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["rows"] == {"patients": 1, "encounters": 3, "referrals": 1, "od_referrals": 1}


def test_since_year_keeps_true_sequence(data):
    tables, _ = build_export(since_year=2025, min_count=1)
    enc = next(t for t in tables if t.name == "encounters")
    assert [(r["seq"], r["gap_from_prev"]) for r in enc.rows] == [("3", "8_30d")]


def test_unknown_agency_errors(data):
    from django.core.management.base import CommandError

    with pytest.raises(CommandError):
        call_command("export_cpimpact", agencies=["nope"], out="/dev/null")
