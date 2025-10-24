import pandas as pd
from django.db.models import Q

from ..core.models import ODReferrals, Patients


def get_odreferral_counts() -> dict:
    total = ODReferrals.objects.count()
    dates = ODReferrals.objects.values_list("od_date", flat=True)
    df = pd.DataFrame({"od_date": dates}).dropna(subset=["od_date"])

    if df.empty:
        by_year = {}
    else:
        df["od_date"] = pd.to_datetime(df["od_date"])
        df["year"] = df["od_date"].dt.year
        by_year = (
            df["year"]
            .value_counts()
            .sort_index(ascending=False)  # descending
            .to_dict()
        )

    return {"total": total, "by_year": by_year}


def get_od_metrics(year: int, population: int = 20_000) -> dict:
    """
    Returns a dict with:
        - count: total OD referrals in `year`
        - rate_per_100k: referrals per 100 000 population
        - percentage: % of population with at least one referral
    """
    # fetch count
    count = ODReferrals.objects.filter(od_date__year=year).count()

    # guard zero‐pop
    if not population:
        return {"count": count, "rate_per_100k": 0.0, "percentage": 0.0}

    # compute
    rate_per_100k = (count / population) * 100_000
    percentage = (count / population) * 100

    return {
        "count": count,
        "rate_per_100k": rate_per_100k,
        "percentage": percentage,
    }


def get_cost_savings_metrics() -> dict:
    """
    Calculate aggregated cost savings metrics from all years of data using patient-based methodology.
    Returns metrics for 911 calls prevented, transports averted, ED visits avoided, and total savings.
    """
    # STATIC authoritative patient counts provided (do not trust historical created_date)
    static_patient_counts = {2021: 419, 2022: 748, 2023: 396, 2024: 411, 2025: 298}

    # Derive ordered years present in static mapping (filter to those <= current year if needed)
    years = sorted(static_patient_counts.keys())

    # Build a lightweight iterable mimicking the old structure
    patients_by_year = [{"year": y, "total_patients": static_patient_counts[y]} for y in years]

    total_calls_prevented = 0
    total_transports_averted = 0
    total_ed_visits_avoided = 0
    total_savings = 0
    yearly_breakdown: list[dict] = []

    # Process each year
    for year_data in patients_by_year:
        total_patients = year_data["total_patients"]

        # 911 Services Usage Calculation
        patients_used_911 = int(total_patients * 0.42)  # 42% of patients used 911 services
        patients_reduced_911 = int(patients_used_911 * 0.71)  # 71% had reduction in 911 usage
        calls_prevented_year = (
            patients_reduced_911  # Total calls prevented = patients with reduction
        )

        # Transport Calculations
        transports_prevented_year = int(
            calls_prevented_year * 0.56
        )  # 56% of 911 calls result in transport
        non_transport_calls = calls_prevented_year - transports_prevented_year

        # Cost Calculations
        savings_transports = transports_prevented_year * 3800  # $3,800 per transport averted
        savings_911_non_transport = (
            non_transport_calls * 1900
        )  # $1,900 per 911 call without transport

        # ED Visits Calculations
        patients_used_ed = int(total_patients * 0.51)  # 51% of patients used ED
        ed_visits_avoided_year = int(patients_used_ed * 0.69)  # 69% had reduction in ED usage
        savings_ed_visits = ed_visits_avoided_year * 1146  # $1,146 per ED visit avoided

        # Total yearly savings
        yearly_savings = savings_transports + savings_911_non_transport + savings_ed_visits

        # Accumulate totals
        total_calls_prevented += calls_prevented_year
        total_transports_averted += transports_prevented_year
        total_ed_visits_avoided += ed_visits_avoided_year
        total_savings += yearly_savings

        # Append per-year details for template table
        yearly_breakdown.append(
            {
                "year": year_data["year"],
                "patients": total_patients,
                "calls_prevented": calls_prevented_year,
                "transports_averted": transports_prevented_year,
                "ed_visits_avoided": ed_visits_avoided_year,
                "total_savings": yearly_savings,
            }
        )

    # Compute year-over-year percentage change on total_savings
    prev_total = None
    for row in yearly_breakdown:
        if prev_total is None:
            row["pct_change_savings"] = None
        else:
            row["pct_change_savings"] = (
                ((row["total_savings"] - prev_total) / prev_total) * 100 if prev_total > 0 else None
            )
        prev_total = row["total_savings"]

    return {
        "calls_prevented": total_calls_prevented,
        "transports_averted": total_transports_averted,
        "ed_visits_avoided": total_ed_visits_avoided,
        "total_savings": total_savings,
        "savings_breakdown": {
            "transports": total_transports_averted * 3800,
            "911_non_transport": (total_calls_prevented - total_transports_averted) * 1900,
            "ed_visits": total_ed_visits_avoided * 1146,
        },
        "total_patients": sum(year_data["total_patients"] for year_data in patients_by_year),
        "years_calculated": len(patients_by_year),
        "yearly_breakdown": yearly_breakdown,
        # Calculated per-patient metrics for template
        "transport_savings_per_patient": (total_transports_averted * 3800)
        / sum(year_data["total_patients"] for year_data in patients_by_year)
        if sum(year_data["total_patients"] for year_data in patients_by_year) > 0
        else 0,
        "prevention_success_rate": (
            total_calls_prevented
            / sum(year_data["total_patients"] for year_data in patients_by_year)
        )
        * 100
        if sum(year_data["total_patients"] for year_data in patients_by_year) > 0
        else 0,
    }


def get_od_fatality_rate_year(
    year: int, fatal_dispositions: list[str], copa_population: int = 20_000
) -> float:
    """
    Returns the fatal overdose referral rate per 100 000 population for `year`,
    where 'disposition' contains any of the provided strings.
    """
    if not copa_population:
        return 0.0

    # Build a Q object matching any of the dispositions
    q_filter = Q()
    for disp in fatal_dispositions:
        q_filter |= Q(disposition__icontains=disp)

    fatal_count = ODReferrals.objects.filter(od_date__year=year).filter(q_filter).count()

    return (fatal_count / copa_population) * 100_000


def _mapping_to_df(mapping: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, count in mapping.items():
        parts = label.split()
        if len(parts) != 2:
            continue
        year_str, quarter = parts
        try:
            year = int(year_str)
        except ValueError:
            continue
        rows.append({"year": year, "quarter": quarter, "count": int(count)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    df["quarter"] = pd.Categorical(df["quarter"], categories=quarter_order, ordered=True)
    return df.sort_values(["year", "quarter"]).reset_index(drop=True)


def _parse_year_quarter(label: str) -> tuple[int, int] | None:
    parts = label.split()
    if len(parts) != 2:
        return None
    year_str, quarter_str = parts
    if not quarter_str.startswith("Q"):
        return None
    try:
        year = int(year_str)
        quarter = int(quarter_str[1:])
    except ValueError:
        return None
    if quarter not in {1, 2, 3, 4}:
        return None
    return year, quarter


def _build_dynamic_quarter_map(patient_dates: list[dict[str, object]]) -> dict[str, int]:
    if not patient_dates:
        return {}

    df = pd.DataFrame.from_records(patient_dates)
    if df.empty:
        return {}

    for col in ["created_date", "modified_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    created_series = df.get("created_date")
    if created_series is None:
        created_series = pd.Series([], dtype="datetime64[ns]")

    if "modified_date" in df.columns:
        modified_series = df["modified_date"]
        created_series = (
            modified_series if created_series.empty else created_series.fillna(modified_series)
        )

    created_series = created_series.dropna()
    if created_series.empty:
        return {}

    df_valid = pd.DataFrame({"date": created_series})
    df_valid["year"] = df_valid["date"].dt.year.astype(int)
    df_valid["quarter"] = df_valid["date"].dt.quarter.astype(int)

    grouped = df_valid.groupby(["year", "quarter"], dropna=False).size().reset_index(name="count")
    grouped["quarter_label"] = grouped["quarter"].apply(lambda q: f"Q{int(q)}")

    return {
        f"{int(row['year'])} {row['quarter_label']}": int(row["count"])
        for _, row in grouped.iterrows()
    }


def get_quarterly_patient_counts() -> dict:
    """Quarterly patient counts sourced from the Patients table with static fallbacks."""

    static_defaults = {
        "2021 Q1": 47,
        "2021 Q2": 121,
        "2021 Q3": 141,
        "2021 Q4": 110,
        "2022 Q1": 166,
        "2022 Q2": 225,
        "2022 Q3": 172,
        "2022 Q4": 185,
        "2023 Q1": 110,
        "2023 Q2": 54,
        "2023 Q3": 106,
        "2023 Q4": 126,
        "2024 Q1": 91,
        "2024 Q2": 117,
        "2024 Q3": 104,
        "2024 Q4": 99,
        "2025 Q1": 165,
        "2025 Q2": 133,
    }

    try:
        patient_dates = list(Patients.objects.all().values("created_date", "modified_date"))
    except Exception:
        patient_dates = []

    dynamic_map = _build_dynamic_quarter_map(patient_dates)

    merged = static_defaults.copy()
    if dynamic_map:
        cutoff = (2025, 2)
        for label, count in dynamic_map.items():
            parsed = _parse_year_quarter(label)
            if parsed is None:
                continue
            if label not in merged or parsed > cutoff:
                merged[label] = count

    df = _mapping_to_df(merged)
    if not df.empty:
        merged = {
            f"{int(record['year'])} {record['quarter']}": int(record["count"])
            for record in df.to_dict("records")
        }

    return {"mapping": merged, "df": df}
