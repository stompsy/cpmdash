import json
from contextlib import suppress
from datetime import date
from pathlib import Path
from typing import TypedDict

import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET

from apps.accounts.forms import ProfileForm
from utils.theme import get_theme_from_request

from ..charts.od_utils import get_quarterly_patient_counts
from ..charts.odreferrals.narcan_charts import (
    build_narcan_administration_grid,
    build_narcan_response_timeline,
    build_narcan_stats,
)
from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts
from ..charts.overdose.od_all_cases_scatter import (
    build_chart_all_cases_scatter,  # noqa: F401 - re-export for tests monkeypatch
)
from ..charts.overdose.od_density_heatmap import (  # noqa: F401 - re-export for tests monkeypatch
    build_chart_od_density_heatmap,
)
from ..charts.overdose.od_hourly_breakdown import (  # noqa: F401 - re-export for tests monkeypatch
    build_chart_day_of_week_totals,
    build_chart_od_hourly_breakdown,
)
from ..charts.overdose.od_map import build_chart_od_map
from ..charts.overdose.od_repeats_scatter import (  # noqa: F401 - re-export for tests monkeypatch
    build_chart_repeats_scatter,
)
from ..charts.overdose.od_shift_scenarios import (  # noqa: F401 - re-export for tests monkeypatch
    build_chart_cost_benefit_analysis,
    build_chart_shift_scenarios,
    calculate_coverage_scenarios,
)
from ..charts.patients.age_chart_variations import build_all_age_chart_variations
from ..charts.patients.patient_field_charts import build_patients_field_charts
from ..charts.referral.referrals_field_charts import build_referrals_field_charts
from ..core.models import Encounters, ODReferrals, Patients, Referrals

OD_HOTSPOT_CONTEXT_PATH = (
    Path(settings.BASE_DIR) / "src" / "static" / "data" / "od_hotspot_context.json"
)
HARGROVE_METRICS_PATH = (
    Path(settings.BASE_DIR)
    / "src"
    / "static"
    / "data"
    / "hargrove_grant"
    / "historical_metrics.json"
)


def _load_hotspot_context() -> list[dict[str, str]]:
    try:
        with OD_HOTSPOT_CONTEXT_PATH.open(encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []
    metrics = data.get("metrics", []) if isinstance(data, dict) else []
    return [metric for metric in metrics if isinstance(metric, dict)]


class EncounterTableRow(TypedDict):
    label: str
    referral_count: int
    share_pct: float
    engaged_count: int
    engagement_pct: float
    port_count: int
    port_pct: float
    avg_touchpoints: float | None


class EncounterTypeTable(TypedDict):
    title: str
    subtitle: str
    rows: list[EncounterTableRow]


class ReferralTypeTable(TypedDict):
    title: str
    subtitle: str
    rows: list[EncounterTableRow]


def overview(request):
    return redirect("home")


# Patients
def _compute_age_insights(df_all: pd.DataFrame) -> dict[str, object] | None:
    try:
        ages_df = (
            df_all[["age"]] if not df_all.empty and "age" in df_all.columns else pd.DataFrame()
        )
        if ages_df.empty:
            return None
        ages = pd.to_numeric(ages_df["age"], errors="coerce")
        total = int(len(ages_df))
        valid = ages.dropna()
        total_valid = int(valid.size)
        bins = [-1, 17, 24, 34, 44, 54, 64, 74, 84, float("inf")]
        labels = ["0–17", "18–24", "25–34", "35–44", "45–54", "55–64", "65–74", "75–84", "85+"]
        grouped = pd.cut(valid, bins=bins, labels=labels, include_lowest=True, right=True)
        counts = grouped.value_counts().reindex(labels, fill_value=0)
        unknown = total - total_valid
        median_age = (
            int(valid.median()) if total_valid > 0 and not pd.isna(valid.median()) else None
        )
        pct_65_plus = (
            round((valid[valid >= 65].size / total_valid) * 100, 1) if total_valid else 0.0
        )
        pct_75_plus = (
            round((valid[valid >= 75].size / total_valid) * 100, 1) if total_valid else 0.0
        )
        largest_group_label = str(counts.idxmax()) if total_valid else None
        largest_group_pct = (
            round((int(counts.max()) / total_valid) * 100, 1)
            if total_valid and counts.max() > 0
            else 0.0
        )
        return {
            "total": total,
            "total_valid": total_valid,
            "unknown": unknown,
            "median_age": median_age,
            "pct_65_plus": pct_65_plus,
            "pct_75_plus": pct_75_plus,
            "largest_group_label": largest_group_label,
            "largest_group_pct": largest_group_pct,
        }
    except Exception:
        return None


def _insights_age_list(age_insights: dict[str, object] | None) -> list[dict[str, object]] | None:
    if not age_insights:
        return None
    ai = age_insights
    items: list[dict[str, object]] = []
    if ai.get("median_age") is not None:
        items.append({"label": "Median age", "value": _to_int(ai["median_age"])})
    # Interquartile range if available from distribution: approximate using percentiles of valid
    # Note: We didn't store quartiles in age_insights; compute quickly here is out of scope—skip unless added later.
    items.append({"label": "65+ years", "value": f"{ai.get('pct_65_plus', 0.0)}%"})
    items.append({"label": "75+ years", "value": f"{ai.get('pct_75_plus', 0.0)}%"})
    if ai.get("largest_group_label"):
        items.append(
            {
                "label": "Largest band",
                "value": f"{ai['largest_group_label']} ({ai.get('largest_group_pct', 0.0)}%)",
            }
        )
    total = _to_int(ai.get("total", 0))
    total_valid = _to_int(ai.get("total_valid", 0))
    valid_rate = round((total_valid / total) * 100, 1) if total else 0.0
    items.append({"label": "Valid age rate", "value": f"{valid_rate}%"})
    unknown_v = _to_int(ai.get("unknown", 0))
    if unknown_v > 0:
        items.append({"label": "Unknown ages", "value": unknown_v})
    return items


def _pct(n: int, d: int) -> float:
    return round((n / d) * 100, 1) if d else 0.0


def _norm_series(s: pd.Series) -> pd.Series:
    s2 = s.fillna("").astype(str).str.strip()
    return s2.replace({"": "Unknown", "NA": "Unknown", "None": "Unknown"})


def _format_int(value: object | None) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if pd.isna(number):
        return "—"
    return f"{int(round(number)):,}"


def _format_percent(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}%"


def _format_float(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def _format_days(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(round(value))} days"


def _metric(label: str, value: str, description: str | None = None) -> dict[str, str]:
    metric = {"label": label, "value": value}
    if description:
        metric["description"] = description
    return metric


def _to_int(v: object, default: int = 0) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception:
        return default


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return default


def _normalize_text(value: object) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    return text or "Unknown"


def _safe_nunique(series: pd.Series) -> int:
    if series is None:
        return 0
    try:
        return int(series.dropna().nunique())
    except Exception:
        return 0


def _df_from_queryset(values_iterable, expected_columns: list[str] | None = None) -> pd.DataFrame:
    try:
        data = list(values_iterable)
    except Exception:
        data = []
    if not data:
        return pd.DataFrame(columns=expected_columns or [])
    df = pd.DataFrame(data)
    if expected_columns:
        for column in expected_columns:
            if column not in df.columns:
                df[column] = pd.NA
        df = df[expected_columns]
    return df


def _chart_html(chart_result: object) -> str:
    if isinstance(chart_result, tuple):
        first = chart_result[0] if chart_result else ""
        return first if isinstance(first, str) else ""
    if isinstance(chart_result, str):
        return chart_result
    return ""


def _compute_shift_coverage_stats() -> dict[str, float | int]:
    timestamps = list(
        ODReferrals.objects.exclude(od_date__isnull=True).values_list("od_date", flat=True)
    )
    total = len(timestamps)
    if total == 0:
        return {"total": 0, "current": 0.0, "proposed": 0.0, "missed": 0.0}

    current_count = 0
    proposed_count = 0

    for dt in timestamps:
        if dt is None:
            continue
        local_dt = timezone.localtime(dt) if timezone.is_aware(dt) else dt
        weekday = local_dt.weekday()
        hour = local_dt.hour

        in_current = weekday < 5 and 8 <= hour < 16
        in_extended = weekday < 5 and 17 <= hour < 19

        if in_current:
            current_count += 1

        if in_current or in_extended:
            proposed_count += 1

    current_pct = round((current_count / total) * 100, 1) if total else 0.0
    proposed_pct = round((proposed_count / total) * 100, 1) if total else 0.0
    missed_pct = round(max(0.0, 100.0 - current_pct), 1)

    return {
        "total": total,
        "current": current_pct,
        "proposed": proposed_pct,
        "missed": missed_pct,
    }


def _insights_boolean_flag(
    df_all: pd.DataFrame, field: str, label_prefix: str
) -> list[dict[str, object]] | None:
    if field not in df_all.columns:
        return None
    s2 = df_all[field].map({True: "Yes", False: "No"}).fillna("Unknown")
    vc = s2.value_counts()
    total = int(vc.sum())
    if total == 0:
        return None
    yes = int(vc.get("Yes", 0))
    no = int(vc.get("No", 0))
    unk = int(vc.get("Unknown", 0))
    known = yes + no
    prevalence_known = _pct(yes, known) if known else 0.0
    prefix = label_prefix.strip()
    items: list[dict[str, object]] = [
        {"label": f"{prefix} Yes", "value": f"{yes} ({_pct(yes, total)}%)"},
        {"label": f"{prefix} No", "value": f"{no} ({_pct(no, total)}%)"},
        {"label": "Unknown", "value": f"{unk} ({_pct(unk, total)}%)"},
        {"label": "Prevalence (known)", "value": f"{prevalence_known}%"},
    ]
    return items


def _insights_sud(df_all: pd.DataFrame) -> list[dict[str, object]] | None:
    return _insights_boolean_flag(df_all, "sud", "SUD")


def _insights_behavioral_health(df_all: pd.DataFrame) -> list[dict[str, object]] | None:
    return _insights_boolean_flag(df_all, "behavioral_health", "Behavioral health")


def _insights_categorical(
    df_all: pd.DataFrame, field: str, pretty: str | None = None
) -> list[dict[str, object]] | None:
    if field not in df_all.columns:
        return None
    s = _norm_series(df_all[field])
    s = s[~s.str.lower().isin({"not disclosed", "single"})]
    vc = s.value_counts()
    total = int(vc.sum())
    if total == 0:
        return None
    # Top 3 categories
    top3 = vc.head(3)
    items: list[dict[str, object]] = []
    for idx, (label, count) in enumerate(top3.items(), start=1):
        items.append(
            {
                "label": f"Top {idx} {pretty or field.replace('_', ' ').title()}",
                "value": f"{label} ({_pct(_to_int(count), total)}%)",
            }
        )
    coverage = _pct(_to_int(top3.sum()), total)
    items.append({"label": "Top 3 coverage", "value": f"{coverage}%"})
    unk = int(vc.get("Unknown", 0))
    if unk:
        items.append({"label": "Unknown", "value": f"{unk} ({_pct(unk, total)}%)"})
    items.append({"label": "Distinct categories", "value": int(vc.size)})
    return items


def _insights_quarterly() -> list[dict[str, object]] | None:
    try:
        q = get_quarterly_patient_counts()
        qdf = q.get("df")
        if qdf is None or qdf.empty:
            return None
        qdf2 = qdf.sort_values(["year", "quarter"]).reset_index(drop=True)
        latest = qdf2.iloc[-1]
        latest_label = f"{int(latest['year'])} {str(latest['quarter'])}"
        latest_count = int(latest["count"])
        if len(qdf2) >= 2:
            prev = qdf2.iloc[-2]
            prev_count = int(prev["count"])
            delta = latest_count - prev_count
            delta_pct = _pct(delta, prev_count) if prev_count else 0.0
            delta_str = f"{delta:+d} ({delta_pct:+.1f}%)"
        else:
            delta_str = "n/a"
        latest_year = int(latest["year"])
        ytd_total = int(qdf2[qdf2["year"] == latest_year]["count"].sum())
        idxmax = int(qdf2["count"].idxmax())
        peak = qdf2.loc[idxmax]
        peak_label = f"{int(peak['year'])} {str(peak['quarter'])}"
        peak_count = int(peak["count"])
        # last 4 quarters sum
        last4 = int(qdf2.tail(4)["count"].sum()) if len(qdf2) >= 4 else int(qdf2["count"].sum())
        # YoY vs same quarter previous year
        try:
            same_q = qdf2[
                (qdf2["year"] == latest_year - 1) & (qdf2["quarter"] == latest["quarter"])
            ]["count"].iloc[0]
            yoy_delta = latest_count - int(same_q)
            yoy_pct = _pct(yoy_delta, int(same_q)) if int(same_q) else 0.0
            yoy_str = f"{yoy_delta:+d} ({yoy_pct:+.1f}%)"
        except Exception:
            yoy_str = "n/a"
        return [
            {"label": "Latest quarter", "value": f"{latest_label} — {latest_count}"},
            {"label": "QoQ change", "value": delta_str},
            {"label": f"{latest_year} YTD", "value": ytd_total},
            {"label": "12-month volume", "value": last4},
            {"label": "YoY (same quarter)", "value": yoy_str},
            {"label": "Peak quarter", "value": f"{peak_label} — {peak_count}"},
        ]
    except Exception:
        return None


def _build_patient_quick_stats() -> dict[str, list[dict[str, str]]]:
    """Build quick stat cards for patient charts."""
    stats: dict[str, list[dict[str, str]]] = {}

    # Quarterly patient counts stats
    try:
        q = get_quarterly_patient_counts()
        qdf = q.get("df")
        if qdf is not None and not qdf.empty:
            total_patients = int(qdf["count"].sum())
            # Calculate baseline average from 2021, 2023, and 2024 only
            # These are the "normal" years excluding 2022 (Behavioral Health spike) and 2025 (+2 CPMs)
            baseline_years = [2021, 2023, 2024]
            baseline_data = qdf[qdf["year"].isin(baseline_years)]
            avg_per_quarter = round(baseline_data["count"].mean(), 1)
            stats["patient_counts_quarterly"] = [
                {
                    "label": "Total Patients",
                    "value": f"{total_patients:,}",
                    "description": "All-time patient count",
                    "icon": "users",
                },
                {
                    "label": "Baseline Avg",
                    "value": f"{avg_per_quarter}",
                    "description": "Average patients per quarter (baseline years)",
                    "icon": "chart",
                },
            ]
    except Exception:
        pass

    return stats


def _build_referral_quick_stats() -> dict[str, list[dict[str, str]]]:
    """Build quick stat cards for referral charts."""
    stats: dict[str, list[dict[str, str]]] = {}

    # Quarterly referral counts stats
    try:
        from ..core.models import Referrals

        qs = Referrals.objects.all().values("date_received")
        data = list(qs)
        if data:
            df = pd.DataFrame.from_records(data)
            df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

            # Group by year and quarter
            qdf = pd.DataFrame(
                {
                    "year": df["date_received"].dt.year,
                    "quarter": df["date_received"].dt.quarter,
                }
            ).dropna()
            qdf = qdf.groupby(["year", "quarter"], dropna=True).size().reset_index(name="count")

            if not qdf.empty:
                total_referrals = int(qdf["count"].sum())
                # Calculate baseline average from 2021, 2023, and 2024 only
                # These are the "normal" years excluding 2022 (Behavioral Health spike) and 2025 (+2 CPMs)
                baseline_years = [2021, 2023, 2024]
                baseline_data = qdf[qdf["year"].isin(baseline_years)]
                if not baseline_data.empty:
                    avg_per_quarter = round(baseline_data["count"].mean(), 1)
                else:
                    avg_per_quarter = round(qdf["count"].mean(), 1)

                stats["referrals_counts_quarterly"] = [
                    {
                        "label": "Total Referrals",
                        "value": f"{total_referrals:,}",
                        "description": "All-time referral count",
                        "icon": "users",
                    },
                    {
                        "label": "Baseline Avg",
                        "value": f"{avg_per_quarter}",
                        "description": "Average referrals per quarter (baseline years)",
                        "icon": "chart",
                    },
                ]
    except Exception:
        pass

    return stats


def _build_odreferrals_quick_stats() -> dict[str, list[dict[str, str]]]:
    """Build quick stat cards for OD referral charts."""
    stats: dict[str, list[dict[str, str]]] = {}

    # Monthly OD referral counts stats
    try:
        from ..core.models import ODReferrals

        qs = ODReferrals.objects.all().values(
            "od_date", "disposition", "patient_id", "referral_to_sud_agency", "referral_source"
        )
        data = list(qs)
        if data:
            df = pd.DataFrame.from_records(data)
            df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
            df = df.dropna(subset=["od_date"])

            if not df.empty:
                # Total overdoses
                total_overdoses = len(df)

                # Fatal overdoses
                fatal_overdoses = len(df[df["disposition"].isin(["CPR attempted", "DOA"])])

                # Repeat overdoses (patients with more than one overdose)
                repeat_counts = df["patient_id"].value_counts()
                repeat_patients = len(repeat_counts[repeat_counts > 1])
                repeat_overdoses = int(repeat_counts[repeat_counts > 1].sum())
                percent_repeat = (
                    round((repeat_overdoses / total_overdoses) * 100, 1)
                    if total_overdoses > 0
                    else 0
                )

                # Referral success rate (excluding 'Other' referral sources)
                referral_df = df[df["referral_source"] != "Other"]
                if len(referral_df) > 0:
                    successful_referrals = referral_df["referral_to_sud_agency"].sum()
                    total_eligible_referrals = len(referral_df)
                    round((successful_referrals / total_eligible_referrals) * 100, 1)

                stats["odreferrals_counts_monthly"] = [
                    {
                        "label": "Total Overdoses",
                        "value": str(total_overdoses),
                        "description": "All recorded overdose cases",
                        "icon": "chart-bar",
                    },
                    {
                        "label": "Fatal Overdoses",
                        "value": str(fatal_overdoses),
                        "description": "CPR attempted or DOA",
                        "icon": "exclamation-triangle",
                    },
                    {
                        "label": "Repeat Overdoses",
                        "value": str(repeat_overdoses),
                        "description": f"{repeat_patients} patients • {percent_repeat}% repeat",
                        "icon": "refresh",
                    },
                ]

                # Weekday stats (calculate day-of-week patterns)
                df["weekday"] = df["od_date"].dt.day_name()
                weekday_counts = df["weekday"].value_counts()

                # Weekend vs Weekday
                weekend_days = ["Saturday", "Sunday"]
                weekend_count = len(df[df["weekday"].isin(weekend_days)])
                weekday_count = total_overdoses - weekend_count
                weekend_pct = (
                    round((weekend_count / total_overdoses) * 100, 1) if total_overdoses > 0 else 0
                )

                # Busiest day
                busiest_day = weekday_counts.index[0] if len(weekday_counts) > 0 else "N/A"
                busiest_day_count = int(weekday_counts.iloc[0]) if len(weekday_counts) > 0 else 0

                # Fatal overdoses on weekends
                weekend_df = df[df["weekday"].isin(weekend_days)]
                weekend_fatal = len(
                    weekend_df[weekend_df["disposition"].isin(["CPR attempted", "DOA"])]
                )
                weekend_fatal_pct = (
                    round((weekend_fatal / weekend_count) * 100, 1) if weekend_count > 0 else 0
                )

                stats["odreferrals_counts_weekday"] = [
                    {
                        "label": "Weekday Cases",
                        "value": str(weekday_count),
                        "description": f"{100 - weekend_pct}% of all overdoses",
                        "icon": "calendar",
                    },
                    {
                        "label": "Busiest Day",
                        "value": busiest_day,
                        "description": f"{busiest_day_count} overdoses",
                        "icon": "trending-up",
                    },
                    {
                        "label": "Weekend Cases",
                        "value": str(weekend_count),
                        "description": f"{weekend_pct}% of all overdoses",
                        "icon": "calendar",
                    },
                    {
                        "label": "Weekend Fatal Rate",
                        "value": f"{weekend_fatal_pct}%",
                        "description": f"{weekend_fatal} of {weekend_count} cases",
                        "icon": "exclamation-triangle",
                    },
                ]
    except Exception:
        pass

    return stats


def _process_referrals_quarterly(referrals_df: pd.DataFrame) -> pd.DataFrame:
    """Process referrals data into quarterly counts with 2022 Q3 adjustment."""
    if referrals_df.empty or "date_received" not in referrals_df.columns:
        return pd.DataFrame(columns=["year", "quarter", "referrals_count"])

    # Convert to list first to satisfy type checkers, then parse to datetimes and back to Series
    raw_dates = referrals_df["date_received"].tolist()
    dt_index = pd.to_datetime(raw_dates, errors="coerce")
    dates = pd.Series(dt_index).dropna()

    referrals_qdf = pd.DataFrame(
        {
            "year": dates.dt.year,
            "quarter": dates.dt.quarter,
        }
    )
    referrals_qdf = (
        referrals_qdf.groupby(["year", "quarter"], dropna=True)
        .size()
        .reset_index(name="referrals_count")
    )

    # Move 2022 Q3 referral to 2023 Q1 (matching chart logic)
    q3_2022_count = referrals_qdf.loc[
        (referrals_qdf["year"] == 2022) & (referrals_qdf["quarter"] == 3), "referrals_count"
    ].sum()
    if q3_2022_count > 0:
        referrals_qdf = referrals_qdf[
            ~((referrals_qdf["year"] == 2022) & (referrals_qdf["quarter"] == 3))
        ]
        q1_2023_idx = referrals_qdf[
            (referrals_qdf["year"] == 2023) & (referrals_qdf["quarter"] == 1)
        ].index
        if len(q1_2023_idx) > 0:
            referrals_qdf.loc[q1_2023_idx[0], "referrals_count"] += q3_2022_count
        else:
            referrals_qdf = pd.concat(
                [
                    referrals_qdf,
                    pd.DataFrame([{"year": 2023, "quarter": 1, "referrals_count": q3_2022_count}]),
                ],
                ignore_index=True,
            )

    return referrals_qdf


def _process_port_referrals_quarterly(encounters_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and process PORT referrals (port_referral_ID > 0) into quarterly counts."""
    if (
        encounters_df.empty
        or "encounter_date" not in encounters_df.columns
        or "port_referral_ID" not in encounters_df.columns
    ):
        return pd.DataFrame(columns=["year", "quarter", "port_referrals_count"])

    port_referrals_df = encounters_df[encounters_df["port_referral_ID"] > 0].copy()
    if port_referrals_df.empty:
        return pd.DataFrame(columns=["year", "quarter", "port_referrals_count"])

    dates = pd.Series(pd.to_datetime(port_referrals_df["encounter_date"], errors="coerce")).dropna()
    port_qdf = pd.DataFrame(
        {
            "year": dates.dt.year,
            "quarter": dates.dt.quarter,
        }
    )
    return (
        port_qdf.groupby(["year", "quarter"], dropna=True)
        .size()
        .reset_index(name="port_referrals_count")
    )


def _process_regular_encounters_quarterly(encounters_df: pd.DataFrame) -> pd.DataFrame:
    """Process regular encounters (excluding PORT referrals) into quarterly counts."""
    if encounters_df.empty or "encounter_date" not in encounters_df.columns:
        return pd.DataFrame(columns=["year", "quarter", "encounters_count"])

    # Exclude PORT referrals where port_referral_ID > 0
    if "port_referral_ID" in encounters_df.columns:
        regular_encounters_df = encounters_df[encounters_df["port_referral_ID"] == 0].copy()
    else:
        regular_encounters_df = encounters_df

    if regular_encounters_df.empty:
        return pd.DataFrame(columns=["year", "quarter", "encounters_count"])

    regular_encounters_df["encounter_date"] = pd.to_datetime(
        regular_encounters_df["encounter_date"], errors="coerce"
    )
    encounters_qdf = pd.DataFrame(
        {
            "year": regular_encounters_df["encounter_date"].dt.year,
            "quarter": regular_encounters_df["encounter_date"].dt.quarter,
        }
    ).dropna()
    return (
        encounters_qdf.groupby(["year", "quarter"], dropna=True)
        .size()
        .reset_index(name="encounters_count")
    )


def _merge_quarterly_data(
    referrals_qdf: pd.DataFrame, port_qdf: pd.DataFrame, encounters_qdf: pd.DataFrame
) -> pd.DataFrame:
    """Merge referrals, PORT referrals, and encounters data into a single quarterly dataframe."""
    # Get all year-quarter combinations
    all_years = set()
    for df in [referrals_qdf, port_qdf, encounters_qdf]:
        if not df.empty:
            all_years.update(df["year"].unique())

    if not all_years:
        return pd.DataFrame()

    qdf = pd.DataFrame(
        [(y, q) for y in sorted(all_years) for q in [1, 2, 3, 4]], columns=["year", "quarter"]
    )

    # Merge each dataframe
    for df, col in [
        (referrals_qdf, "referrals_count"),
        (port_qdf, "port_referrals_count"),
        (encounters_qdf, "encounters_count"),
    ]:
        if not df.empty:
            qdf = qdf.merge(df, on=["year", "quarter"], how="left")
        else:
            qdf[col] = 0

    # Fill NaN and calculate totals
    qdf["referrals_count"] = qdf["referrals_count"].fillna(0).astype(int)
    qdf["port_referrals_count"] = qdf["port_referrals_count"].fillna(0).astype(int)
    qdf["encounters_count"] = qdf["encounters_count"].fillna(0).astype(int)
    qdf["total_count"] = (
        qdf["referrals_count"] + qdf["port_referrals_count"] + qdf["encounters_count"]
    )

    return qdf


def _build_encounters_quarterly_quick_stats() -> dict[str, list[dict[str, str]]]:
    """Build quick stat cards for encounters quarterly chart."""
    stats: dict[str, list[dict[str, str]]] = {}

    try:
        from ..core.models import Encounters, Referrals

        # Get data
        encounters_data = list(
            Encounters.objects.all().values("encounter_date", "port_referral_ID")
        )
        referrals_data = list(Referrals.objects.all().values("date_received"))

        if not encounters_data and not referrals_data:
            return stats

        # Convert to DataFrames
        encounters_df = (
            pd.DataFrame.from_records(encounters_data) if encounters_data else pd.DataFrame()
        )
        referrals_df = (
            pd.DataFrame.from_records(referrals_data) if referrals_data else pd.DataFrame()
        )

        # Process each data type into quarterly counts
        referrals_qdf = _process_referrals_quarterly(referrals_df)
        port_qdf = _process_port_referrals_quarterly(encounters_df)
        encounters_qdf = _process_regular_encounters_quarterly(encounters_df)

        # Merge all quarterly data
        qdf = _merge_quarterly_data(referrals_qdf, port_qdf, encounters_qdf)

        if qdf.empty:
            return stats

        # Exclude 2022 Q1-Q3 (did not track)
        qdf_tracked = qdf[~((qdf["year"] == 2022) & (qdf["quarter"].isin([1, 2, 3])))]

        if qdf_tracked.empty:
            return stats

        # Calculate statistics
        total_combined = int(qdf_tracked["total_count"].sum())

        # Calculate baseline average from 2021, 2023, and 2024 only
        baseline_years = [2021, 2023, 2024]
        baseline_data = qdf_tracked[qdf_tracked["year"].isin(baseline_years)]
        avg_per_quarter = (
            round(baseline_data["total_count"].mean(), 1)
            if not baseline_data.empty
            else round(qdf_tracked["total_count"].mean(), 1)
        )

        stats["encounters_counts_quarterly"] = [
            {
                "label": "Total Encounters",
                "value": f"{total_combined:,}",
                "description": "Encounters + Referrals + PORT Referrals",
                "icon": "activity",
            },
            {
                "label": "Baseline Avg",
                "value": f"{avg_per_quarter}",
                "description": "Average per quarter (baseline years)",
                "icon": "chart",
            },
        ]
    except Exception:
        pass

    return stats


def _insights_boxplot_by(df_all: pd.DataFrame, group_field: str) -> list[dict[str, object]] | None:
    if not {"age", group_field}.issubset(df_all.columns):
        return None
    gp = df_all.dropna(subset=[group_field]).copy()
    gp["age_num"] = pd.to_numeric(gp["age"], errors="coerce")
    gp = gp.dropna(subset=["age_num"]) if not gp.empty else gp
    if gp is None or gp.empty:
        return None
    med = gp.groupby(group_field)["age_num"].median().sort_values(ascending=False)
    if med.empty:
        return None
    max_cat, max_val = med.index[0], float(med.iloc[0])
    items: list[dict[str, object]] = []
    ages = pd.to_numeric(df_all["age"], errors="coerce").dropna()
    overall_median = float(ages.median()) if not ages.empty else None
    if overall_median is not None:
        items.append({"label": "Overall median age", "value": int(round(overall_median))})
    items.append({"label": "Highest median", "value": f"{max_cat} ({int(round(max_val))})"})
    if med.size > 1:
        min_cat, min_val = med.index[-1], float(med.iloc[-1])
        diff = max_val - min_val
        items.append({"label": "Lowest median", "value": f"{min_cat} ({int(round(min_val))})"})
        items.append({"label": "Gap between groups", "value": f"{int(round(diff))} years"})
    return items


def _insights_age_referral_sankey() -> list[dict[str, object]] | None:  # noqa: C901
    """Generate insights for the age → referral type Sankey diagram."""
    try:
        # Get referrals count
        referrals_qs = Referrals.objects.all().values(
            "referral_1", "referral_2", "referral_3", "referral_4", "referral_5"
        )
        referrals_data = list(referrals_qs)

        # Get OD referrals count
        od_count = ODReferrals.objects.count()

        if not referrals_data and od_count == 0:
            return None

        # Flatten all referral columns
        all_referrals = []
        for ref in referrals_data:
            for i in range(1, 6):
                val = ref.get(f"referral_{i}")
                if val and str(val).strip() and str(val).lower() not in {"nan", "none", ""}:
                    all_referrals.append(str(val).strip())

        # Count specific referral types shown in Sankey
        specific_types = {
            "911 calls": 0,
            "Vaccinations": 0,
            "Lab - Blood Draw": 0,
            "Eval - Assessment": 0,
            "Case Management": 0,
            "Eval - Psych/Dementia/Crisis": 0,
            "Overdose": od_count,  # Start with OD referrals
            "Med - Rx Reconciliation": 0,
            "Med - Antipsychotic IM": 0,
        }

        for ref in all_referrals:
            ref_lower = ref.lower().strip()
            if "911" in ref_lower or "walk-in" in ref_lower or "walk in" in ref_lower:
                specific_types["911 calls"] += 1
            elif "vax -" in ref_lower or "vax-" in ref_lower or ref_lower.startswith("vax "):
                specific_types["Vaccinations"] += 1
            elif "lab - blood draw" in ref_lower or "lab-blood draw" in ref_lower:
                specific_types["Lab - Blood Draw"] += 1
            elif "eval - assessment" in ref_lower or "eval-assessment" in ref_lower:
                specific_types["Eval - Assessment"] += 1
            elif "case management" in ref_lower:
                specific_types["Case Management"] += 1
            elif "eval - psych" in ref_lower or "dementia" in ref_lower or "crisis" in ref_lower:
                specific_types["Eval - Psych/Dementia/Crisis"] += 1
            elif "overdose" in ref_lower or ref_lower == "od":
                specific_types["Overdose"] += 1
            elif "med - rx reconciliation" in ref_lower or "med-rx reconciliation" in ref_lower:
                specific_types["Med - Rx Reconciliation"] += 1
            elif "med - antipsychotic im" in ref_lower or "antipsychotic" in ref_lower:
                specific_types["Med - Antipsychotic IM"] += 1

        # Get top referral type
        total_specific = sum(specific_types.values())
        if total_specific == 0:
            return None

        top_type = max(specific_types.items(), key=lambda x: x[1])
        top_pct = (top_type[1] / total_specific * 100) if total_specific else 0
        types_shown = sum(1 for count in specific_types.values() if count > 0)

        return [
            {"label": "Total referrals shown", "value": total_specific},
            {"label": "Service types displayed", "value": types_shown},
            {"label": "Top service pathway", "value": f"{top_type[0]} ({top_pct:.1f}%)"},
            {
                "label": "Overdose referrals",
                "value": f"{specific_types['Overdose']} (combined sources)",
            },
        ]
    except Exception:
        return None


def _insights_age_gender_pyramid() -> list[dict[str, object]] | None:
    """Generate insights for the age/gender population pyramid."""
    try:
        patients_qs = Patients.objects.all().values("age", "sex")
        patients_data = list(patients_qs)
        if not patients_data:
            return None

        df = pd.DataFrame.from_records(patients_data)

        # Count by gender
        total = len(df)
        male_count = len(df[df["sex"].str.lower().isin(["male", "m"])])
        female_count = len(df[df["sex"].str.lower().isin(["female", "f"])])
        other_count = total - male_count - female_count

        male_pct = (male_count / total * 100) if total else 0
        female_pct = (female_count / total * 100) if total else 0

        # Calculate age statistics
        ages = pd.to_numeric(df["age"], errors="coerce").dropna()
        if len(ages) == 0:
            return None

        median_age = ages.median()

        return [
            {"label": "Total patients", "value": total},
            {"label": "Male patients", "value": f"{male_count} ({male_pct:.1f}%)"},
            {"label": "Female patients", "value": f"{female_count} ({female_pct:.1f}%)"},
            {"label": "Other/Unknown", "value": other_count},
            {"label": "Median age", "value": f"{median_age:.0f} years"},
        ]
    except Exception:
        return None


def _build_patients_chart_insights(  # noqa: C901
    df_all: pd.DataFrame, age_insights: dict[str, object] | None
) -> dict[str, list[dict[str, object]]]:
    if df_all.empty:
        return {}
    insights: dict[str, list[dict[str, object]]] = {}
    age_list = _insights_age_list(age_insights)
    if age_list:
        insights["age"] = age_list
    sud_list = _insights_sud(df_all)
    if sud_list:
        insights["sud"] = sud_list
    behavioral_health_list = _insights_behavioral_health(df_all)
    if behavioral_health_list:
        insights["behavioral_health"] = behavioral_health_list
    for f, label in [
        ("insurance", None),
        ("pcp_agency", "Primary Care Agency"),
        ("zip_code", "ZIP Code"),
        ("marital_status", None),
        ("veteran_status", None),
    ]:
        cat = _insights_categorical(df_all, f, label)
        if cat:
            insights[f] = cat
    q_list = _insights_quarterly()
    if q_list:
        insights["patient_counts_quarterly"] = q_list
    sex_box = _insights_boxplot_by(df_all, "sex")
    if sex_box:
        insights["sex_age_boxplot"] = sex_box
    race_box = _insights_boxplot_by(df_all, "race")
    if race_box:
        insights["race_age_boxplot"] = race_box
    sankey_list = _insights_age_referral_sankey()
    if sankey_list:
        insights["age_referral_sankey"] = sankey_list
    return insights


def _format_bool_label(value: object) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unknown"
    if isinstance(value, (int, float)) and not pd.isna(value):  # noqa: UP038
        if int(value) == 1:
            return "Yes"
        if int(value) == 0:
            return "No"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "unknown", "na", "none"}:
            return "Unknown"
        if text in {"yes", "y", "true", "1"}:
            return "Yes"
        if text in {"no", "n", "false", "0"}:
            return "No"
    return "Unknown"


def _load_patient_touchpoint_datasets() -> dict[str, pd.DataFrame]:
    patient_fields = [
        "id",
        "age",
        "insurance",
        "pcp_agency",
        "race",
        "sex",
        "sud",
        "behavioral_health",
        "zip_code",
        "marital_status",
        "veteran_status",
    ]
    patients_df = _df_from_queryset(Patients.objects.all().values(*patient_fields), patient_fields)

    datasets = _load_referral_datasets()
    datasets["patients"] = patients_df
    return datasets


def _enrich_patient_touchpoints(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    patients_df = datasets.get("patients", pd.DataFrame()).copy()
    if patients_df.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "age",
                "insurance",
                "pcp_agency",
                "race",
                "sex",
                "sud",
                "behavioral_health",
                "zip_code",
                "marital_status",
                "veteran_status",
                "referrals",
                "encounters",
                "odreferrals",
                "total_touchpoints",
                "has_touchpoint",
                "has_encounter",
                "has_od",
            ]
        )

    patients_df = patients_df.rename(columns={"id": "patient_id"})
    patients_df["patient_id"] = pd.to_numeric(patients_df["patient_id"], errors="coerce").astype(
        "Int64"
    )
    patients_df = patients_df.dropna(subset=["patient_id"]).copy()
    patients_df["patient_id"] = patients_df["patient_id"].astype(int)

    referrals_df = datasets.get("referrals", pd.DataFrame())
    encounters_df = datasets.get("encounters", pd.DataFrame())
    od_df = datasets.get("odreferrals", pd.DataFrame())

    def _counts_by(df: pd.DataFrame, column: str) -> pd.Series:
        if df.empty or column not in df.columns:
            return pd.Series(dtype="int64")
        return df.dropna(subset=[column]).groupby(column, dropna=False).size().astype(int)

    referral_counts = _counts_by(referrals_df, "patient_ID")
    encounter_counts = _counts_by(encounters_df, "patient_ID")
    od_counts = _counts_by(od_df, "patient_id")

    patients_df["referrals"] = patients_df["patient_id"].map(referral_counts).fillna(0).astype(int)
    patients_df["encounters"] = (
        patients_df["patient_id"].map(encounter_counts).fillna(0).astype(int)
    )
    patients_df["odreferrals"] = patients_df["patient_id"].map(od_counts).fillna(0).astype(int)
    patients_df["total_touchpoints"] = (
        patients_df["referrals"] + patients_df["encounters"] + patients_df["odreferrals"]
    )
    patients_df["has_touchpoint"] = patients_df["total_touchpoints"] > 0
    patients_df["has_encounter"] = patients_df["encounters"] > 0
    patients_df["has_od"] = patients_df["odreferrals"] > 0

    return patients_df


def _build_patient_summary_cards(enriched: pd.DataFrame) -> list[dict[str, object]]:
    total_patients = int(enriched.shape[0])
    active_df = enriched[enriched["has_touchpoint"]]
    active_count = int(active_df.shape[0])
    single_touch_count = int(active_df[active_df["total_touchpoints"] == 1].shape[0])
    encounter_patients = active_df[active_df["has_encounter"]]
    encounter_count = int(encounter_patients.shape[0])
    high_touch_df = active_df[active_df["total_touchpoints"] >= 6]
    high_touch_count = int(high_touch_df.shape[0])
    avg_touchpoints = (
        round(float(active_df["total_touchpoints"].mean()), 1) if not active_df.empty else 0.0
    )
    od_linked_df = active_df[active_df["has_od"]]
    od_linked_count = int(od_linked_df.shape[0])

    return [
        {
            "label": "Patients tracked",
            "value": f"{total_patients:,}",
            "description": "Individuals in the CPM registry with demographic context for outreach.",
        },
        {
            "label": "Active with touchpoints",
            "value": f"{active_count:,}",
            "description": (
                f"{_pct(active_count, total_patients)}% engaged; {single_touch_count} are single-touch cases to revisit."
            ),
        },
        {
            "label": "Encounter-engaged",
            "value": f"{encounter_count:,}",
            "description": (
                f"{_pct(encounter_count, active_count)}% of engaged patients have at least one CPM encounter."
            ),
        },
        {
            "label": "High-touch cohort",
            "value": f"{high_touch_count:,}",
            "description": (
                f"6+ combined referrals, encounters, or PORT contacts; avg {avg_touchpoints:.1f} touchpoints overall."
            ),
        },
        {
            "label": "PORT-connected patients",
            "value": f"{od_linked_count:,}",
            "description": (
                f"{_pct(od_linked_count, active_count)}% of engaged patients include overdose follow-up touchpoints."
            ),
        },
    ]


def _build_patient_top_patients(enriched: pd.DataFrame) -> list[dict[str, object]]:
    active_df = enriched[enriched["has_touchpoint"]]
    if active_df.empty:
        return []

    sorted_df = active_df.sort_values(
        ["total_touchpoints", "encounters", "referrals", "patient_id"],
        ascending=[False, False, False, True],
    ).head(10)

    rows: list[dict[str, object]] = []
    for _, row in sorted_df.iterrows():
        age_value: object
        age_raw = row.get("age")
        if age_raw is None or (isinstance(age_raw, float) and pd.isna(age_raw)):
            age_value = "Unknown"
        else:
            try:
                age_value = int(age_raw)
            except Exception:
                age_value = "Unknown"

        rows.append(
            {
                "patient_id": int(row["patient_id"]),
                "total_touchpoints": int(row["total_touchpoints"]),
                "referrals": int(row["referrals"]),
                "encounters": int(row["encounters"]),
                "insurance": _normalize_text(row.get("insurance")),
                "pcp_agency": _normalize_text(row.get("pcp_agency")),
                "age": age_value,
                "sex": _normalize_text(row.get("sex")),
                "sud": _format_bool_label(row.get("sud")),
                "behavioral_health": _format_bool_label(row.get("behavioral_health")),
            }
        )
    return rows


def _patient_examples(df: pd.DataFrame, limit: int = 3) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    if df.empty:
        return examples
    subset = df.sort_values(
        ["total_touchpoints", "encounters", "referrals", "patient_id"],
        ascending=[False, False, False, True],
    ).head(limit)
    for _, row in subset.iterrows():
        examples.append(
            {
                "patient_id": int(row["patient_id"]),
                "touchpoints": int(row["total_touchpoints"]),
                "referrals": int(row["referrals"]),
                "encounters": int(row["encounters"]),
            }
        )
    return examples


def _build_patient_insight_sections(enriched: pd.DataFrame) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    active_df = enriched[enriched["has_touchpoint"]]
    if active_df.empty:
        return sections

    high_touch_df = active_df[active_df["total_touchpoints"] >= 6]
    if not high_touch_df.empty:
        share_pct = _pct(int(high_touch_df.shape[0]), int(active_df.shape[0]))
        median_encounters = high_touch_df["encounters"].median()
        median_encounters_value = int(median_encounters) if not pd.isna(median_encounters) else 0
        sections.append(
            {
                "tag": "High-touch",
                "title": "Coordinate multi-touch care plans",
                "summary": (
                    f"{int(high_touch_df.shape[0])} patients log 6+ touchpoints, representing {share_pct}% of engaged cases."
                ),
                "details": [
                    f"Median CPM encounters: {median_encounters_value}",
                    f"Average total touchpoints: {round(float(high_touch_df['total_touchpoints'].mean()), 1)}",
                ],
                "examples": _patient_examples(high_touch_df),
            }
        )

    encounter_only_df = active_df[(active_df["encounters"] > 0) & (active_df["referrals"] == 0)]
    if not encounter_only_df.empty:
        sections.append(
            {
                "tag": "Encounter-only",
                "title": "Close the referral loop",
                "summary": (
                    f"{int(encounter_only_df.shape[0])} patients receive CPM encounters without a logged referral."
                ),
                "details": [
                    f"Average encounters: {round(float(encounter_only_df['encounters'].mean()), 1)}",
                    f"{_pct(int(encounter_only_df['has_od'].sum()), int(encounter_only_df.shape[0]))}% also have PORT follow-ups.",
                ],
                "examples": _patient_examples(encounter_only_df),
            }
        )

    od_linked_df = active_df[active_df["has_od"]]
    if not od_linked_df.empty:
        od_share = _pct(int(od_linked_df.shape[0]), int(active_df.shape[0]))
        sections.append(
            {
                "tag": "PORT",
                "title": "PORT follow-ups driving engagement",
                "summary": (
                    f"{int(od_linked_df.shape[0])} patients link overdose follow-ups with CPM touchpoints ({od_share}% of engaged)."
                ),
                "details": [
                    f"Average total touchpoints: {round(float(od_linked_df['total_touchpoints'].mean()), 1)}",
                    f"Referrals accompany PORT work in {_pct(int((od_linked_df['referrals'] > 0).sum()), int(od_linked_df.shape[0]))}% of these cases.",
                ],
                "examples": _patient_examples(od_linked_df),
            }
        )

    return sections


def top_engaged_patients(request):
    """Return just the top engaged patients table for display in modals/popovers."""
    datasets = _load_patient_touchpoint_datasets()
    patients_df = datasets.get("patients", pd.DataFrame())

    if patients_df.empty:
        context = {"top_patients": []}
    else:
        enriched = _enrich_patient_touchpoints(datasets)
        if enriched.empty or not enriched["has_touchpoint"].any():
            context = {"top_patients": []}
        else:
            top_patients = _build_patient_top_patients(enriched)
            context = {"top_patients": top_patients}

    return render(request, "dashboard/partials/top_engaged_patients.html", context)


def referral_types_table(request):
    """Return combined referral types breakdown table for display in modal."""
    df_all = _load_referrals_dataframe()

    if df_all.empty:
        context = {"referral_types": [], "total_types": 0, "total_referrals": 0}
    else:
        referral_fields = ["referral_1", "referral_2", "referral_3", "referral_4", "referral_5"]

        # Collect all values across all referral fields
        all_values = []

        # Values to filter out
        excluded_values = {"911/ED HU - Q1 2023", "911/ED HU - Q4 2022"}

        for field in referral_fields:
            if field in df_all.columns:
                s = df_all[field].fillna("").astype(str).str.strip()
                s = s.replace({"": "Unknown"})
                s = s[s != "Unknown"]
                # Filter out excluded values
                s = s[~s.isin(excluded_values)]
                all_values.extend(s.tolist())

        # Calculate combined totals across all fields
        if all_values:
            combined_series = pd.Series(all_values)
            combined_vc = combined_series.value_counts().reset_index()
            combined_vc.columns = ["type", "count"]
            total_count = combined_vc["count"].sum()
            combined_vc["percentage"] = (
                (combined_vc["count"] / total_count * 100).round(1) if total_count > 0 else 0
            )

            # Extract group from first word of type
            combined_vc["group"] = combined_vc["type"].str.split().str[0]

            # Sort by group, then by count descending within each group
            combined_vc = combined_vc.sort_values(
                by=["group", "count"], ascending=[True, False]
            ).reset_index(drop=True)

            # Assign color index to each group (cycling through color palette)
            unique_groups = combined_vc["group"].unique()
            group_to_color = {group: idx % 8 for idx, group in enumerate(unique_groups)}
            combined_vc["group_color"] = combined_vc["group"].map(group_to_color)

            referral_types = combined_vc.to_dict("records")
            total_types = len(combined_vc)
            total_referrals = int(total_count)
        else:
            referral_types = []
            total_types = 0
            total_referrals = 0

        context = {
            "referral_types": referral_types,
            "total_types": total_types,
            "total_referrals": total_referrals,
        }

    return render(request, "dashboard/partials/referral_types_table.html", context)


# Referrals insights helpers
def _build_referrals_quarterly_insights(df_all: pd.DataFrame) -> list[dict[str, object]] | None:
    if df_all.empty or "date_received" not in df_all.columns:
        return None
    try:
        dates = pd.to_datetime(df_all["date_received"], errors="coerce")
        qdf = (
            pd.DataFrame(
                {
                    "year": dates.dt.year,
                    "quarter": "Q" + dates.dt.quarter.astype("Int64").astype(str),
                }
            )
            .dropna()
            .groupby(["year", "quarter"], dropna=True)
            .size()
            .reset_index(name="count")
        )
        if qdf.empty:
            return None
        qdf2 = qdf.sort_values(["year", "quarter"]).reset_index(drop=True)
        latest = qdf2.iloc[-1]
        latest_label = f"{int(latest['year'])} {str(latest['quarter'])}"
        latest_count = int(latest["count"])
        if len(qdf2) >= 2:
            prev = qdf2.iloc[-2]
            prev_count = int(prev["count"])
            delta = latest_count - prev_count
            delta_pct = _pct(delta, prev_count) if prev_count else 0.0
            delta_str = f"{delta:+d} ({delta_pct:+.1f}%)"
        else:
            delta_str = "n/a"
        latest_year = int(latest["year"])
        ytd_total = int(qdf2[qdf2["year"] == latest_year]["count"].sum())
        peak_idx = int(qdf2["count"].idxmax())
        peak_year = int(qdf2.loc[peak_idx, "year"])  # type: ignore[arg-type]
        peak_quarter = str(qdf2.loc[peak_idx, "quarter"])  # type: ignore[arg-type]
        peak_count = int(qdf2.loc[peak_idx, "count"])  # type: ignore[arg-type]
        peak_label = f"{peak_year} {peak_quarter}"
        last4 = int(qdf2.tail(4)["count"].sum()) if len(qdf2) >= 4 else int(qdf2["count"].sum())
        try:
            same_q = qdf2[
                (qdf2["year"] == latest_year - 1) & (qdf2["quarter"] == latest["quarter"])
            ]["count"].iloc[0]
            yoy_delta = latest_count - int(same_q)
            yoy_pct = _pct(yoy_delta, int(same_q)) if int(same_q) else 0.0
            yoy_str = f"{yoy_delta:+d} ({yoy_pct:+.1f}%)"
        except Exception:
            yoy_str = "n/a"
        return [
            {"label": "Latest quarter", "value": f"{latest_label} — {latest_count}"},
            {"label": "QoQ change", "value": delta_str},
            {"label": f"{latest_year} YTD", "value": ytd_total},
            {"label": "12-month volume", "value": last4},
            {"label": "YoY (same quarter)", "value": yoy_str},
            {"label": "Peak quarter", "value": f"{peak_label} — {peak_count}"},
        ]
    except Exception:
        return None


def _build_referrals_chart_insights(df_all: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    if df_all.empty:
        return {}
    insights: dict[str, list[dict[str, object]]] = {}
    age_insights = _compute_age_insights(df_all)
    age_list = _insights_age_list(age_insights)
    if age_list:
        insights["age"] = age_list
    for f, label in [
        ("insurance", None),
        ("referral_agency", "Referral Agency"),
        ("zipcode", "ZIP Code"),
        ("referral_closed_reason", "Closed Reason"),
        ("sex", None),
        ("encounter_type_cat1", "Encounter Type (1)"),
        ("encounter_type_cat2", "Encounter Type (2)"),
        ("encounter_type_cat3", "Encounter Type (3)"),
    ]:
        tips = _insights_categorical(df_all, f, label)
        if tips:
            insights[f] = tips
    rq = _build_referrals_quarterly_insights(df_all)
    if rq:
        insights["referrals_counts_quarterly"] = rq
    return insights


REFERRALS_RATIONALE_HEADINGS: dict[str, str] = {
    "age": "Vulnerability Mapping",
    "insurance": "Financial Reality Check",
    "referral_agency": "Relationship Health Monitor",
    "zipcode": "Geographic Access Analysis",
    "referral_closed_reason": "Intake Friction Report",
    "sex": "Access Equity Check",
    "encounter_type_cat1": "Core Function Analysis",
    "encounter_type_cat2": "Clinical Complexity Drill-Down",
    "encounter_type_cat3": "Service Nuance & Value",
    "referrals_counts_quarterly": "Demand Heartbeat & Capacity Warning",
    "encounters_counts_quarterly": "Efficiency & Workload Analysis",
    "encounter_type": "Operational Style Profile",
}

REFERRALS_RATIONALE_MAP: dict[str, str] = {
    "age": (
        "This distribution acts as a map of vulnerability rather than just a demographic breakdown. A dominance of the 55-84 age bracket "
        "often signals a system managing frailty, isolation, and chronic disease—patients who may be falling through the cracks of "
        "traditional primary care. Conversely, spikes in younger cohorts can indicate emerging behavioral health needs or substance use "
        "challenges. Understanding these shifts allows for more precise resource allocation, whether that means prioritizing fall "
        "prevention strategies or expanding mental health navigation support."
    ),
    "insurance": (
        "Payer mix serves as a critical financial reality check for the program. A heavy skew towards Medicaid and Uninsured populations "
        "often highlights gaps in the safety net, suggesting that community paramedics are serving as a de facto medical home for those "
        "without consistent primary care. Alternatively, higher commercial insurance rates may indicate the program is bridging access "
        "barriers for insured patients who cannot reach their providers. This insight helps determine whether to focus on enrollment "
        "assistance and charity care or on coordinating with private systems."
    ),
    "referral_agency": (
        "This histogram functions as a relationship health monitor for the program. Dominance by key partners like NOHN and OMC suggests "
        "strong clinical integration and trust from the medical community. However, a high volume of 'Self/Family' referrals or low "
        "participation from expected partners may reveal awareness gaps or friction in the referral pathway. Identifying underrepresented "
        "sources highlights opportunities to strengthen outreach and ensure all patient populations have a clear pathway to care."
    ),
    "zipcode": (
        "Geography often dictates access in healthcare. This map reveals where the system may be failing to reach patients, with clusters "
        "in specific areas potentially correlating with 'food deserts,' transportation barriers, or housing instability. Hotspots in "
        "remote locations suggest a need for optimized mobile unit routing, while downtown clusters may point to density-driven issues "
        "like homelessness. Reading the terrain of need allows for more targeted and effective community deployment."
    ),
    "referral_closed_reason": (
        "This chart serves as a 'friction report' for the intake process. While high 'Goals Met' rates are a success metric, significant "
        "numbers of 'Unable to Contact' or 'Declined' outcomes highlight process challenges rather than patient disinterest. These "
        "patterns may indicate that intake scripts need refinement or that follow-up timing requires adjustment. addressing these "
        "friction points ensures that patients who reach out for help don't get lost in the noise of the system."
    ),
    "sex": (
        "Gender balance is a subtle but important indicator of access equity. While a balanced distribution is typical, significant skews "
        "prompt further inquiry into potential blind spots. For instance, are men less likely to accept help, or are women being referred "
        "more frequently for social needs? Monitoring this balance ensures that referral criteria and provider perceptions remain inclusive "
        "and do not inadvertently filter out specific demographics."
    ),
    "encounter_type_cat1": (
        "This chart clarifies the core function of the program by categorizing the actual work performed. A dominance of 'Medical' "
        "encounters suggests the program is filling gaps in primary care, while a lead in 'Social' or 'Behavioral' categories indicates "
        "a strong need for social work capabilities in the field. This distinction is vital for aligning staffing models and job "
        "descriptions with the reality of patient needs."
    ),
    "encounter_type_cat2": (
        "Drilling down into secondary categories reveals the complexity of care. Broad labels like 'Medical' are broken down into specifics "
        "such as 'Wound Care' or 'Medication Management,' exposing the daily clinical skills required by the team. This level of detail "
        "moves beyond guesswork, allowing for targeted training and resource allocation based on what paramedics are actually encountering "
        "in the field."
    ),
    "encounter_type_cat3": (
        "This granular view captures the nuance of service delivery. Tracking specific interventions—like particular types of education "
        "or distinct social referrals—provides the detailed evidence needed to demonstrate value to funders and stakeholders. It transforms "
        "general statements of help into quantifiable data, supporting robust grant reporting and continuous quality improvement initiatives."
    ),
    "referrals_counts_quarterly": (
        "This chart tracks the heartbeat of demand, serving as an early warning system for capacity planning. By normalizing against a "
        "baseline (excluding atypical years), it distinguishes between steady operations and demand surges. Consistent 'above average' "
        "activity signals high utilization, while 'below average' periods may suggest a need for increased outreach. The moving average "
        "helps identify trends early, allowing for proactive adjustments before capacity issues arise."
    ),
    "encounters_counts_quarterly": (
        "Comparing Workload against Demand offers a clear view of program efficiency. While Referrals represent the 'ask,' Encounters "
        "show the 'answer.' A divergence where Referrals rise but Encounters remain flat may indicate a capacity ceiling, whereas "
        "parallel growth suggests successful scaling. Conversely, fewer Referrals with high Encounters points to higher acuity cases. "
        "Monitoring the gap between these metrics reveals the true efficiency and capacity of the service."
    ),
    "encounter_type": (
        "This multidimensional view combines engagement method, communication channel, and contact type to reveal the program's "
        "operational style. It highlights the balance between high-touch field work and remote coordination, showing whether the team "
        "operates more as 'boots on the ground' or a coordination hub. Understanding this balance helps ensure that resources are "
        "deployed effectively to meet patient needs while maintaining operational efficiency."
    ),
}

REFERRALS_STORY_CARDS = [
    {
        "eyebrow": "Story",
        "title": "Referral pipeline with normalized context",
        "lede": (
            "Share-of-total overlays make it clear which partners supply the majority of volume and how "
            "close you are to diversification targets."
        ),
        "bullets": [
            "Treemap hover states now expose both count and percent for each referring agency.",
            "Quarterly bars include percent share labels plus a three-quarter moving average to track momentum.",
            "Insurance mix percent labels point to coverage gaps that can delay onboarding.",
        ],
        "callout": "Use the percent view when negotiating outreach cadences with partners.",
    },
    {
        "eyebrow": "Throughput",
        "title": "Closed reasons at-a-glance",
        "lede": (
            "Horizontal bars combine counts and percent so high-friction buckets (unable to contact, declined) "
            "surface instantly."
        ),
        "bullets": [
            "Unknown segments stand out for data hygiene follow-up.",
            "Percentages make it easier to compare reasons even when total referrals fluctuate.",
            "Layer insights with encounter type shares to prioritize process fixes.",
        ],
        "callout": "Assign action owners for any category above 25%.",
    },
    {
        "eyebrow": "Next steps",
        "title": "Focus partner conversations",
        "lede": (
            "Work the narrative: start with normalized quarterly trend, pivot to partner contributions, and finish "
            "on closure-barriers."
        ),
        "bullets": [
            "Export referral shares into briefing decks without extra spreadsheet work.",
            "Use hover text percent values to align with grant reporting language (share of total referrals).",
            "Document open questions directly in the dashboard so the team can respond asynchronously.",
        ],
        "callout": "Re-run the dashboard after each data import to keep the story fresh.",
    },
]


REFERRALS_QUERY_FIELDS = [
    "age",
    "sex",
    "date_received",
    "referral_agency",
    "encounter_type_cat1",
    "encounter_type_cat2",
    "encounter_type_cat3",
    "referral_closed_reason",
    "zipcode",
    "insurance",
    "referral_1",
    "referral_2",
    "referral_3",
    "referral_4",
    "referral_5",
]

REFERRALS_DISPLAY_FIELDS = [
    "referrals_counts_quarterly",
    "encounters_counts_quarterly",
    "age",
    "sex",
    "referral_closed_reason",
    "referral_agency",
    "encounter_type",  # Special field - uses cat1/2/3 columns
]

REFERRALS_SINGLE_COLUMN_FIELDS = {"sex", "referral_closed_reason"}

REFERRALS_LABEL_OVERRIDES = {
    "age": "Referrals by Age",
    "sex": "Referrals by Sex",
    "insurance": "Referrals by Insurance",
    "zipcode": "Referrals by ZIP Code",
    "referral_agency": "Referrals by Agency",
    "encounter_type": "Encounter Type",
    "referrals_counts_quarterly": "Referrals by Quarter",
    "encounters_counts_quarterly": "Encounters by Quarter",
}


def _load_referrals_dataframe() -> pd.DataFrame:
    try:
        values = list(Referrals.objects.all().values(*REFERRALS_QUERY_FIELDS))
    except Exception:
        values = []
    df = _df_from_queryset(values, REFERRALS_QUERY_FIELDS)
    if "date_received" in df.columns:
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    return df


def _ordered_referral_fields(df: pd.DataFrame) -> list[str]:
    fields = [field for field in REFERRALS_DISPLAY_FIELDS if field in df.columns]
    # Add referrals_counts_quarterly (it's a virtual field, not in DataFrame)
    if "referrals_counts_quarterly" not in fields:
        try:
            insert_idx = REFERRALS_DISPLAY_FIELDS.index("referrals_counts_quarterly")
            actual_idx = sum(1 for f in REFERRALS_DISPLAY_FIELDS[:insert_idx] if f in fields)
            fields.insert(actual_idx, "referrals_counts_quarterly")
        except (ValueError, IndexError):
            fields.insert(0, "referrals_counts_quarterly")
    # Add encounters_counts_quarterly (it's a virtual field, not in DataFrame)
    if "encounters_counts_quarterly" not in fields:
        try:
            insert_idx = REFERRALS_DISPLAY_FIELDS.index("encounters_counts_quarterly")
            actual_idx = sum(1 for f in REFERRALS_DISPLAY_FIELDS[:insert_idx] if f in fields)
            fields.insert(actual_idx, "encounters_counts_quarterly")
        except (ValueError, IndexError):
            fields.append("encounters_counts_quarterly")
    # Add encounter_type virtual field if cat columns exist
    if "encounter_type" not in fields and all(
        col in df.columns
        for col in ["encounter_type_cat1", "encounter_type_cat2", "encounter_type_cat3"]
    ):
        # Insert encounter_type in its display order position
        try:
            insert_idx = REFERRALS_DISPLAY_FIELDS.index("encounter_type")
            # Adjust for fields that were filtered out
            actual_idx = sum(1 for f in REFERRALS_DISPLAY_FIELDS[:insert_idx] if f in fields)
            fields.insert(actual_idx, "encounter_type")
        except (ValueError, IndexError):
            fields.append("encounter_type")
    return fields


PATIENT_BASE_FIELDS = [
    "age",
    "marital_status",
    "veteran_status",
    "insurance",
    "sud",
    "pcp_agency",
    "zip_code",
    "race",
    "sex",
    # "behavioral_health",  # Hidden - insufficient data for meaningful visualization
]

PATIENT_SINGLE_COLUMN_FIELDS = {
    "insurance",
    "sud",
    # "behavioral_health",  # Hidden - insufficient data
    "marital_status",
    "veteran_status",
}

PATIENT_LABEL_OVERRIDES = {
    "pcp_agency": "Primary Care Agency",
    "zip_code": "ZIP Code",
    "patient_counts_quarterly": "Patients by Quarter",
    "race_age_boxplot": "Age Distribution by Race",
    "sud": "Substance Use Disorder",
    "age": "Age Distribution by Sex",
    "age_referral_sankey": "Age Groups → Key Services",
}

PATIENT_CHART_RATIONALE = {
    "age": (
        "Stacked bars show age distribution split by sex, revealing both demographic patterns and gender balance "
        "across age cohorts. This guides risk stratification, service planning, and helps identify gender-specific "
        "needs in each age group. It is used to plan targeted outreach and ensure equitable access across demographics."
    ),
    "insurance": (
        "Insurance indicates coverage pathways and barriers to access. It helps target enrollment assistance, "
        "reduce out-of-pocket costs, and select services that minimize financial burden."
    ),
    "pcp_agency": (
        "Knowing a patient’s primary care agency enables closed-loop referrals and shared care plans. It also "
        "reveals gaps in primary care linkage where CPMs can coordinate warm handoffs."
    ),
    "zip_code": (
        "ZIP codes help identify geographic clusters, travel barriers, and social determinant of health needs. They inform mobile "
        "clinic routing, outreach scheduling, and partnerships with local resources."
    ),
    "marital_status": (
        "Marital status approximates available social support. It can inform safety planning, caregiver outreach, "
        "and the intensity of follow-up needed to ensure adherence."
    ),
    "veteran_status": (
        "Veteran status opens access to VA services, care coordination, and benefits unique to veterans. "
        "Capturing it ensures no-duplicate coverage and better coordination."
    ),
    "sud": (
        "Substance use disorder screening is central to harm reduction and linkage to treatment. Tracking prevalence "
        "and screening completeness helps target naloxone distribution, MAT referrals, and proactive outreach. "
        "Understanding SUD patterns guides resource allocation and partnership development with treatment providers."
    ),
    "patient_counts_quarterly": (
        "This chart tracks quarterly patient enrollment trends with contextual intelligence. Bar colors dynamically "
        "distinguish above-average quarters (emerald green) from at or below-average quarters (cyan blue). The "
        "baseline average is calculated specifically from 2021, 2023, and 2024 data—our 'normal' operational years—"
        "excluding 2022's Behavioral Health program expansion spike and 2025's ongoing growth from adding 2 new "
        "Community Paramedics. This selective baseline provides a more meaningful reference point for evaluating "
        "current performance. Vertical year labels identify key contextual factors: COVID-19 impact (2021), "
        "Behavioral Health expansion (2022), Normalization period (2023-2024), and staffing increase (2025). "
    ),
    "race_age_boxplot": (
        "Age distribution by race helps detect potential disparities and informs culturally responsive, "
        "community-informed outreach strategies."
    ),
    "age_referral_sankey": (
        "The Sankey diagram visualizes patient flow from age cohorts to key service types including 911 calls, "
        "vaccinations, assessments, case management, overdose response, and medication services. "
        "Flow thickness indicates service volume, helping identify which age groups receive which services most. "
        "IMPORTANT: This chart represents a PORTION of all referral types (focused on high-impact services). "
        "KEY INSIGHT: Notice how each age group shows similar needs and service gaps—this reveals "
        "systemic care patterns rather than age-specific issues, suggesting opportunities for program-wide improvements."
    ),
}

PATIENTS_STORY_CARDS = [
    {
        "eyebrow": "Story",
        "title": "Quarterly demand normalized per 100 patients",
        "lede": (
            "Counts now surface both raw totals and their share of the patient panel, while a three-quarter "
            "rolling average highlights sustained growth beyond seasonal spikes."
        ),
        "bullets": [
            "Hover labels pair counts with percent of panel so mix shifts stand out immediately.",
            "Per 100 patient rates clarify throughput even as enrollment expands or contracts.",
            "Use the rolling average trace to defend staffing pivots and resource pulls.",
        ],
        "callout": "Cross-check with encounter cadence to make the caseload story concrete.",
    },
    {
        "eyebrow": "Equity",
        "title": "Demographic mix in context",
        "lede": (
            "Horizontal bars now annotate percent share, making it easier to scan for underrepresented "
            "communities across insurance, geography, and social determinants."
        ),
        "bullets": [
            "Review combined count and percent text to prioritize outreach invitations.",
            "Boxplots keep age distribution front of mind when coordinating partner services.",
            "SUD and behavioral health donuts quantify screening completion and prevalence side by side.",
        ],
        "callout": "Flag any share under 10% for deeper qualitative follow-up.",
    },
    {
        "eyebrow": "Next steps",
        "title": "Translate insight to action",
        "lede": (
            "Pair the normalized dashboards with patient-level exports to queue specific follow-ups and "
            "community clinics."
        ),
        "bullets": [
            "Draft targeted narratives for quarterly reports using the new percent annotations.",
            "Bundle charts into partner briefings to show progress toward equity and engagement goals.",
            "Document any data quality issues surfaced by large Unknown buckets and assign clean-up owners.",
        ],
        "callout": "Schedule a monthly review to keep the normalization assumptions current.",
    },
]


def _load_patients_dataframe() -> pd.DataFrame:
    try:
        data = list(Patients.objects.all().values(*PATIENT_BASE_FIELDS))
    except Exception:
        data = []
    return pd.DataFrame.from_records(data, columns=PATIENT_BASE_FIELDS)


def _ordered_patient_fields(df: pd.DataFrame) -> list[str]:
    # Start with quarterly chart at the top
    ordered = ["patient_counts_quarterly"]

    # Then age-related charts grouped together
    age_charts = ["age", "race_age_boxplot", "age_referral_sankey"]
    ordered.extend(age_charts)

    # Then donut charts in specified order: marital, veteran, insurance, sud
    donut_fields = ["marital_status", "veteran_status", "insurance", "sud"]
    for field in donut_fields:
        if field in df.columns:
            ordered.append(field)

    # Then Primary Care Agency and ZIP Code
    for field in ["pcp_agency", "zip_code"]:
        if field in df.columns:
            ordered.append(field)

    return ordered


def _build_patient_chart_meta() -> dict[str, object]:
    meta: dict[str, object] = {}
    return meta


def _load_patients_story_frames() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    patients_records = list(
        Patients.objects.all().values(
            "id", "age", "zip_code", "sud", "behavioral_health", "created_date"
        )
    )
    if not patients_records:
        return None

    df_patients = pd.DataFrame.from_records(patients_records)
    df_patients["age"] = pd.to_numeric(df_patients["age"], errors="coerce")
    df_patients["id"] = pd.to_numeric(df_patients["id"], errors="coerce")
    df_patients["created_date"] = pd.to_datetime(df_patients["created_date"], errors="coerce")
    df_patients = df_patients.dropna(subset=["id"])
    df_patients["id"] = df_patients["id"].astype(int)

    encounters_records = list(
        Encounters.objects.exclude(patient_ID__isnull=True).values("patient_ID", "encounter_date")
    )
    df_enc = pd.DataFrame.from_records(encounters_records)
    if df_enc.empty:
        df_enc = pd.DataFrame(columns=["patient_ID", "encounter_date"])
    else:
        df_enc["patient_ID"] = pd.to_numeric(df_enc["patient_ID"], errors="coerce")
        df_enc["encounter_date"] = pd.to_datetime(df_enc["encounter_date"], errors="coerce")
        df_enc = df_enc.dropna(subset=["patient_ID", "encounter_date"])
        if not df_enc.empty:
            df_enc["patient_ID"] = df_enc["patient_ID"].astype(int)

    return df_patients, df_enc


def _patients_story_demand_section(
    df_patients: pd.DataFrame, total_patients: int
) -> dict[str, object]:
    seniors_mask = df_patients["age"] >= 65
    share_65 = _pct(int(seniors_mask.sum()), total_patients)
    share_75 = _pct(int((df_patients["age"] >= 75).sum()), total_patients)

    zip_series = (
        df_patients.get("zip_code", pd.Series(dtype="object")).fillna("").astype(str).str.strip()
    )
    zip_series = zip_series[zip_series != ""]
    top_zip_counts = zip_series.value_counts()
    top_zip_share = (
        _pct(int(top_zip_counts.head(3).sum()), total_patients) if not top_zip_counts.empty else 0.0
    )
    top_zip_names = ", ".join(top_zip_counts.head(3).index.tolist())

    demand_lede = f"We're seeing {_format_percent(share_65)} of enrolled patients are 65+"
    if top_zip_names:
        demand_lede += f", and top ZIPs ({top_zip_names}) account for {_format_percent(top_zip_share)} of the caseload."
    else:
        demand_lede += "."
    demand_lede += (
        " Step 1 frames where demand is concentrated so the team can open with the right context."
    )

    demand_actions: list[str] = []
    if top_zip_names:
        demand_actions.append(
            f"Coordinate mobile teams in {top_zip_names} to match the {_format_percent(top_zip_share)} of enrollment clustered there."
        )
    if share_75:
        demand_actions.append("Partner with aging services to support the growing 75+ caseload.")
    demand_actions.append("Keep the quarterly volume chart handy for grant and staffing updates.")

    metrics = [
        _metric("Active patients", _format_int(total_patients), "Current registry size"),
        _metric(
            "65+ share",
            _format_percent(share_65),
            f"{_format_percent(share_75)} of patients are 75+" if share_75 else None,
        ),
        _metric(
            "Top ZIP concentration",
            _format_percent(top_zip_share),
            f"{top_zip_names}" if top_zip_names else None,
        ),
    ]

    return {
        "sequence": 1,
        "title": "Community Need",
        "lede": demand_lede,
        "metrics": metrics,
        "actions": demand_actions,
        "related_charts": [
            "patient_counts_quarterly",
            "age",
            "zip_code",
            "sud",
            "behavioral_health",
        ],
    }


def _patients_story_response_section(
    df_patients: pd.DataFrame, df_enc: pd.DataFrame, total_patients: int
) -> dict[str, object]:
    patients_with_encounter = int(df_enc["patient_ID"].nunique()) if not df_enc.empty else 0
    engagement_rate = _pct(patients_with_encounter, total_patients)

    recent_patients = 0
    recent_pct = 0.0
    avg_encounters_per_engaged: float | None = None
    median_days_to_first: float | None = None

    if not df_enc.empty:
        now = pd.Timestamp.utcnow().tz_localize(None)
        recent_cutoff = now - pd.Timedelta(days=90)
        recent_patients = int(
            df_enc[df_enc["encounter_date"] >= recent_cutoff]["patient_ID"].nunique()
        )
        recent_pct = _pct(recent_patients, total_patients)

        encounters_per_patient = df_enc.groupby("patient_ID").size()
        if not encounters_per_patient.empty:
            avg_encounters_per_engaged = float(encounters_per_patient.mean())

        first_encounters = df_enc.groupby("patient_ID")["encounter_date"].min()
        patient_created = df_patients[["id", "created_date"]].dropna(subset=["id", "created_date"])
        if not patient_created.empty:
            first_df = patient_created.merge(
                first_encounters.to_frame("first_encounter"),
                left_on="id",
                right_index=True,
                how="inner",
            )
            if not first_df.empty:
                first_df["days_to_first"] = (
                    first_df["first_encounter"] - first_df["created_date"]
                ).dt.days
                valid_first = first_df[
                    first_df["days_to_first"].notna() & (first_df["days_to_first"] >= 0)
                ]
                if not valid_first.empty:
                    median_days_to_first = float(valid_first["days_to_first"].median())

    response_lede = f"Step 2 highlights how we're responding: {_format_percent(engagement_rate)} of patients have completed at least one encounter, reaching {_format_int(recent_patients)} people in the past 90 days."
    if median_days_to_first is not None:
        response_lede += f" Median time from enrollment to first visit is {_format_days(median_days_to_first)}, showing how fast we activate services."

    metrics = [
        _metric(
            "Engagement rate",
            _format_percent(engagement_rate),
            f"{_format_int(patients_with_encounter)} of {_format_int(total_patients)} patients have at least one encounter.",
        ),
        _metric(
            "Reached in last 90 days",
            _format_int(recent_patients),
            f"{_format_percent(recent_pct)} of the panel received a touchpoint this quarter.",
        ),
        _metric(
            "Median days to first visit",
            _format_days(median_days_to_first),
            "Enrollment to first encounter."
            if median_days_to_first is not None
            else "Enrollment timing data unavailable.",
        ),
    ]
    if avg_encounters_per_engaged is not None:
        metrics.append(
            _metric(
                "Avg encounters per engaged patient",
                _format_float(avg_encounters_per_engaged, digits=1),
                "Calculated across patients with at least one visit.",
            )
        )

    actions = [
        "Track enrollment-to-first-visit lag each week and clear blockers when it rises.",
        "Pair encounter cadence with the age distribution to plan multidisciplinary rounds.",
    ]
    if recent_patients < patients_with_encounter:
        actions.append("Schedule wellness checks for patients without contact in the last 90 days.")

    return {
        "sequence": 2,
        "title": "Response",
        "lede": response_lede,
        "metrics": metrics,
        "actions": actions,
        "related_charts": ["patient_counts_quarterly", "sud", "behavioral_health"],
    }


def _patients_story_impact_section(
    df_patients: pd.DataFrame,
    df_enc: pd.DataFrame,
    total_patients: int,
) -> dict[str, object]:
    patients_with_encounter = int(df_enc["patient_ID"].nunique()) if not df_enc.empty else 0
    repeat_rate = 0.0
    sud_engaged = 0
    sud_count = 0
    sud_engagement_pct = 0.0
    senior_avg: float | None = None

    if not df_enc.empty and patients_with_encounter:
        df_enc_sorted = df_enc.sort_values(["patient_ID", "encounter_date"])
        df_enc_sorted["prev_encounter"] = df_enc_sorted.groupby("patient_ID")[
            "encounter_date"
        ].shift(1)
        df_enc_sorted["days_since_prior"] = (
            df_enc_sorted["encounter_date"] - df_enc_sorted["prev_encounter"]
        ).dt.days
        repeat_within_30 = df_enc_sorted[
            df_enc_sorted["days_since_prior"].notna()
            & (df_enc_sorted["days_since_prior"] >= 1)
            & (df_enc_sorted["days_since_prior"] <= 30)
        ]
        repeat_rate = _pct(int(repeat_within_30["patient_ID"].nunique()), patients_with_encounter)

        sud_mask = df_patients.get("sud")
        if sud_mask is not None:
            sud_mask = sud_mask.fillna(False).astype(bool)
            sud_ids = df_patients.loc[sud_mask, "id"]
            sud_count = int(sud_ids.nunique())
            if sud_count:
                sud_engaged = int(
                    df_enc[df_enc["patient_ID"].isin(sud_ids.tolist())]["patient_ID"].nunique()
                )
                sud_engagement_pct = _pct(sud_engaged, sud_count)

        senior_ids = df_patients.loc[df_patients["age"] >= 65, "id"]
        if not senior_ids.empty:
            senior_encounters = df_enc[df_enc["patient_ID"].isin(senior_ids.tolist())]
            if not senior_encounters.empty:
                senior_avg = float(senior_encounters.groupby("patient_ID").size().mean())

    impact_lede = f"Step 3 shows the impact we're driving: {_format_percent(repeat_rate)} of engaged patients need a follow-up within 30 days."
    if sud_count:
        impact_lede += f" {_format_percent(sud_engagement_pct)} of the SUD panel ({_format_int(sud_engaged)} people) is in active care, keeping high-risk neighbors connected."
    if senior_avg is not None:
        impact_lede += f" Seniors average {_format_float(senior_avg, digits=1)} encounters each, underscoring the preventive value of regular visits."

    metrics = [
        _metric(
            "Repeat visit within 30 days",
            _format_percent(repeat_rate),
            "Share of engaged patients requiring rapid follow-up.",
        ),
        _metric(
            "SUD cohort engaged",
            _format_percent(sud_engagement_pct),
            f"{_format_int(sud_engaged)} of {_format_int(sud_count)} flagged patients are in care."
            if sud_count
            else "SUD flags not recorded.",
        ),
        _metric(
            "Avg encounters per 65+ patient",
            _format_float(senior_avg, digits=1),
            "Average visits among older adults who engaged."
            if senior_avg is not None
            else "No senior encounter data.",
        ),
    ]

    actions = [
        "Use the repeat-within-30-days list to prioritize proactive follow-up routes.",
        "Check SUD-flagged patients without recent encounters and coordinate MAT or harm reduction outreach.",
    ]
    if senior_avg and senior_avg > 2:
        actions.append(
            "Align staffing for home visits to keep pace with high senior visit intensity."
        )

    return {
        "sequence": 3,
        "title": "Impact",
        "lede": impact_lede,
        "metrics": metrics,
        "actions": actions,
        "related_charts": ["patient_counts_quarterly", "age"],
    }


def _build_patients_story_sections() -> list[dict[str, object]]:
    frames = _load_patients_story_frames()
    if frames is None:
        return []

    df_patients, df_enc = frames
    total_patients = int(df_patients.shape[0])
    if total_patients == 0:
        return []

    demand_section = _patients_story_demand_section(df_patients, total_patients)
    response_section = _patients_story_response_section(df_patients, df_enc, total_patients)
    impact_section = _patients_story_impact_section(df_patients, df_enc, total_patients)

    return [demand_section, response_section, impact_section]


def _build_yearly_accordions() -> list[dict[str, object]]:
    """Build accordion data for each year with Community Need, Response, and Impact."""
    # Get current year data (2025 - using real data)
    frames = _load_patients_story_frames()
    current_sections = []
    if frames is not None:
        df_patients, df_enc = frames
        total_patients = int(df_patients.shape[0])
        if total_patients > 0:
            current_sections = [
                _patients_story_demand_section(df_patients, total_patients),
                _patients_story_response_section(df_patients, df_enc, total_patients),
                _patients_story_impact_section(df_patients, df_enc, total_patients),
            ]

    # Define yearly data (2025 uses real/current data, others are placeholders)
    # Return in reverse chronological order (2025 -> 2019)
    years_data = [
        {
            "year": 2025,
            "sections": current_sections if current_sections else _get_placeholder_sections(2025),
            "is_current": True,
        },
        {"year": 2024, "sections": _get_placeholder_sections(2024), "is_current": False},
        {"year": 2023, "sections": _get_placeholder_sections(2023), "is_current": False},
        {"year": 2022, "sections": _get_placeholder_sections(2022), "is_current": False},
        {"year": 2021, "sections": _get_placeholder_sections(2021), "is_current": False},
        {"year": 2020, "sections": _get_placeholder_sections(2020), "is_current": False},
        {"year": 2019, "sections": _get_placeholder_sections(2019), "is_current": False},
    ]

    return years_data


def _get_placeholder_sections(year: int) -> list[dict[str, object]]:
    """Generate placeholder sections for a given year."""
    return [
        {
            "sequence": 1,
            "title": "Community Need",
            "lede": f"In {year}, community health needs continued to evolve with changing demographics and service requirements. Analysis of patient enrollment and geographic distribution informed resource allocation.",
            "metrics": [
                _metric("Active patients", f"{450 + (year - 2019) * 50}", "Yearly registry size"),
                _metric(
                    "65+ share", f"{42 + (year - 2019) * 2}%", f"{28 + (year - 2019)}% are 75+"
                ),
                _metric("Top ZIP concentration", f"{65 - (year - 2019)}%", "98362, 98363, 98382"),
            ],
            "actions": [
                "Expand mobile services to high-concentration ZIP codes",
                "Strengthen partnerships with aging and social services",
                f"Monitor quarterly trends for {year + 1} planning cycles",
            ],
            "related_charts": ["patient_counts_quarterly", "age", "zip_code"],
        },
        {
            "sequence": 2,
            "title": "Response",
            "lede": f"During {year}, the program delivered {2800 + (year - 2019) * 400} encounters across home visits, clinic support, and care coordination. Field teams prioritized high-acuity patients while maintaining strong primary care partnerships.",
            "metrics": [
                _metric(
                    "Total encounters",
                    f"{2800 + (year - 2019) * 400}",
                    f"Avg {5.2 + (year - 2019) * 0.3}/patient",
                ),
                _metric("Home visit %", f"{38 + (year - 2019) * 2}%", "Most common encounter type"),
                _metric("Top agency partnership", "NOHN", f"{35 - (year - 2019)}% of referrals"),
            ],
            "actions": [
                f"Scale home visit capacity for {year + 1}",
                "Deepen NOHN and OMC care coordination",
                "Train staff on new encounter documentation protocols",
            ],
            "related_charts": ["encounter_type", "pcp_agency"],
        },
        {
            "sequence": 3,
            "title": "Impact",
            "lede": f"In {year}, {180 + (year - 2019) * 25} patients engaged in 4+ encounters, demonstrating sustained program value. Coordinated care and follow-up visits helped manage chronic conditions and reduce acute episodes.",
            "metrics": [
                _metric("Highly engaged", f"{180 + (year - 2019) * 25}", "4+ encounters/year"),
                _metric("Avg engagement", f"{5.2 + (year - 2019) * 0.3}", "Encounters per patient"),
                _metric("Care continuity", f"{72 + (year - 2019)}%", "Follow-up completion rate"),
            ],
            "actions": [
                f"Celebrate high-engagement patients in {year + 1} reviews",
                "Analyze barriers for low-engagement cohorts",
                "Share impact stories with funders and stakeholders",
            ],
            "related_charts": ["top_engaged_patients"],
        },
    ]


def patients(request):
    theme = get_theme_from_request(request)
    df_all = _load_patients_dataframe()
    ordered_fields = _ordered_patient_fields(df_all)

    chart_cards = [
        {
            "field": field,
            "label": PATIENT_LABEL_OVERRIDES.get(field, field.replace("_", " ").title()),
            "col_span": 1 if field in PATIENT_SINGLE_COLUMN_FIELDS else 2,
        }
        for field in ordered_fields
    ]

    context = {
        "chart_cards": chart_cards,
        "theme": theme,
        "story_sections": _build_patients_story_sections(),
        "yearly_accordions": _build_yearly_accordions(),
    }
    updated_on = date(2025, 11, 30)
    context.update(
        {
            "page_header_updated_at": updated_on,
            "page_header_updated_at_iso": updated_on.isoformat(),
            "page_header_read_time": "7 min read",
        }
    )
    return render(request, "dashboard/patients.html", context)


@require_GET
def patients_chart_fragment(request, field: str):
    theme = get_theme_from_request(request)
    df_all = _load_patients_dataframe()
    valid_fields = set(_ordered_patient_fields(df_all))
    if field not in valid_fields:
        raise Http404

    charts = build_patients_field_charts(theme=theme, fields=[field])
    chart_html = charts.get(field, "")
    has_chart = bool(chart_html)
    age_insights = _compute_age_insights(df_all)
    chart_insights = _build_patients_chart_insights(df_all, age_insights)
    chart_meta = _build_patient_chart_meta()
    quick_stats = _build_patient_quick_stats()

    item = {
        "field": field,
        "label": PATIENT_LABEL_OVERRIDES.get(field, field.replace("_", " ").title()),
        "chart": chart_html,
        "has_chart": has_chart,
        "category_label": "Patients",
        "insights": chart_insights.get(field),
        "rationale": PATIENT_CHART_RATIONALE.get(field),
        "meta": chart_meta.get(field),
        "quick_stats": quick_stats.get(field),
    }

    return render(request, "dashboard/partials/patient_chart_fragment.html", {"item": item})


@require_GET
def referrals_chart_fragment(request, field: str):
    theme = get_theme_from_request(request)
    df_all = _load_referrals_dataframe()
    valid_fields = set(_ordered_referral_fields(df_all))
    if field not in valid_fields:
        raise Http404

    charts = build_referrals_field_charts(theme=theme, fields=[field])
    chart_html = charts.get(field, "")
    has_chart = bool(chart_html)

    chart_insights = _build_referrals_chart_insights(df_all)
    quick_stats = _build_referral_quick_stats()
    encounters_stats = _build_encounters_quarterly_quick_stats()
    quick_stats.update(encounters_stats)

    item = {
        "field": field,
        "label": REFERRALS_LABEL_OVERRIDES.get(field, field.replace("_", " ").title()),
        "chart": chart_html,
        "has_chart": has_chart,
        "category_label": "Referrals",
        "insights": chart_insights.get(field),
        "rationale": REFERRALS_RATIONALE_MAP.get(field),
        "rationale_heading": REFERRALS_RATIONALE_HEADINGS.get(field),
        "meta": None,
        "quick_stats": quick_stats.get(field),
    }

    return render(request, "dashboard/partials/patient_chart_fragment.html", {"item": item})


@require_GET
def odreferrals_chart_fragment(request, field: str):
    theme = get_theme_from_request(request)
    df_all = _load_odreferrals_dataframe()
    valid_fields = set(_ordered_odreferral_fields(df_all))
    if field not in valid_fields:
        raise Http404

    charts = build_odreferrals_field_charts(theme=theme, fields=[field])
    chart_html = charts.get(field, "")
    has_chart = bool(chart_html)

    chart_insights = _build_odreferrals_chart_insights(df_all)
    quick_stats = _build_odreferrals_quick_stats()

    item = {
        "field": field,
        "label": OD_REFERRALS_LABEL_MAP.get(field, field.replace("_", " ").title()),
        "chart": chart_html,
        "has_chart": has_chart,
        "category_label": "OD Referrals",
        "insights": chart_insights.get(field),
        "rationale": OD_REFERRALS_RATIONALE_MAP.get(field),
        "meta": None,
        "quick_stats": quick_stats.get(field),
    }

    return render(request, "dashboard/partials/patient_chart_fragment.html", {"item": item})


# Referrals


def referrals(request):
    theme = get_theme_from_request(request)
    df_all = _load_referrals_dataframe()
    ordered_fields = _ordered_referral_fields(df_all)

    chart_cards = [
        {
            "field": field,
            "label": REFERRALS_LABEL_OVERRIDES.get(field, field.replace("_", " ").title()),
            "col_span": 1 if field in REFERRALS_SINGLE_COLUMN_FIELDS else 2,
        }
        for field in ordered_fields
    ]

    context = {
        "chart_cards": chart_cards,
        "theme": theme,
        "story_cards": REFERRALS_STORY_CARDS,
    }
    updated_on = date(2025, 11, 30)
    context.update(
        {
            "page_header_updated_at": updated_on,
            "page_header_updated_at_iso": updated_on.isoformat(),
            "page_header_read_time": "8 min read",
        }
    )
    return render(request, "dashboard/referrals.html", context)


def odreferrals_shift_coverage(request):
    theme = get_theme_from_request(request)

    fig_density_map = _chart_html(build_chart_od_density_heatmap(theme=theme))
    fig_day_of_week_totals = _chart_html(build_chart_day_of_week_totals(theme=theme))
    fig_hourly_breakdown = _chart_html(build_chart_od_hourly_breakdown(theme=theme))
    fig_shift_scenarios = _chart_html(build_chart_shift_scenarios(theme=theme))
    fig_cost_benefit_analysis = _chart_html(build_chart_cost_benefit_analysis(theme=theme))

    scenarios = calculate_coverage_scenarios()
    try:
        scenarios_data = json.dumps(scenarios)
    except TypeError:
        sanitized = {
            name: {
                key: value for key, value in data.items() if isinstance(value, int | float | str)
            }
            for name, data in scenarios.items()
        }
        scenarios_data = json.dumps(sanitized)

    coverage_stats = _compute_shift_coverage_stats()

    context = {
        "fig_density_map": fig_density_map,
        "fig_day_of_week_totals": fig_day_of_week_totals,
        "fig_hourly_breakdown": fig_hourly_breakdown,
        "fig_shift_scenarios": fig_shift_scenarios,
        "fig_cost_benefit_analysis": fig_cost_benefit_analysis,
        "scenarios_data": scenarios_data,
        "current_coverage": coverage_stats["current"],
        "missed_opportunities": coverage_stats["missed"],
        "proposed_coverage": coverage_stats["proposed"],
        "total_cases": coverage_stats["total"],
    }
    updated_on = date(2025, 11, 30)
    context.update(
        {
            "page_header_updated_at": updated_on,
            "page_header_updated_at_iso": updated_on.isoformat(),
            "page_header_read_time": "9 min read",
        }
    )
    return render(request, "dashboard/odreferrals_shift_coverage.html", context)


def _compute_hotspot_stats() -> dict[str, object]:
    qs = ODReferrals.objects.exclude(lat__isnull=True).exclude(long__isnull=True)
    df = _df_from_queryset(
        qs.values("lat", "long", "disposition", "od_date"),
        expected_columns=["lat", "long", "disposition", "od_date"],
    )
    if df.empty:
        return {
            "total": 0,
            "fatal_count": 0,
            "fatal_pct": 0.0,
            "unique_locations": 0,
            "top_location_count": 0,
            "top_location_share": 0.0,
            "weekend_pct": 0.0,
        }

    fatal_conditions = {"CPR attempted", "DOA"}
    total = int(len(df))
    fatal_count = int(df["disposition"].isin(fatal_conditions).sum())
    fatal_pct = round((fatal_count / total) * 100, 1) if total else 0.0

    unique_locations = int(df[["lat", "long"]].dropna().drop_duplicates().shape[0])

    grouped = df.groupby(["lat", "long"]).size().sort_values(ascending=False)
    top_location_count = int(grouped.iloc[0]) if not grouped.empty else 0
    top_location_share = round((top_location_count / total) * 100, 1) if total else 0.0

    weekend_count = 0
    for raw_dt in df["od_date"].dropna():
        converted = pd.to_datetime(raw_dt, errors="coerce")
        if pd.isna(converted):
            continue
        if isinstance(converted, pd.Timestamp):
            if converted.tzinfo is not None:
                converted = converted.tz_convert(timezone.get_current_timezone())
            converted = converted.to_pydatetime()
        if timezone.is_naive(converted):
            converted = timezone.make_aware(converted, timezone.get_current_timezone())
        local_dt = timezone.localtime(converted)
        weekend_count += int(local_dt.weekday() >= 5)

    weekend_pct = round((weekend_count / total) * 100, 1) if total else 0.0

    return {
        "total": total,
        "fatal_count": fatal_count,
        "fatal_pct": fatal_pct,
        "unique_locations": unique_locations,
        "top_location_count": top_location_count,
        "top_location_share": top_location_share,
        "weekend_pct": weekend_pct,
    }


def odreferrals_hotspots(request):
    theme = get_theme_from_request(request)
    zoom_mode = "full" if request.user.is_authenticated else "restricted"
    fig_od_map = _chart_html(build_chart_od_map(theme=theme, zoom_mode=zoom_mode))
    stats = _compute_hotspot_stats()
    document_metrics = _load_hotspot_context()

    insights: list[dict[str, str]] = []
    if stats["total"]:
        insights.append(
            {
                "title": "Concentrated clusters",
                "body": (
                    f"{stats['unique_locations']} unique hotspot addresses recorded; the most active"
                    f" location represents {stats['top_location_share']}% of EMS overdose responses."
                ),
            }
        )
        insights.append(
            {
                "title": "Severity mix",
                "body": (
                    f"Fatal overdoses make up {stats['fatal_pct']}% of mapped incidents"
                    f" ({stats['fatal_count']} of {stats['total']} cases), reinforcing targeted naloxone support."
                ),
            }
        )
        insights.append(
            {
                "title": "Weekend activity",
                "body": (
                    f"Weekend calls account for {stats['weekend_pct']}% of overdoses, aligning with scheduling"
                    " strategies that extend PORT coverage beyond standard weekday hours."
                ),
            }
        )

    context = {
        "fig_od_map": fig_od_map,
        "hotspot_stats": stats,
        "hotspot_insights": insights,
        "document_metrics": document_metrics,
    }
    updated_on = date(2025, 11, 30)
    context.update(
        {
            "page_header_updated_at": updated_on,
            "page_header_updated_at_iso": updated_on.isoformat(),
            "page_header_read_time": "6 min read",
        }
    )
    return render(request, "dashboard/odreferrals_hotspots.html", context)


# Referrals insights helpers
def _load_referral_datasets() -> dict[str, pd.DataFrame]:
    referral_fields = [
        "ID",
        "patient_ID",
        "date_received",
        "referral_agency",
        "encounter_type_cat1",
        "encounter_type_cat2",
        "encounter_type_cat3",
        "referral_1",
        "referral_2",
        "referral_3",
        "referral_4",
        "referral_5",
    ]
    encounter_fields = [
        "ID",
        "referral_ID",
        "patient_ID",
        "encounter_date",
        "pcp_agency",
        "encounter_type_cat1",
        "encounter_type_cat2",
        "encounter_type_cat3",
    ]
    odreferral_fields = ["ID", "patient_id", "od_date", "referral_agency"]

    referrals_values = list(Referrals.objects.all().values(*referral_fields))
    encounters_values = list(Encounters.objects.all().values(*encounter_fields))
    odreferral_values = list(ODReferrals.objects.all().values(*odreferral_fields))

    referrals_df = _df_from_queryset(referrals_values, referral_fields)
    encounters_df = _df_from_queryset(encounters_values, encounter_fields)
    od_df = _df_from_queryset(odreferral_values, odreferral_fields)

    return {
        "referrals": referrals_df,
        "encounters": encounters_df,
        "odreferrals": od_df,
    }


def _attach_touchpoint_metrics(
    referrals_df: pd.DataFrame,
    encounters_df: pd.DataFrame,
    od_df: pd.DataFrame,
) -> pd.DataFrame:
    if referrals_df.empty:
        empty_df = referrals_df.copy()
        for col in ["encounters_count", "odreferrals_count", "touchpoints_total"]:
            empty_df[col] = 0
        empty_df["has_encounter"] = False
        empty_df["has_od"] = False
        empty_df["days_to_first_encounter"] = pd.NA
        return empty_df

    enriched = referrals_df.copy()

    encounters_counts = (
        encounters_df.dropna(subset=["patient_ID"]).groupby("patient_ID").size().astype(int)
        if not encounters_df.empty and "patient_ID" in encounters_df.columns
        else pd.Series(dtype="int64")
    )
    od_counts = (
        od_df.dropna(subset=["patient_id"]).groupby("patient_id").size().astype(int)
        if not od_df.empty and "patient_id" in od_df.columns
        else pd.Series(dtype="int64")
    )

    enriched["encounters_count"] = (
        enriched["patient_ID"].map(encounters_counts).fillna(0).astype(int)
    )
    enriched["odreferrals_count"] = enriched["patient_ID"].map(od_counts).fillna(0).astype(int)
    enriched["has_encounter"] = enriched["encounters_count"] > 0
    enriched["has_od"] = enriched["odreferrals_count"] > 0
    enriched["touchpoints_total"] = 1 + enriched["encounters_count"] + enriched["odreferrals_count"]

    if "date_received" in enriched.columns:
        enriched["date_received"] = pd.to_datetime(enriched["date_received"], errors="coerce")

    if not encounters_df.empty and {"referral_ID", "encounter_date"}.issubset(
        encounters_df.columns
    ):
        encounters_dates = encounters_df.dropna(subset=["referral_ID", "encounter_date"]).copy()
        if not encounters_dates.empty:
            encounters_dates["encounter_date"] = pd.to_datetime(
                encounters_dates["encounter_date"], errors="coerce"
            )
            first_encounter = encounters_dates.groupby("referral_ID")["encounter_date"].min()
            enriched["first_encounter_at"] = enriched["ID"].map(first_encounter)
            if "date_received" in enriched.columns:
                enriched["days_to_first_encounter"] = (
                    enriched["first_encounter_at"] - enriched["date_received"]
                ).dt.days
    if "days_to_first_encounter" not in enriched.columns:
        enriched["days_to_first_encounter"] = pd.NA

    return enriched


def _build_referrals_summary_cards(
    total_referrals: int,
    unique_patients: int,
    referrals_with_encounters: int,
    engagement_rate: float,
    median_days: float | None,
    referrals_with_od: int,
    port_rate: float,
    avg_touchpoints: float,
) -> list[dict[str, object]]:
    encounter_desc = f"{engagement_rate:.1f}% convert to at least one CPM encounter."
    if median_days is not None and not pd.isna(median_days):
        encounter_desc += f" Median time to first encounter: {int(median_days)} days."

    return [
        {
            "label": "Referrals logged",
            "value": f"{total_referrals:,}",
            "description": "Total referrals received for CPM review.",
        },
        {
            "label": "Patients referred",
            "value": f"{unique_patients:,}",
            "description": "Unique individuals represented in the referral queue.",
        },
        {
            "label": "Referrals with encounters",
            "value": f"{referrals_with_encounters:,}",
            "description": encounter_desc,
        },
        {
            "label": "Referrals with PORT follow-up",
            "value": f"{referrals_with_od:,}",
            "description": f"{port_rate:.1f}% routed through overdose response workflows.",
        },
        {
            "label": "Avg touchpoints per referral",
            "value": f"{avg_touchpoints:.1f}",
            "description": "Includes CPM encounters and overdose follow-ups.",
        },
    ]


def _build_top_agency_rows(referrals_df: pd.DataFrame) -> list[dict[str, object]]:
    if referrals_df.empty or "referral_agency" not in referrals_df.columns:
        return []

    agency_df = referrals_df.copy()
    agency_df["referral_agency"] = agency_df["referral_agency"].apply(_normalize_text)
    grouped = (
        agency_df.groupby("referral_agency", dropna=False)
        .agg(
            total_referrals=("ID", "count"),
            engaged=("has_encounter", "sum"),
            port=("has_od", "sum"),
            unique_patients=("patient_ID", _safe_nunique),
            avg_touchpoints=("touchpoints_total", "mean"),
        )
        .reset_index()
    )
    if grouped.empty:
        return []

    rows: list[dict[str, object]] = []
    for _, row in (
        grouped.sort_values(["total_referrals", "engaged"], ascending=[False, False])
        .head(10)
        .iterrows()
    ):
        total = int(row["total_referrals"])
        engaged = int(row["engaged"])
        port = int(row["port"])
        rows.append(
            {
                "agency": _normalize_text(row["referral_agency"]),
                "total_referrals": total,
                "encounters": engaged,
                "port_followups": port,
                "unique_patients": int(row["unique_patients"]),
                "engagement_rate": _pct(engaged, total),
                "port_rate": _pct(port, total),
                "avg_touchpoints": float(row["avg_touchpoints"])
                if not pd.isna(row["avg_touchpoints"])
                else 0.0,
            }
        )
    return rows


def _aggregate_referrals_by_column(
    referrals_df: pd.DataFrame, column: str
) -> list[EncounterTableRow]:
    if referrals_df.empty or column not in referrals_df.columns:
        return []

    required_cols = ["ID", column, "has_encounter", "has_od", "touchpoints_total"]
    if not set(required_cols).issubset(referrals_df.columns):
        return []

    total_referrals = int(referrals_df.shape[0])
    df = referrals_df[required_cols].copy()
    df[column] = df[column].apply(_normalize_text)

    grouped = (
        df.groupby(column, dropna=False)
        .agg(
            referrals=("ID", "count"),
            engaged=("has_encounter", "sum"),
            port=("has_od", "sum"),
            avg_touchpoints=("touchpoints_total", "mean"),
        )
        .reset_index()
    )

    rows: list[EncounterTableRow] = []
    for _, row in grouped.sort_values("referrals", ascending=False).iterrows():
        referrals_count = int(row["referrals"])
        engaged_count = int(row["engaged"])
        port_count = int(row["port"])
        avg_touchpoints = (
            round(float(row["avg_touchpoints"]), 1) if not pd.isna(row["avg_touchpoints"]) else None
        )
        rows.append(
            {
                "label": _normalize_text(row[column]),
                "referral_count": referrals_count,
                "share_pct": _pct(referrals_count, total_referrals),
                "engaged_count": engaged_count,
                "engagement_pct": _pct(engaged_count, referrals_count),
                "port_count": port_count,
                "port_pct": _pct(port_count, referrals_count),
                "avg_touchpoints": avg_touchpoints,
            }
        )
    return rows


def _aggregate_referral_type_rows(referrals_df: pd.DataFrame) -> list[EncounterTableRow]:
    referral_columns = [
        col
        for col in ["referral_1", "referral_2", "referral_3", "referral_4", "referral_5"]
        if col in referrals_df.columns
    ]
    if referrals_df.empty or not referral_columns:
        return []

    required_cols = ["ID", "patient_ID", "has_encounter", "has_od", "touchpoints_total"]
    if not set(required_cols).issubset(referrals_df.columns):
        return []

    frames: list[pd.DataFrame] = []
    for col in referral_columns:
        subset = referrals_df[required_cols + [col]].copy()
        frames.append(subset.rename(columns={col: "referral_type"}))

    long_df = pd.concat(frames, ignore_index=True)
    long_df["referral_type"] = long_df["referral_type"].apply(_normalize_text)
    long_df = long_df.dropna(subset=["referral_type"])
    long_df = long_df.drop_duplicates(subset=["ID", "referral_type"])

    if long_df.empty:
        return []

    grouped = (
        long_df.groupby("referral_type", dropna=False)
        .agg(
            referrals=("ID", "nunique"),
            engaged=("has_encounter", "sum"),
            port=("has_od", "sum"),
            avg_touchpoints=("touchpoints_total", "mean"),
        )
        .reset_index()
    )

    total_referrals = int(referrals_df["ID"].nunique()) or int(referrals_df.shape[0])
    rows: list[EncounterTableRow] = []
    for _, row in grouped.sort_values("referrals", ascending=False).iterrows():
        referrals_count = int(row["referrals"])
        engaged_count = int(row["engaged"])
        port_count = int(row["port"])
        avg_touchpoints = (
            round(float(row["avg_touchpoints"]), 1) if not pd.isna(row["avg_touchpoints"]) else None
        )
        rows.append(
            {
                "label": _normalize_text(row["referral_type"]),
                "referral_count": referrals_count,
                "share_pct": _pct(referrals_count, total_referrals),
                "engaged_count": engaged_count,
                "engagement_pct": _pct(engaged_count, referrals_count),
                "port_count": port_count,
                "port_pct": _pct(port_count, referrals_count),
                "avg_touchpoints": avg_touchpoints,
            }
        )
    return rows


def _build_encounter_type_tables(referrals_df: pd.DataFrame) -> list[EncounterTypeTable]:
    tables: list[EncounterTypeTable] = []
    for column, title, subtitle in [
        ("encounter_type_cat1", "Encounter type — Cat 1", "Primary referral focus"),
        ("encounter_type_cat2", "Encounter type — Cat 2", "Secondary referral details"),
        ("encounter_type_cat3", "Encounter type — Cat 3", "Tertiary referral details"),
    ]:
        rows = _aggregate_referrals_by_column(referrals_df, column)
        if rows:
            tables.append({"title": title, "subtitle": subtitle, "rows": rows})
    return tables


def _build_referral_type_table(referrals_df: pd.DataFrame) -> ReferralTypeTable:
    rows = _aggregate_referral_type_rows(referrals_df)
    return {
        "title": "Referral reason mix",
        "subtitle": "Unique referrals containing each referral reason across slots 1–5.",
        "rows": rows,
    }


def _encounter_table_insights(encounter_tables: list[EncounterTypeTable]) -> list[str]:
    if not encounter_tables:
        return []

    insights: list[str] = []
    cat1_rows = encounter_tables[0]["rows"]
    if cat1_rows:
        top_cat1 = cat1_rows[0]
        insights.append(
            f"{top_cat1['label']} leads Cat 1 requests, representing {top_cat1['share_pct']:.1f}% of referrals."
        )

    high_engagement_row: tuple[str, EncounterTableRow] | None = None
    for table in encounter_tables:
        for row in table["rows"]:
            if row["referral_count"] < 3:
                continue
            if (
                high_engagement_row is None
                or row["engagement_pct"] > high_engagement_row[1]["engagement_pct"]
            ):
                high_engagement_row = (table["title"], row)

    if high_engagement_row:
        table_title, row = high_engagement_row
        insights.append(
            f"{row['label']} ({table_title}) converts {row['engagement_pct']:.1f}% of referrals into encounters."
        )

    return insights


def _referral_rows_insights(referral_rows: list[EncounterTableRow]) -> list[str]:
    if not referral_rows:
        return []

    insights: list[str] = []
    top_referral = referral_rows[0]
    insights.append(
        f"{top_referral['label']} appears on {top_referral['share_pct']:.1f}% of referrals, covering the widest set of needs."
    )

    def _port_pct(row: EncounterTableRow) -> float:
        return float(row["port_pct"])

    high_port_row = next(
        (
            row
            for row in sorted(referral_rows, key=_port_pct, reverse=True)
            if row["referral_count"] >= 3 and row["port_pct"] > 0
        ),
        None,
    )
    if high_port_row:
        insights.append(
            f"Referrals mentioning {high_port_row['label']} link to PORT follow-ups {high_port_row['port_pct']:.1f}% of the time."
        )

    return insights


def _build_referral_table_insights(
    encounter_tables: list[EncounterTypeTable],
    referral_rows: list[EncounterTableRow],
) -> list[str]:
    insights = _encounter_table_insights(encounter_tables)
    insights.extend(_referral_rows_insights(referral_rows))
    return insights[:3]


def _prepare_referrals_insights_context() -> dict[str, object]:
    datasets = _load_referral_datasets()
    referrals_df = datasets["referrals"]
    encounters_df = datasets["encounters"]
    od_df = datasets["odreferrals"]

    if referrals_df.empty:
        empty_cards = _build_referrals_summary_cards(0, 0, 0, 0.0, None, 0, 0.0, 0.0)
        return {
            "summary_cards": empty_cards,
            "top_agencies": [],
            "encounter_type_tables": [],
            "referral_type_table": {"title": "Referral reason mix", "subtitle": "", "rows": []},
            "table_insights": [],
            "data_is_empty": True,
        }

    enriched_referrals = _attach_touchpoint_metrics(referrals_df, encounters_df, od_df)

    total_referrals = int(enriched_referrals.shape[0])
    unique_patients = _safe_nunique(enriched_referrals["patient_ID"])
    referrals_with_encounters = int(enriched_referrals["has_encounter"].sum())
    referrals_with_od = int(enriched_referrals["has_od"].sum())
    engagement_rate = _pct(referrals_with_encounters, total_referrals)
    port_rate = _pct(referrals_with_od, total_referrals)
    avg_touchpoints = (
        float(enriched_referrals["touchpoints_total"].mean()) if total_referrals else 0.0
    )
    if "days_to_first_encounter" in enriched_referrals.columns:
        median_days = enriched_referrals["days_to_first_encounter"].dropna().median()
    else:
        median_days = None

    summary_cards = _build_referrals_summary_cards(
        total_referrals,
        unique_patients,
        referrals_with_encounters,
        engagement_rate,
        median_days,
        referrals_with_od,
        port_rate,
        avg_touchpoints,
    )

    top_agencies = _build_top_agency_rows(enriched_referrals)
    encounter_type_tables = _build_encounter_type_tables(enriched_referrals)
    referral_type_table = _build_referral_type_table(enriched_referrals)
    table_insights = _build_referral_table_insights(
        encounter_type_tables,
        referral_type_table["rows"],
    )

    return {
        "summary_cards": summary_cards,
        "top_agencies": top_agencies,
        "encounter_type_tables": encounter_type_tables,
        "referral_type_table": referral_type_table,
        "table_insights": table_insights,
        "data_is_empty": False,
    }


def odreferrals_insights_fragment(request):
    context = _prepare_odreferrals_insights_context()
    return render(request, "dashboard/partials/odreferrals_insights_fragment.html", context)


# Encounters placeholder insights
def _build_encounter_summary_cards(
    enriched: pd.DataFrame, encounters_df: pd.DataFrame
) -> list[dict[str, object]]:
    total_encounters = int(encounters_df.shape[0])
    encounter_patients_df = enriched[enriched["has_encounter"]]
    encounter_patient_count = int(encounter_patients_df.shape[0])
    engaged_count = int(enriched[enriched["has_touchpoint"]].shape[0])
    avg_per_patient = (
        round(total_encounters / encounter_patient_count, 1) if encounter_patient_count else 0.0
    )
    encounter_only_df = encounter_patients_df[encounter_patients_df["referrals"] == 0]
    encounter_only_count = int(encounter_only_df.shape[0])
    high_frequency_df = encounter_patients_df[encounter_patients_df["encounters"] >= 3]
    high_frequency_count = int(high_frequency_df.shape[0])
    port_overlap_count = int(encounter_patients_df[encounter_patients_df["has_od"]].shape[0])

    return [
        {
            "label": "Documented encounters",
            "value": f"{total_encounters:,}",
            "description": "Total CPM encounter records available for analysis.",
        },
        {
            "label": "Patients with encounters",
            "value": f"{encounter_patient_count:,}",
            "description": (
                f"{_pct(encounter_patient_count, engaged_count)}% of engaged patients have CPM encounters on file."
            ),
        },
        {
            "label": "Encounter-only patients",
            "value": f"{encounter_only_count:,}",
            "description": (
                "Have CPM visit history but no referral logged—close the loop for reporting."
            ),
        },
        {
            "label": "High-frequency visits",
            "value": f"{high_frequency_count:,}",
            "description": (f"3+ encounters; average per patient currently {avg_per_patient:.1f}."),
        },
        {
            "label": "PORT overlap",
            "value": f"{port_overlap_count:,}",
            "description": (
                f"{_pct(port_overlap_count, encounter_patient_count)}% of encounter patients also have PORT follow-ups."
            ),
        },
    ]


def _build_encounter_top_patients(enriched: pd.DataFrame) -> list[dict[str, object]]:
    encounter_patients_df = enriched[enriched["has_encounter"]]
    if encounter_patients_df.empty:
        return []

    sorted_df = encounter_patients_df.sort_values(
        ["encounters", "total_touchpoints", "referrals", "patient_id"],
        ascending=[False, False, False, True],
    ).head(10)

    rows: list[dict[str, object]] = []
    for _, row in sorted_df.iterrows():
        age_value: object
        age_raw = row.get("age")
        if age_raw is None or (isinstance(age_raw, float) and pd.isna(age_raw)):
            age_value = "Unknown"
        else:
            try:
                age_value = int(age_raw)
            except Exception:
                age_value = "Unknown"

        rows.append(
            {
                "patient_id": int(row["patient_id"]),
                "encounters": int(row["encounters"]),
                "total_touchpoints": int(row["total_touchpoints"]),
                "referrals": int(row["referrals"]),
                "odreferrals": int(row["odreferrals"]),
                "insurance": _normalize_text(row.get("insurance")),
                "pcp_agency": _normalize_text(row.get("pcp_agency")),
                "zip_code": _normalize_text(row.get("zip_code")),
                "age": age_value,
                "sex": _normalize_text(row.get("sex")),
                "sud": _format_bool_label(row.get("sud")),
                "behavioral_health": _format_bool_label(row.get("behavioral_health")),
            }
        )
    return rows


def _build_top_encounter_counts(enriched: pd.DataFrame, limit: int = 5) -> list[dict[str, object]]:
    encounter_patients_df = enriched[enriched["has_encounter"]]
    if encounter_patients_df.empty:
        return []

    total_encounters = int(encounter_patients_df["encounters"].sum())
    if total_encounters == 0:
        return []

    subset = encounter_patients_df.sort_values(
        ["encounters", "total_touchpoints", "patient_id"],
        ascending=[False, False, True],
    ).head(limit)

    rows: list[dict[str, object]] = []
    for _, row in subset.iterrows():
        encounters_count = int(row["encounters"])
        rows.append(
            {
                "patient_id": int(row["patient_id"]),
                "encounters": encounters_count,
                "share_pct": _pct(encounters_count, total_encounters),
                "referrals": int(row["referrals"]),
                "odreferrals": int(row["odreferrals"]),
            }
        )
    return rows


def _build_encounter_pcp_table(encounters_df: pd.DataFrame) -> list[dict[str, object]]:
    if encounters_df.empty or "pcp_agency" not in encounters_df.columns:
        return []

    df = encounters_df[["pcp_agency", "patient_ID"]].copy()
    df["pcp_agency"] = _norm_series(df["pcp_agency"])

    grouped = (
        df.groupby("pcp_agency", dropna=False)
        .agg(
            encounters=("pcp_agency", "count"),
            unique_patients=("patient_ID", _safe_nunique),
        )
        .reset_index()
    )

    if grouped.empty:
        return []

    total_encounters = int(grouped["encounters"].sum())
    rows: list[dict[str, object]] = []
    for _, row in grouped.sort_values("encounters", ascending=False).iterrows():
        encounters_count = int(row["encounters"])
        unique_patients = int(row["unique_patients"])
        avg_per_patient = round(encounters_count / unique_patients, 1) if unique_patients else 0.0
        rows.append(
            {
                "pcp_agency": _normalize_text(row["pcp_agency"]),
                "encounters": encounters_count,
                "share_pct": _pct(encounters_count, total_encounters),
                "unique_patients": unique_patients,
                "avg_per_patient": avg_per_patient,
            }
        )
    return rows


def _build_encounter_type_percentage_tables(encounters_df: pd.DataFrame) -> list[dict[str, object]]:
    if encounters_df.empty:
        return []

    total_encounters = int(encounters_df.shape[0])
    if total_encounters == 0:
        return []

    tables: list[dict[str, object]] = []
    for column, title, subtitle in [
        ("encounter_type_cat1", "Encounter Type — Cat 1", "Primary encounter classification"),
        ("encounter_type_cat2", "Encounter Type — Cat 2", "Secondary encounter details"),
        ("encounter_type_cat3", "Encounter Type — Cat 3", "Tertiary encounter specifics"),
    ]:
        if column not in encounters_df.columns:
            continue

        df = encounters_df[[column, "patient_ID"]].copy()
        df[column] = _norm_series(df[column])

        grouped = (
            df.groupby(column, dropna=False)
            .agg(
                encounters=("patient_ID", "count"),
                unique_patients=("patient_ID", _safe_nunique),
            )
            .reset_index()
        )

        if grouped.empty:
            continue

        rows: list[dict[str, object]] = []
        for _, row in grouped.sort_values("encounters", ascending=False).iterrows():
            encounters_count = int(row["encounters"])
            unique_patients = int(row["unique_patients"])
            avg_per_patient = (
                round(encounters_count / unique_patients, 1) if unique_patients else 0.0
            )
            rows.append(
                {
                    "label": _normalize_text(row[column]),
                    "encounters": encounters_count,
                    "share_pct": _pct(encounters_count, total_encounters),
                    "unique_patients": unique_patients,
                    "avg_per_patient": avg_per_patient,
                }
            )

        tables.append({"title": title, "subtitle": subtitle, "rows": rows})

    return tables


def _prepare_encounters_insights_context() -> dict[str, object]:
    datasets = _load_patient_touchpoint_datasets()
    encounters_df = datasets.get("encounters", pd.DataFrame())
    if encounters_df.empty:
        return {
            "data_is_empty": True,
            "summary_cards": [],
            "top_patients": [],
            "top_encounter_counts": [],
            "pcp_agency_interactions": [],
            "encounter_type_percentages": [],
        }

    enriched = _enrich_patient_touchpoints(datasets)
    if enriched.empty or not enriched["has_encounter"].any():
        return {
            "data_is_empty": True,
            "summary_cards": [],
            "top_patients": [],
            "top_encounter_counts": [],
            "pcp_agency_interactions": [],
            "encounter_type_percentages": [],
        }

    summary_cards = _build_encounter_summary_cards(enriched, encounters_df)
    top_patients = _build_encounter_top_patients(enriched)
    top_encounter_counts = _build_top_encounter_counts(enriched)
    pcp_agency_interactions = _build_encounter_pcp_table(encounters_df)
    encounter_type_percentages = _build_encounter_type_percentage_tables(encounters_df)

    return {
        "data_is_empty": False,
        "summary_cards": summary_cards,
        "top_patients": top_patients,
        "top_encounter_counts": top_encounter_counts,
        "pcp_agency_interactions": pcp_agency_interactions,
        "encounter_type_percentages": encounter_type_percentages,
    }


# OD Referrals
OD_REFERRALS_WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    return series.map({True: "Yes", False: "No"}).fillna("Unknown")


def _odreferrals_clean_dates(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    dates = pd.to_datetime(series, errors="coerce").dropna()
    if dates.empty:
        return None
    with suppress(TypeError, AttributeError, ValueError):
        dates = dates.dt.tz_convert(None)
    with suppress(TypeError, AttributeError, ValueError):
        dates = dates.dt.tz_localize(None)
    return dates


def _odreferrals_monthly_insights(dates: pd.Series) -> list[dict[str, object]]:
    monthly_series = dates.dt.to_period("M").value_counts().sort_index()
    if monthly_series.empty:
        return []
    latest_period = monthly_series.index[-1]
    latest_count = _to_int(monthly_series.iloc[-1])
    latest_label = latest_period.to_timestamp().strftime("%b %Y")

    items: list[dict[str, object]] = [
        {"label": "Most recent month", "value": f"{latest_label} — {latest_count}"},
    ]

    if len(monthly_series) >= 2:
        prev_count = _to_int(monthly_series.iloc[-2])
        delta = latest_count - prev_count
        delta_pct = _pct(delta, prev_count) if prev_count else 0.0
        items.append({"label": "Month-over-month", "value": f"{delta:+d} ({delta_pct:+.1f}%)"})

    yoy_period = latest_period - 12
    yoy_count = monthly_series.get(yoy_period)
    if yoy_count is not None and not pd.isna(yoy_count):
        yoy_value = _to_int(yoy_count)
        yoy_delta = latest_count - yoy_value
        yoy_pct = _pct(yoy_delta, yoy_value) if yoy_value else 0.0
        items.append({"label": "Year-over-year", "value": f"{yoy_delta:+d} ({yoy_pct:+.1f}%)"})

    trailing_12 = _to_int(monthly_series.tail(12).sum())
    items.append({"label": "Trailing 12 months", "value": trailing_12})

    return items


def _odreferrals_weekday_insights(dates: pd.Series) -> list[dict[str, object]]:
    weekdays = dates.dt.day_name()
    total_records = len(weekdays)
    if total_records == 0:
        return []
    weekday_counts = weekdays.value_counts().reindex(OD_REFERRALS_WEEKDAY_ORDER, fill_value=0)
    items: list[dict[str, object]] = []
    if weekday_counts.max() > 0:
        top_day = str(weekday_counts.idxmax())
        items.append(
            {
                "label": "Busiest day",
                "value": f"{top_day} ({_pct(_to_int(weekday_counts.max()), total_records):.1f}%)",
            }
        )
    weekend_total = _to_int(weekday_counts.get("Saturday", 0)) + _to_int(
        weekday_counts.get("Sunday", 0)
    )
    items.append(
        {
            "label": "Weekend share",
            "value": f"{_pct(weekend_total, total_records):.1f}% of referrals",
        }
    )
    return items


def _build_odreferrals_chart_insights(df_all: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    if df_all.empty:
        return {}

    df = df_all.copy()
    for col in ["narcan_given", "referral_to_sud_agency"]:
        if col in df.columns:
            df[col] = _normalize_bool_series(df[col])

    insights: dict[str, list[dict[str, object]]] = {}

    dates = _odreferrals_clean_dates(df.get("od_date"))
    if dates is not None:
        monthly_items = _odreferrals_monthly_insights(dates)
        if monthly_items:
            insights["odreferrals_counts_monthly"] = monthly_items
        weekday_items = _odreferrals_weekday_insights(dates)
        if weekday_items:
            insights["odreferrals_counts_weekday"] = weekday_items

    for field, label in [
        ("referral_source", "Referral Source"),
        ("suspected_drug", "Suspected Drug"),
        ("cpm_disposition", "CPM Disposition"),
        ("living_situation", "Living Situation"),
        ("engagement_location", "Engagement Location"),
        ("narcan_given", "Narcan Given"),
        ("referral_to_sud_agency", "Referral to SUD Agency"),
    ]:
        tips = _insights_categorical(df, field, label)
        if tips:
            insights[field] = tips

    return insights


def _load_odreferrals_dataframe() -> pd.DataFrame:
    fields = [
        "ID",
        "patient_id",
        "od_date",
        "referral_source",
        "suspected_drug",
        "cpm_disposition",
        "living_situation",
        "engagement_location",
        "narcan_given",
        "referral_to_sud_agency",
        "disposition",
        "leave_behind_narcan_amount",
        "persons_trained",
    ]
    values = list(ODReferrals.objects.all().values(*fields))
    df = _df_from_queryset(values, fields)

    if df.empty:
        return df

    if "od_date" in df.columns:
        df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
        with suppress(TypeError, AttributeError, ValueError):
            df["od_date"] = df["od_date"].dt.tz_convert(None)
        with suppress(TypeError, AttributeError, ValueError):
            df["od_date"] = df["od_date"].dt.tz_localize(None)

    df["narcan_flag"] = 0
    if "narcan_given" in df.columns:
        df["narcan_flag"] = df["narcan_given"].fillna(False).astype(bool).astype(int)

    df["sud_flag"] = 0
    if "referral_to_sud_agency" in df.columns:
        df["sud_flag"] = df["referral_to_sud_agency"].fillna(False).astype(bool).astype(int)

    fatal_terms = {"fatal", "death", "deceased", "doa", "cpr attempted", "cpr"}
    df["fatal_flag"] = 0
    if "disposition" in df.columns:
        disposition_series = df["disposition"].fillna("").astype(str).str.strip().str.lower()
        df["fatal_flag"] = disposition_series.isin(fatal_terms).astype(int)

    return df


def _compute_odreferrals_metrics(df: pd.DataFrame) -> dict[str, int | float | str | None]:
    total = _to_int(df.shape[0])
    patient_series = df.get("patient_id", pd.Series(dtype=object))
    unique_patients = _safe_nunique(patient_series)

    narcan_count = _to_int(df.get("narcan_flag", pd.Series(dtype=int)).sum()) if total else 0
    narcan_rate = _pct(narcan_count, total)

    sud_count = _to_int(df.get("sud_flag", pd.Series(dtype=int)).sum()) if total else 0
    sud_rate = _pct(sud_count, total)

    fatal_count = _to_int(df.get("fatal_flag", pd.Series(dtype=int)).sum()) if total else 0
    fatal_rate = _pct(fatal_count, total)

    top_agency_label: str | None = None
    top_agency_pct = 0.0
    unique_agencies = 0
    if "referral_agency" in df.columns:
        agency_series = _norm_series(df["referral_agency"])
        agency_counts = agency_series.value_counts()
        unique_agencies = _to_int(agency_counts.size)
        if not agency_counts.empty:
            top_agency_label = str(agency_counts.index[0])
            top_agency_pct = _pct(_to_int(agency_counts.iloc[0]), total)

    weekend_pct = 0.0
    busiest_day: str | None = None
    if "od_date" in df.columns:
        valid_dates = df["od_date"].dropna()
        if not valid_dates.empty:
            weekdays = valid_dates.dt.day_name()
            weekday_counts = weekdays.value_counts().reindex(
                OD_REFERRALS_WEEKDAY_ORDER, fill_value=0
            )
            if weekday_counts.max() > 0:
                busiest_day = str(weekday_counts.idxmax())
            weekend_total = _to_int(weekday_counts.get("Saturday", 0)) + _to_int(
                weekday_counts.get("Sunday", 0)
            )
            weekend_pct = _pct(weekend_total, len(weekdays))

    return {
        "total": total,
        "unique_patients": unique_patients,
        "narcan_count": narcan_count,
        "narcan_rate": narcan_rate,
        "sud_count": sud_count,
        "sud_rate": sud_rate,
        "fatal_count": fatal_count,
        "fatal_rate": fatal_rate,
        "unique_agencies": unique_agencies,
        "top_agency_label": top_agency_label,
        "top_agency_pct": top_agency_pct,
        "weekend_pct": weekend_pct,
        "busiest_day": busiest_day,
    }


def _build_odreferrals_summary_cards(
    metrics: dict[str, int | float | str | None],
) -> list[dict[str, object]]:
    total = _to_int(metrics.get("total"))
    unique_patients = _to_int(metrics.get("unique_patients"))
    narcan_count = _to_int(metrics.get("narcan_count"))
    narcan_rate = _to_float(metrics.get("narcan_rate"))
    sud_count = _to_int(metrics.get("sud_count"))
    sud_rate = _to_float(metrics.get("sud_rate"))
    unique_agencies = _to_int(metrics.get("unique_agencies"))
    top_agency_label = metrics.get("top_agency_label")
    top_agency_pct = _to_float(metrics.get("top_agency_pct"))

    if isinstance(top_agency_label, str) and top_agency_label:
        agency_desc = f"Top contributor: {top_agency_label} ({top_agency_pct:.1f}% of referrals)."
    else:
        agency_desc = "No referral agency data captured yet."

    return [
        {
            "label": "OD referrals logged",
            "value": f"{total:,}",
            "description": "Documented overdose response records routed to CPM.",
        },
        {
            "label": "Individuals involved",
            "value": f"{unique_patients:,}",
            "description": "Unique patient IDs present across overdose referrals.",
        },
        {
            "label": "Narcan administered",
            "value": f"{narcan_count:,} ({narcan_rate:.1f}%)",
            "description": "Events where naloxone was administered before or during CPM response.",
        },
        {
            "label": "SUD handoffs",
            "value": f"{sud_count:,} ({sud_rate:.1f}%)",
            "description": "Referrals that included a linkage to SUD treatment partners.",
        },
        {
            "label": "Referral agencies engaged",
            "value": f"{unique_agencies:,}",
            "description": agency_desc,
        },
    ]


def _build_odreferrals_agency_rows(
    df: pd.DataFrame, total_referrals: int
) -> list[dict[str, object]]:
    if df.empty or "referral_agency" not in df.columns or total_referrals == 0:
        return []

    subset = df.copy()
    subset["referral_agency"] = subset["referral_agency"].apply(_normalize_text)

    grouped = (
        subset.groupby("referral_agency", dropna=False)
        .agg(
            referrals=("ID", "count"),
            unique_patients=("patient_id", _safe_nunique),
            narcan_events=("narcan_flag", "sum"),
            sud_referrals=("sud_flag", "sum"),
            fatal_events=("fatal_flag", "sum"),
        )
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for _, row in grouped.sort_values("referrals", ascending=False).head(10).iterrows():
        referrals_count = _to_int(row["referrals"])
        if referrals_count == 0:
            continue
        narcan_events = _to_int(row["narcan_events"])
        sud_referrals = _to_int(row["sud_referrals"])
        rows.append(
            {
                "agency": _normalize_text(row["referral_agency"]),
                "referrals": referrals_count,
                "share_pct": _pct(referrals_count, total_referrals),
                "narcan_events": narcan_events,
                "narcan_pct": _pct(narcan_events, referrals_count),
                "sud_referrals": sud_referrals,
                "sud_pct": _pct(sud_referrals, referrals_count),
                "fatal_events": _to_int(row["fatal_events"]),
                "unique_patients": _to_int(row["unique_patients"]),
            }
        )
    return rows


def _build_odreferrals_categorical_table(
    df: pd.DataFrame, column: str, total_referrals: int, limit: int = 8
) -> list[dict[str, object]]:
    if df.empty or column not in df.columns or total_referrals == 0:
        return []

    subset = df[["ID", column, "narcan_flag", "sud_flag"]].copy()
    subset[column] = subset[column].apply(_normalize_text)
    subset = subset.dropna(subset=[column])

    grouped = (
        subset.groupby(column, dropna=False)
        .agg(
            referrals=("ID", "count"),
            narcan_events=("narcan_flag", "sum"),
            sud_referrals=("sud_flag", "sum"),
        )
        .reset_index()
        .rename(columns={column: "label"})
    )

    rows: list[dict[str, object]] = []
    for _, row in grouped.sort_values("referrals", ascending=False).head(limit).iterrows():
        referrals_count = _to_int(row["referrals"])
        if referrals_count == 0:
            continue
        narcan_events = _to_int(row["narcan_events"])
        sud_referrals = _to_int(row["sud_referrals"])
        rows.append(
            {
                "label": _normalize_text(row["label"]),
                "referrals": referrals_count,
                "share_pct": _pct(referrals_count, total_referrals),
                "narcan_pct": _pct(narcan_events, referrals_count),
                "sud_pct": _pct(sud_referrals, referrals_count),
            }
        )

    return rows


def _build_odreferrals_takeaways(
    metrics: dict[str, int | float | str | None],
    agency_rows: list[dict[str, object]],
    suspected_rows: list[dict[str, object]],
    engagement_rows: list[dict[str, object]],
) -> list[str]:
    takeaways: list[str] = []

    if agency_rows:
        top_agency = agency_rows[0]
        takeaways.append(
            f"{top_agency['agency']} submits {top_agency['share_pct']:.1f}% of overdose referrals and routes SUD handoffs {top_agency['sud_pct']:.1f}% of the time."
        )

    if suspected_rows:
        top_drug = suspected_rows[0]
        takeaways.append(
            f"{top_drug['label']} is the leading suspected substance, appearing in {top_drug['share_pct']:.1f}% of cases."
        )

    if engagement_rows:
        top_location = engagement_rows[0]
        takeaways.append(
            f"{top_location['label']} hosts {top_location['share_pct']:.1f}% of CPM engagements, with naloxone used in {top_location['narcan_pct']:.1f}% of those events."
        )

    weekend_pct = _to_float(metrics.get("weekend_pct"))
    busiest_day = metrics.get("busiest_day")
    if weekend_pct > 0.0:
        if isinstance(busiest_day, str) and busiest_day:
            takeaways.append(
                f"{busiest_day} is the busiest day for overdose referrals, while weekends represent {weekend_pct:.1f}% of activity."
            )
        else:
            takeaways.append(f"Weekends account for {weekend_pct:.1f}% of overdose referrals.")

    fatal_rate = _to_float(metrics.get("fatal_rate"))
    fatal_count = _to_int(metrics.get("fatal_count"))
    if fatal_rate > 0.0:
        takeaways.append(
            f"{fatal_count} records ({fatal_rate:.1f}%) document fatal or CPR-required outcomes—prioritize rapid partner notifications."
        )

    return takeaways[:4]


def _prepare_odreferrals_insights_context() -> dict[str, object]:
    df = _load_odreferrals_dataframe()
    if df.empty:
        return {
            "data_is_empty": True,
            "summary_cards": [],
            "top_agencies": [],
            "suspected_drugs": [],
            "engagement_locations": [],
            "referral_sources": [],
            "living_situations": [],
            "takeaways": [],
        }

    metrics = _compute_odreferrals_metrics(df)
    summary_cards = _build_odreferrals_summary_cards(metrics)

    total_referrals = _to_int(metrics.get("total"))
    agency_rows = _build_odreferrals_agency_rows(df, total_referrals)
    suspected_rows = _build_odreferrals_categorical_table(df, "suspected_drug", total_referrals)
    engagement_rows = _build_odreferrals_categorical_table(
        df, "engagement_location", total_referrals
    )
    referral_source_rows = _build_odreferrals_categorical_table(
        df, "referral_source", total_referrals
    )
    living_situation_rows = _build_odreferrals_categorical_table(
        df, "living_situation", total_referrals
    )

    takeaways = _build_odreferrals_takeaways(metrics, agency_rows, suspected_rows, engagement_rows)

    return {
        "data_is_empty": False,
        "summary_cards": summary_cards,
        "top_agencies": agency_rows,
        "suspected_drugs": suspected_rows,
        "engagement_locations": engagement_rows,
        "referral_sources": referral_source_rows,
        "living_situations": living_situation_rows,
        "takeaways": takeaways,
    }


OD_REFERRALS_LABEL_MAP: dict[str, str] = {
    "odreferrals_counts_monthly": "OD referrals by month",
    "odreferrals_counts_weekday": "OD referrals by weekday",
    "referral_source": "Referral source",
    "suspected_drug": "Suspected substance",
    "living_situation": "Living situation",
    "engagement_location": "Engagement location",
    "narcan_given": "Narcan administered",
    "referral_to_sud_agency": "SUD referrals",
}


OD_REFERRALS_RATIONALE_MAP: dict[str, str] = {
    "odreferrals_counts_monthly": (
        "Monthly trends show when overdose activity spikes so you can staff CPM and partner teams accordingly."
    ),
    "odreferrals_counts_weekday": (
        "Day-of-week patterns highlight when overdose referrals peak, informing shift coverage and outreach campaigns."
    ),
    "referral_source": (
        "Knowing who flags overdoses reveals partnership reach and where to reinforce notifications."
    ),
    "suspected_drug": (
        "Substance mix guides harm reduction supplies and treatment referrals tailored to current trends."
    ),
    "living_situation": (
        "Housing context helps tailor engagement approaches and partner connections."
    ),
    "engagement_location": (
        "Locations show where CPM teams most often meet clients, aiding deployment and cross-agency coordination."
    ),
    "narcan_given": (
        "Naloxone usage rates signal training needs and kit distribution effectiveness."
    ),
    "referral_to_sud_agency": (
        "SUD treatment linkages demonstrate how often overdose follow-ups convert into recovery support."
    ),
}


OD_REFERRALS_DISPLAY_FIELDS = [
    "odreferrals_counts_monthly",
    "odreferrals_counts_weekday",
    "referral_source",
    "suspected_drug",
    "living_situation",
    "engagement_location",
    "narcan_given",
    "referral_to_sud_agency",
]


OD_REFERRALS_COL_SPAN: dict[str, int] = {
    "odreferrals_counts_monthly": 2,
    "odreferrals_counts_weekday": 2,
    "narcan_given": 1,
    "referral_to_sud_agency": 1,
}

OD_REFERRALS_STORY_CARDS = [
    {
        "eyebrow": "Story",
        "title": "Critical overdose windows",
        "lede": (
            "Monthly cadence now displays percent share and a rolling average so you can see when surges "
            "are trend versus spike."
        ),
        "bullets": [
            "Weekday bars surface share-of-total alongside counts to focus scheduling.",
            "Rolling three-month line pins down sustained increases for board updates.",
            "Percent annotations simplify rate comparisons if total referrals change after outreach pushes.",
        ],
        "callout": "Map the peaks against staffing rosters to close coverage gaps.",
    },
    {
        "eyebrow": "Response",
        "title": "Narcan and SUD linkages",
        "lede": (
            "Donut charts now couple totals with share so you can quantify how often harm reduction closes the loop."
        ),
        "bullets": [
            "Narcan deployment and treatment referrals show proportion of cases supported.",
            "Hover to see exact counts for grant documentation without leaving the dashboard.",
            "Large Unknown segments signal documentation clean-up needs.",
        ],
        "callout": "Share percent improvements directly with harm-reduction partners.",
    },
    {
        "eyebrow": "Next steps",
        "title": "Align outreach and prevention",
        "lede": (
            "Use the normalized suspected drug and living situation shares to tailor messaging and kit placement."
        ),
        "bullets": [
            "Trace spikes back to referral sources to reinforce what is working.",
            "Pair engagement location shares with map layers for deployment planning.",
            "Document emerging patterns (e.g., evening rise) in the weekly operations brief.",
        ],
        "callout": "Revisit assumptions monthly so the percent context remains accurate.",
    },
]


def _ordered_odreferral_fields(df: pd.DataFrame) -> list[str]:
    fields: list[str] = []
    fields.extend(["odreferrals_counts_monthly"])
    for field in [
        "referral_agency",
    ]:
        if field in df.columns:
            fields.append(field)
    return fields


def odreferrals(request):
    theme = get_theme_from_request(request)
    df_all = _load_odreferrals_dataframe()
    ordered_fields = _ordered_odreferral_fields(df_all)

    chart_cards = [
        {
            "field": field,
            "label": OD_REFERRALS_LABEL_MAP.get(field, field.replace("_", " ").title()),
            "col_span": OD_REFERRALS_COL_SPAN.get(field, 2),
        }
        for field in ordered_fields
    ]

    # Add the repeat overdoses timeline chart
    fig_repeats_scatter = _chart_html(build_chart_repeats_scatter(theme=theme))

    # Add hotspot map and statistics
    zoom_mode = "full" if request.user.is_authenticated else "restricted"
    fig_od_map = _chart_html(build_chart_od_map(theme=theme, zoom_mode=zoom_mode))
    hotspot_stats = _compute_hotspot_stats()

    # Add Narcan administration analysis
    narcan_grid_chart = _chart_html(build_narcan_administration_grid(theme=theme))
    narcan_timeline_chart = _chart_html(build_narcan_response_timeline(theme=theme))
    narcan_stats = build_narcan_stats()

    context = {
        "chart_cards": chart_cards,
        "fig_repeats_scatter": fig_repeats_scatter,
        "fig_od_map": fig_od_map,
        "hotspot_stats": hotspot_stats,
        "narcan_grid_chart": narcan_grid_chart,
        "narcan_timeline_chart": narcan_timeline_chart,
        "narcan_stats": narcan_stats,
        "theme": theme,
        "story_cards": OD_REFERRALS_STORY_CARDS,
    }
    updated_on = date(2025, 11, 30)
    context.update(
        {
            "page_header_updated_at": updated_on,
            "page_header_updated_at_iso": updated_on.isoformat(),
            "page_header_read_time": "9 min read",
        }
    )
    return render(request, "dashboard/odreferrals.html", context)


def od_weekday_detail(request):
    """Serve detailed weekday analysis modal content."""
    theme = get_theme_from_request(request)

    # Build weekday chart
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(theme=theme, fields=["odreferrals_counts_weekday"])
    weekday_chart = charts.get("odreferrals_counts_weekday", "")

    # Get quick stats for weekday
    quick_stats = _build_odreferrals_quick_stats()
    weekday_stats = quick_stats.get("odreferrals_counts_weekday", [])

    context = {
        "theme": theme,
        "title": "Overdose Referrals by Day of Week",
        "description": "Detailed analysis of overdose patterns across the week",
        "weekday_chart": weekday_chart,
        "weekday_stats": weekday_stats,
    }

    return render(request, "dashboard/partials/od_weekday_detail.html", context)


def od_referral_source_detail(request):
    """Serve detailed referral source analysis modal content."""
    theme = get_theme_from_request(request)

    # Build referral source chart
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(theme=theme, fields=["referral_source"])
    referral_source_chart = charts.get("referral_source", "")

    # Calculate referral source stats
    import pandas as pd

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values(
        "referral_source", "disposition", "referral_to_sud_agency"
    )
    data = list(qs)

    source_stats = []
    if data:
        df = pd.DataFrame.from_records(data)
        df = df[df["referral_source"].notna()]

        if not df.empty:
            total_referrals = len(df)

            # Top source
            top_source = df["referral_source"].value_counts().index[0] if len(df) > 0 else "N/A"
            top_source_count = (
                int(df["referral_source"].value_counts().iloc[0]) if len(df) > 0 else 0
            )
            top_source_pct = (
                round((top_source_count / total_referrals) * 100, 1) if total_referrals > 0 else 0
            )

            # Fatal referrals
            fatal_count = len(df[df["disposition"].isin(["CPR attempted", "DOA"])])
            round((fatal_count / total_referrals) * 100, 1) if total_referrals > 0 else 0

            # SUD referral success
            sud_success = df["referral_to_sud_agency"].sum()
            round((sud_success / total_referrals) * 100, 1) if total_referrals > 0 else 0

            source_stats = [
                {
                    "label": "Top Source",
                    "value": top_source,
                    "description": f"{top_source_count} referrals • {top_source_pct}%",
                }
            ]

    context = {
        "theme": theme,
        "referral_source_chart": referral_source_chart,
        "source_stats": source_stats,
    }

    return render(request, "dashboard/partials/od_referral_source_detail.html", context)


def od_suspected_drug_detail(request):
    """Serve detailed suspected drug analysis modal content."""
    theme = get_theme_from_request(request)

    # Build suspected drug chart
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(theme=theme, fields=["suspected_drug"])
    suspected_drug_chart = charts.get("suspected_drug", "")

    # Calculate suspected drug stats
    import pandas as pd

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values("suspected_drug", "disposition", "narcan_given")
    data = list(qs)

    drug_stats = []
    if data:
        df = pd.DataFrame.from_records(data)
        df = df[df["suspected_drug"].notna()]

        if not df.empty:
            total_cases = len(df)

            # Top drug
            top_drug = df["suspected_drug"].value_counts().index[0] if len(df) > 0 else "N/A"
            top_drug_count = int(df["suspected_drug"].value_counts().iloc[0]) if len(df) > 0 else 0
            top_drug_pct = round((top_drug_count / total_cases) * 100, 1) if total_cases > 0 else 0

            # Unique substances
            unique_drugs = df["suspected_drug"].nunique()

            # Fatal cases
            fatal_count = len(df[df["disposition"].isin(["CPR attempted", "DOA"])])
            round((fatal_count / total_cases) * 100, 1) if total_cases > 0 else 0

            # Narcan administered
            narcan_count = df["narcan_given"].sum() if "narcan_given" in df.columns else 0
            round((narcan_count / total_cases) * 100, 1) if total_cases > 0 else 0

            drug_stats = [
                {
                    "label": "Most Common",
                    "value": top_drug,
                    "description": f"{top_drug_count} cases • {top_drug_pct}%",
                },
                {
                    "label": "Unique Substances",
                    "value": str(unique_drugs),
                    "description": "Tracked compounds",
                },
            ]

    context = {
        "theme": theme,
        "suspected_drug_chart": suspected_drug_chart,
        "drug_stats": drug_stats,
    }

    return render(request, "dashboard/partials/od_suspected_drug_detail.html", context)


def od_living_situation_detail(request):
    """Serve detailed living situation analysis modal content."""
    theme = get_theme_from_request(request)

    # Build living situation chart
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(theme=theme, fields=["living_situation"])
    living_situation_chart = charts.get("living_situation", "")

    # Calculate living situation stats
    import pandas as pd

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values(
        "living_situation", "disposition", "referral_to_sud_agency"
    )
    data = list(qs)

    living_stats = []
    if data:
        df = pd.DataFrame.from_records(data)
        df = df[df["living_situation"].notna()]

        if not df.empty:
            total_cases = len(df)

            # Top situation
            top_living = df["living_situation"].value_counts().index[0] if len(df) > 0 else "N/A"
            top_living_count = (
                int(df["living_situation"].value_counts().iloc[0]) if len(df) > 0 else 0
            )
            top_living_pct = (
                round((top_living_count / total_cases) * 100, 1) if total_cases > 0 else 0
            )

            # Unique situations
            unique_situations = df["living_situation"].nunique()

            # Fatal cases
            fatal_count = len(df[df["disposition"].isin(["CPR attempted", "DOA"])])
            fatal_pct = round((fatal_count / total_cases) * 100, 1) if total_cases > 0 else 0

            # SUD referral success
            sud_success = df["referral_to_sud_agency"].sum()
            sud_success_pct = round((sud_success / total_cases) * 100, 1) if total_cases > 0 else 0

            living_stats = [
                {
                    "label": "Most Common",
                    "value": top_living,
                    "description": f"{top_living_count} cases • {top_living_pct}%",
                },
                {
                    "label": "Housing Types",
                    "value": str(unique_situations),
                    "description": "Distinct categories",
                },
                {
                    "label": "Fatal Rate",
                    "value": f"{fatal_pct}%",
                    "description": f"{fatal_count} of {total_cases} cases",
                },
                {
                    "label": "SUD Linkage",
                    "value": f"{sud_success_pct}%",
                    "description": f"{sud_success} treatment connections",
                },
            ]

    context = {
        "theme": theme,
        "living_situation_chart": living_situation_chart,
        "living_stats": living_stats,
    }

    return render(request, "dashboard/partials/od_living_situation_detail.html", context)


def od_engagement_location_detail(request):
    """Serve detailed engagement location analysis modal content."""
    theme = get_theme_from_request(request)

    # Build engagement location chart
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(theme=theme, fields=["engagement_location"])
    engagement_location_chart = charts.get("engagement_location", "")

    # Calculate engagement location stats
    import pandas as pd

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values("engagement_location", "disposition", "narcan_given")
    data = list(qs)

    location_stats = []
    if data:
        df = pd.DataFrame.from_records(data)
        df = df[df["engagement_location"].notna()]

        if not df.empty:
            total_cases = len(df)

            # Top location
            top_location = (
                df["engagement_location"].value_counts().index[0] if len(df) > 0 else "N/A"
            )
            top_location_count = (
                int(df["engagement_location"].value_counts().iloc[0]) if len(df) > 0 else 0
            )
            top_location_pct = (
                round((top_location_count / total_cases) * 100, 1) if total_cases > 0 else 0
            )

            # Unique locations
            unique_locations = df["engagement_location"].nunique()

            # Fatal cases
            fatal_count = len(df[df["disposition"].isin(["CPR attempted", "DOA"])])
            fatal_pct = round((fatal_count / total_cases) * 100, 1) if total_cases > 0 else 0

            # Narcan administered
            narcan_count = df["narcan_given"].sum() if "narcan_given" in df.columns else 0
            narcan_pct = round((narcan_count / total_cases) * 100, 1) if total_cases > 0 else 0

            location_stats = [
                {
                    "label": "Primary Location",
                    "value": top_location,
                    "description": f"{top_location_count} encounters • {top_location_pct}%",
                },
                {
                    "label": "Unique Venues",
                    "value": str(unique_locations),
                    "description": "Engagement sites",
                },
                {
                    "label": "Fatal Rate",
                    "value": f"{fatal_pct}%",
                    "description": f"{fatal_count} of {total_cases} cases",
                },
                {
                    "label": "Narcan Used",
                    "value": f"{narcan_pct}%",
                    "description": f"{narcan_count} administrations",
                },
            ]

    context = {
        "theme": theme,
        "engagement_location_chart": engagement_location_chart,
        "location_stats": location_stats,
    }

    return render(request, "dashboard/partials/od_engagement_location_detail.html", context)


def od_sud_referral_detail(request):
    """Serve detailed SUD referral analysis modal content."""
    theme = get_theme_from_request(request)

    # Build SUD referral chart
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(theme=theme, fields=["referral_to_sud_agency"])
    sud_referral_chart = charts.get("referral_to_sud_agency", "")

    # Calculate SUD referral stats
    import pandas as pd

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values(
        "referral_to_sud_agency", "disposition", "referral_source", "cpm_disposition"
    )
    data = list(qs)

    sud_stats = []
    if data:
        df = pd.DataFrame.from_records(data)

        if not df.empty:
            total_cases = len(df)

            # Successful referrals
            sud_success = df["referral_to_sud_agency"].sum()
            sud_success_pct = round((sud_success / total_cases) * 100, 1) if total_cases > 0 else 0

            # Failed referrals
            sud_failed = total_cases - sud_success - 22
            sud_failed_pct = round((sud_failed / total_cases) * 100, 1) if total_cases > 0 else 0

            sud_stats = [
                {
                    "label": "Success Rate",
                    "value": f"{sud_success_pct}%",
                    "description": f"{sud_success} of {total_cases} linked to treatment",
                },
                {
                    "label": "Not Referred",
                    "value": f"{sud_failed_pct}%",
                    "description": f"{sud_failed} cases without SUD linkage",
                },
            ]

    context = {
        "theme": theme,
        "sud_referral_chart": sud_referral_chart,
        "sud_stats": sud_stats,
    }

    return render(request, "dashboard/partials/od_sud_referral_detail.html", context)


def od_narcan_detail(request):
    """Serve detailed narcan administration analysis modal content."""
    theme = get_theme_from_request(request)

    # Build narcan chart
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(theme=theme, fields=["narcan_given"])
    narcan_chart = charts.get("narcan_given", "")

    # Calculate narcan stats
    import pandas as pd

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values("narcan_given", "disposition", "suspected_drug")
    data = list(qs)

    narcan_stats = []
    if data:
        df = pd.DataFrame.from_records(data)

        if not df.empty:
            total_cases = len(df)

            # Narcan administered
            narcan_given = df["narcan_given"].sum()
            narcan_pct = round((narcan_given / total_cases) * 100, 1) if total_cases > 0 else 0

            # Not administered
            narcan_not_given = total_cases - narcan_given
            narcan_not_pct = (
                round((narcan_not_given / total_cases) * 100, 1) if total_cases > 0 else 0
            )

            # Fatal cases that received narcan
            fatal_with_narcan = len(
                df[
                    df["disposition"].isin(["CPR attempted", "DOA"]) & (df["narcan_given"] == True)  # noqa: E712
                ]
            )

            # Fatal cases that did NOT receive narcan
            fatal_without_narcan = len(
                df[
                    df["disposition"].isin(["CPR attempted", "DOA"]) & (df["narcan_given"] == False)  # noqa: E712
                ]
            )

            narcan_stats = [
                {
                    "label": "Administered",
                    "value": f"{narcan_pct}%",
                    "description": f"{narcan_given} of {total_cases} cases",
                },
                {
                    "label": "Not Given",
                    "value": f"{narcan_not_pct}%",
                    "description": f"{narcan_not_given} cases without narcan",
                },
                {
                    "label": "Fatal w/ Narcan",
                    "value": str(fatal_with_narcan),
                    "description": "Fatal cases that received narcan",
                },
                {
                    "label": "Fatal w/o Narcan",
                    "value": str(fatal_without_narcan),
                    "description": "Fatal cases without narcan",
                },
            ]

    context = {
        "theme": theme,
        "narcan_chart": narcan_chart,
        "narcan_stats": narcan_stats,
    }

    return render(request, "dashboard/partials/od_narcan_detail.html", context)


def od_referral_delay_detail(request):
    """Serve detailed referral delay analysis modal content."""
    theme = get_theme_from_request(request)

    # Build delay_in_referral chart
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(theme=theme, fields=["delay_in_referral"])
    delay_chart = charts.get("delay_in_referral", "")

    # Calculate delay stats
    import pandas as pd

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values("delay_in_referral", "disposition", "od_date")
    data = list(qs)

    delay_stats = []
    if data:
        df = pd.DataFrame.from_records(data)

        if not df.empty and "delay_in_referral" in df.columns:
            total_cases = len(df)

            # Most common delay category
            delay_counts = df["delay_in_referral"].value_counts()
            most_common_delay = delay_counts.index[0] if len(delay_counts) > 0 else "N/A"
            most_common_count = delay_counts.iloc[0] if len(delay_counts) > 0 else 0

            # Unique delay categories
            unique_delays = df["delay_in_referral"].nunique()

            # Fatal cases
            fatal_cases = len(df[df["disposition"].isin(["CPR attempted", "DOA"])])

            # Average delay (if numeric data available - we'll calculate this differently)
            # Since delay_in_referral might be categorical, we'll show percentage with delays
            has_delay_count = len(df[df["delay_in_referral"].notna()])
            delay_pct = round((has_delay_count / total_cases) * 100, 1) if total_cases > 0 else 0

            delay_stats = [
                {
                    "label": "Most Common",
                    "value": str(most_common_delay),
                    "description": f"{most_common_count} cases in this category",
                },
                {
                    "label": "Delay Categories",
                    "value": str(unique_delays),
                    "description": "Different delay types tracked",
                },
                {
                    "label": "Cases w/ Delay Data",
                    "value": f"{delay_pct}%",
                    "description": f"{has_delay_count} of {total_cases} cases",
                },
                {
                    "label": "Fatal Cases",
                    "value": str(fatal_cases),
                    "description": "Total fatal outcomes",
                },
            ]

    context = {
        "theme": theme,
        "delay_chart": delay_chart,
        "delay_stats": delay_stats,
    }

    return render(request, "dashboard/partials/od_referral_delay_detail.html", context)


def od_cpm_notification_detail(request):
    """Serve detailed CPM notification analysis modal content."""
    theme = get_theme_from_request(request)

    # Build cpm_notification chart
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(theme=theme, fields=["cpm_notification"])
    notification_chart = charts.get("cpm_notification", "")

    # Calculate notification stats
    import pandas as pd

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values("cpm_notification", "disposition", "referral_source")
    data = list(qs)

    notification_stats = []
    if data:
        df = pd.DataFrame.from_records(data)

        if not df.empty and "cpm_notification" in df.columns:
            total_cases = len(df)

            # Most common notification method
            notification_counts = df["cpm_notification"].value_counts()
            most_common_notification = (
                notification_counts.index[0] if len(notification_counts) > 0 else "N/A"
            )
            most_common_count = notification_counts.iloc[0] if len(notification_counts) > 0 else 0

            # Unique notification methods
            unique_notifications = df["cpm_notification"].nunique()

            # Fatal cases
            fatal_cases = len(df[df["disposition"].isin(["CPR attempted", "DOA"])])

            # Cases with notification data
            has_notification_count = len(df[df["cpm_notification"].notna()])
            notification_pct = (
                round((has_notification_count / total_cases) * 100, 1) if total_cases > 0 else 0
            )

            notification_stats = [
                {
                    "label": "Primary Method",
                    "value": str(most_common_notification),
                    "description": f"{most_common_count} cases used this method",
                },
                {
                    "label": "Notification Types",
                    "value": str(unique_notifications),
                    "description": "Different methods tracked",
                },
                {
                    "label": "Notification Rate",
                    "value": f"{notification_pct}%",
                    "description": f"{has_notification_count} of {total_cases} cases",
                },
                {
                    "label": "Fatal Cases",
                    "value": str(fatal_cases),
                    "description": "Total fatal outcomes",
                },
            ]

    context = {
        "theme": theme,
        "notification_chart": notification_chart,
        "notification_stats": notification_stats,
    }

    return render(request, "dashboard/partials/od_cpm_notification_detail.html", context)


def od_scene_responders_detail(request):
    """Serve detailed on-scene responder analysis modal content."""
    theme = get_theme_from_request(request)

    # Build scene responders chart
    import pandas as pd
    import plotly.express as px
    from plotly.offline import plot

    from utils.chart_colors import CHART_COLORS_VIBRANT
    from utils.plotly import style_plotly_layout

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values(
        "number_of_nonems_onscene",
        "number_of_ems_onscene",
        "number_of_peers_onscene",
        "number_of_police_onscene",
        "disposition",
    )
    data = list(qs)

    responder_chart = ""
    responder_stats = []

    if data:
        df = pd.DataFrame.from_records(data)

        if not df.empty:
            # Average responders per scene (excluding scenes where count is 0)
            # Only average when responders were actually present
            avg_nonems = df[df["number_of_nonems_onscene"] > 0]["number_of_nonems_onscene"].mean()
            avg_ems = df[df["number_of_ems_onscene"] > 0]["number_of_ems_onscene"].mean()
            avg_peers = df[df["number_of_peers_onscene"] > 0]["number_of_peers_onscene"].mean()
            avg_police = df[df["number_of_police_onscene"] > 0]["number_of_police_onscene"].mean()

            # Handle NaN results (when all values are 0)
            avg_nonems = 0 if pd.isna(avg_nonems) else avg_nonems
            avg_ems = 0 if pd.isna(avg_ems) else avg_ems
            avg_peers = 0 if pd.isna(avg_peers) else avg_peers
            avg_police = 0 if pd.isna(avg_police) else avg_police

            # Total responders across all scenes
            total_nonems = df["number_of_nonems_onscene"].sum()
            total_ems = df["number_of_ems_onscene"].sum()
            total_peers = df["number_of_peers_onscene"].sum()
            total_police = df["number_of_police_onscene"].sum()

            # Scenes where each responder type was present
            scenes_with_nonems = len(df[df["number_of_nonems_onscene"] > 0])
            scenes_with_ems = len(df[df["number_of_ems_onscene"] > 0])
            scenes_with_peers = len(df[df["number_of_peers_onscene"] > 0])
            scenes_with_police = len(df[df["number_of_police_onscene"] > 0])

            # Build simple horizontal bar chart showing average responders by type
            chart_data = [
                {
                    "Responder Type": "EMS/Paramedics",
                    "Average per Scene": round(avg_ems, 2),
                    "Total Across All Scenes": int(total_ems),
                    "Scenes with Presence": scenes_with_ems,
                },
                {
                    "Responder Type": "Law Enforcement",
                    "Average per Scene": round(avg_police, 2),
                    "Total Across All Scenes": int(total_police),
                    "Scenes with Presence": scenes_with_police,
                },
                {
                    "Responder Type": "Peer Support",
                    "Average per Scene": round(avg_peers, 2),
                    "Total Across All Scenes": int(total_peers),
                    "Scenes with Presence": scenes_with_peers,
                },
                {
                    "Responder Type": "Non-EMS Personnel",
                    "Average per Scene": round(avg_nonems, 2),
                    "Total Across All Scenes": int(total_nonems),
                    "Scenes with Presence": scenes_with_nonems,
                },
            ]

            chart_df = pd.DataFrame(chart_data)

            fig = px.bar(
                chart_df,
                y="Responder Type",
                x="Average per Scene",
                orientation="h",
                color="Responder Type",
                color_discrete_sequence=CHART_COLORS_VIBRANT,
                custom_data=["Total Across All Scenes", "Scenes with Presence"],
            )

            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>"
                + "Average: %{x:.2f} when present<br>"
                + "Present at: %{customdata[1]} scenes<br>"
                + "Total: %{customdata[0]} responders<extra></extra>",
                texttemplate="%{x:.1f}",
                textposition="outside",
            )

            fig = style_plotly_layout(
                fig,
                theme=theme,
                height=400,
                x_title="Average Number of Responders per Overdose Scene",
                y_title=None,
            )

            fig.update_layout(showlegend=False)

            responder_chart = plot(
                fig,
                output_type="div",
                config={
                    "responsive": True,
                    "displaylogo": False,
                    "displayModeBar": "hover",
                    "modeBarButtonsToRemove": [
                        "zoom2d",
                        "pan2d",
                        "select2d",
                        "lasso2d",
                        "zoomIn2d",
                        "zoomOut2d",
                        "autoScale2d",
                    ],
                },
            )

            responder_stats = [
                {
                    "label": "Avg EMS/Scene",
                    "value": f"{avg_ems:.1f}",
                    "description": f"{total_ems:.0f} total EMS responders",
                },
                {
                    "label": "Avg Police/Scene",
                    "value": f"{avg_police:.1f}",
                    "description": f"{total_police:.0f} total officers",
                },
                {
                    "label": "Avg Peer Support",
                    "value": f"{avg_peers:.1f}",
                    "description": f"{total_peers:.0f} total peer responders",
                },
                {
                    "label": "Avg Non-EMS",
                    "value": f"{avg_nonems:.1f}",
                    "description": f"{total_nonems:.0f} total non-EMS personnel",
                },
            ]

    context = {
        "theme": theme,
        "responder_chart": responder_chart,
        "responder_stats": responder_stats,
    }

    return render(request, "dashboard/partials/od_scene_responders_detail.html", context)


def od_cpr_administered_detail(request):
    """Serve detailed CPR administration analysis modal content."""
    theme = get_theme_from_request(request)

    # Build CPR chart
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(theme=theme, fields=["cpr_administered"])
    cpr_chart = charts.get("cpr_administered", "")

    # Calculate CPR stats
    import pandas as pd

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values("cpr_administered", "disposition", "narcan_given")
    data = list(qs)

    cpr_stats = []
    if data:
        df = pd.DataFrame.from_records(data)

        if not df.empty and "cpr_administered" in df.columns:
            total_cases = len(df)

            # CPR administered count - combination of specific CPR types
            cpr_types = [
                "CPR - bystander only",
                "CPR - bystander & EMS",
                "CPR - EMS only",
                "CPR - LE & EMS only",
                "CPR - bystander, LE & EMS",
            ]
            cpr_given = len(df[df["cpr_administered"].isin(cpr_types)])
            cpr_pct = round((cpr_given / total_cases) * 100, 1) if total_cases > 0 else 0

            # No CPR cases
            no_cpr = len(df[df["cpr_administered"] == "No CPR"])
            no_cpr_pct = round((no_cpr / total_cases) * 100, 1) if total_cases > 0 else 0

            # Fatal cases
            fatal_cases = len(df[df["disposition"].isin(["CPR attempted", "DOA"])])

            # CPR + Narcan cases - check for CPR types AND narcan given
            cpr_with_narcan = len(
                df[df["cpr_administered"].isin(cpr_types) & (df["narcan_given"] == True)]  # noqa: E712
            )

            cpr_stats = [
                {
                    "label": "CPR Administered",
                    "value": f"{cpr_pct}%",
                    "description": f"{cpr_given} of {total_cases} cases",
                },
                {
                    "label": "No CPR",
                    "value": f"{no_cpr_pct}%",
                    "description": f"{no_cpr} cases without CPR",
                },
                {
                    "label": "Fatal Cases",
                    "value": str(fatal_cases),
                    "description": "Total DOA/CPR attempted",
                },
                {
                    "label": "CPR + Narcan",
                    "value": str(cpr_with_narcan),
                    "description": "Both interventions used",
                },
            ]

    context = {
        "theme": theme,
        "cpr_chart": cpr_chart,
        "cpr_stats": cpr_stats,
    }

    return render(request, "dashboard/partials/od_cpr_administered_detail.html", context)


def od_police_ita_detail(request):
    """Serve detailed Police ITA analysis modal content."""
    theme = get_theme_from_request(request)

    # Build Police ITA chart
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(theme=theme, fields=["police_ita"])
    ita_chart = charts.get("police_ita", "")

    # Calculate Police ITA stats
    import pandas as pd

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values("police_ita", "disposition", "living_situation")
    data = list(qs)

    ita_stats = []
    if data:
        df = pd.DataFrame.from_records(data)

        if not df.empty and "police_ita" in df.columns:
            total_cases = len(df)

            # ITA Issued count - specifically "ITA served"
            ita_count = len(df[df["police_ita"] == "ITA served"])
            ita_pct = round((ita_count / total_cases) * 100, 1) if total_cases > 0 else 0

            # No ITA count - specifically "No ITA"
            no_ita = len(df[df["police_ita"] == "No ITA"])
            no_ita_pct = round((no_ita / total_cases) * 100, 1) if total_cases > 0 else 0

            ita_stats = [
                {
                    "label": "ITA Issued",
                    "value": f"{ita_pct}%",
                    "description": f"{ita_count} of {total_cases} cases",
                },
                {
                    "label": "No ITA",
                    "value": f"{no_ita_pct}%",
                    "description": f"{no_ita} cases without ITA",
                },
            ]

    context = {
        "theme": theme,
        "ita_chart": ita_chart,
        "ita_stats": ita_stats,
    }

    return render(request, "dashboard/partials/od_police_ita_detail.html", context)


def od_disposition_detail(request):
    """Serve detailed disposition analysis modal content."""
    theme = get_theme_from_request(request)

    # Build disposition chart
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(theme=theme, fields=["disposition"])
    disposition_chart = charts.get("disposition", "")

    # Calculate disposition stats
    import pandas as pd

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values("disposition", "narcan_given", "cpr_administered")
    data = list(qs)

    disposition_stats = []
    if data:
        df = pd.DataFrame.from_records(data)

        if not df.empty and "disposition" in df.columns:
            total_cases = len(df)

            # Most common disposition
            disposition_counts = df["disposition"].value_counts()
            most_common = disposition_counts.index[0] if len(disposition_counts) > 0 else "N/A"
            most_common_count = disposition_counts.iloc[0] if len(disposition_counts) > 0 else 0

            # Fatal cases
            fatal_cases = len(df[df["disposition"].isin(["CPR attempted", "DOA"])])
            fatal_pct = round((fatal_cases / total_cases) * 100, 1) if total_cases > 0 else 0

            # Unique dispositions
            unique_dispositions = df["disposition"].nunique()

            # Transported cases (any transport-related disposition)
            transport_keywords = ["transport", "hospital", "ems"]
            transported = len(
                df[
                    df["disposition"]
                    .str.lower()
                    .str.contains("|".join(transport_keywords), na=False)
                ]
            )
            transport_pct = round((transported / total_cases) * 100, 1) if total_cases > 0 else 0

            disposition_stats = [
                {
                    "label": "Most Common",
                    "value": str(most_common),
                    "description": f"{most_common_count} cases",
                },
                {
                    "label": "Fatal Outcomes",
                    "value": f"{fatal_pct}%",
                    "description": f"{fatal_cases} DOA/CPR attempted",
                },
                {
                    "label": "Disposition Types",
                    "value": str(unique_dispositions),
                    "description": "Different outcomes tracked",
                },
                {
                    "label": "Transported",
                    "value": f"{transport_pct}%",
                    "description": f"{transported} to hospital/care",
                },
            ]

    context = {
        "theme": theme,
        "disposition_chart": disposition_chart,
        "disposition_stats": disposition_stats,
    }

    return render(request, "dashboard/partials/od_disposition_detail.html", context)


def od_transport_detail(request):
    """Serve detailed transport analysis modal content."""
    theme = get_theme_from_request(request)

    # Build transport charts
    from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts

    charts = build_odreferrals_field_charts(
        theme=theme, fields=["transport_to_location", "transported_by"]
    )
    transport_location_chart = charts.get("transport_to_location", "")
    transported_by_chart = charts.get("transported_by", "")

    # Calculate transport stats
    import pandas as pd

    from ..core.models import ODReferrals

    qs = ODReferrals.objects.all().values("transport_to_location", "transported_by", "disposition")
    data = list(qs)

    transport_stats = []
    if data:
        df = pd.DataFrame.from_records(data)

        if not df.empty:
            total_cases = len(df)

            # Cases with transport location data
            # Only count specific destinations as "transported"
            valid_destinations = {"hospital", "home", "jail", "shelter"}
            if "transport_to_location" in df.columns:
                # Normalize to lowercase and strip whitespace for comparison
                transport_locs = df["transport_to_location"].astype(str).str.lower().str.strip()
                has_location = len(df[transport_locs.isin(valid_destinations)])
            else:
                has_location = 0

            location_pct = round((has_location / total_cases) * 100, 1) if total_cases > 0 else 0

            # Most common transport location
            if "transport_to_location" in df.columns and df["transport_to_location"].notna().any():
                location_counts = df["transport_to_location"].value_counts()
                most_common_location = (
                    location_counts.index[0] if len(location_counts) > 0 else "N/A"
                )
                location_count = location_counts.iloc[0] if len(location_counts) > 0 else 0
            else:
                most_common_location = "N/A"
                location_count = 0

            # Most common transport method
            if "transported_by" in df.columns and df["transported_by"].notna().any():
                method_counts = df["transported_by"].value_counts()
                most_common_method = method_counts.index[0] if len(method_counts) > 0 else "N/A"
                method_count = method_counts.iloc[0] if len(method_counts) > 0 else 0
            else:
                most_common_method = "N/A"
                method_count = 0

            transport_stats = [
                {
                    "label": "Transport Rate",
                    "value": f"{location_pct}%",
                    "description": f"{has_location} of {total_cases} transported",
                },
                {
                    "label": "Top Destination",
                    "value": str(most_common_location),
                    "description": f"{location_count} cases",
                },
                {
                    "label": "Primary Method",
                    "value": str(most_common_method),
                    "description": f"{method_count} cases",
                },
            ]

    context = {
        "theme": theme,
        "transport_location_chart": transport_location_chart,
        "transported_by_chart": transported_by_chart,
        "transport_stats": transport_stats,
    }

    return render(request, "dashboard/partials/od_transport_detail.html", context)


def odreferrals_insights(request):
    context = _prepare_odreferrals_insights_context()
    updated_on = date(2025, 11, 30)
    context.update(
        {
            "page_header_updated_at": updated_on,
            "page_header_updated_at_iso": updated_on.isoformat(),
            "page_header_read_time": "7 min read",
        }
    )
    return render(request, "dashboard/odreferrals_insights.html", context)


# Encounters
def _load_encounters_dataframe() -> pd.DataFrame:
    fields = [
        "encounter_date",
        "pcp_agency",
        "encounter_type_cat1",
        "encounter_type_cat2",
        "encounter_type_cat3",
    ]
    try:
        values = list(Encounters.objects.all().values(*fields))
    except Exception:
        values = []
    df = _df_from_queryset(values, fields)
    if "encounter_date" in df.columns:
        df["encounter_date"] = pd.to_datetime(df["encounter_date"], errors="coerce")
    return df


def _ordered_encounter_fields(df: pd.DataFrame) -> list[str]:
    fields = ["encounters_counts_monthly", "encounters_counts_weekday"]
    for field in [
        "pcp_agency",
        "encounter_type_cat1",
        "encounter_type_cat2",
        "encounter_type_cat3",
    ]:
        if field in df.columns:
            fields.append(field)
    return fields


def _encounters_monthly_insights(dates: pd.Series) -> list[dict[str, object]]:
    monthly_counts = (
        dates.dt.to_period("M").value_counts().sort_index()
        if not dates.empty
        else pd.Series(dtype="int64")
    )
    if monthly_counts.empty:
        return []
    latest_period = monthly_counts.index[-1]
    latest_count = _to_int(monthly_counts.iloc[-1])
    latest_label = latest_period.to_timestamp().strftime("%b %Y")

    items: list[dict[str, object]] = [
        {"label": "Most recent month", "value": f"{latest_label} — {latest_count}"},
    ]

    if len(monthly_counts) >= 2:
        prev_count = _to_int(monthly_counts.iloc[-2])
        delta = latest_count - prev_count
        delta_pct = _pct(delta, prev_count) if prev_count else 0.0
        items.append({"label": "Month-over-month", "value": f"{delta:+d} ({delta_pct:+.1f}%)"})

    yoy_period = latest_period - 12
    yoy_value = monthly_counts.get(yoy_period)
    if yoy_value is not None and not pd.isna(yoy_value):
        yoy_count = _to_int(yoy_value)
        yoy_delta = latest_count - yoy_count
        yoy_pct = _pct(yoy_delta, yoy_count) if yoy_count else 0.0
        items.append({"label": "Year-over-year", "value": f"{yoy_delta:+d} ({yoy_pct:+.1f}%)"})

    trailing_12 = _to_int(monthly_counts.tail(12).sum())
    items.append({"label": "Trailing 12 months", "value": trailing_12})

    return items


def _encounters_weekday_insights(dates: pd.Series) -> list[dict[str, object]]:
    weekdays = dates.dt.day_name()
    if weekdays.empty:
        return []
    weekday_counts = weekdays.value_counts()
    total = _to_int(weekday_counts.sum())
    if total == 0:
        return []
    items: list[dict[str, object]] = []
    top_day = str(weekday_counts.idxmax())
    top_share = _pct(_to_int(weekday_counts.max()), total)
    items.append({"label": "Busiest day", "value": f"{top_day} ({top_share:.1f}%)"})
    weekend_total = _to_int(weekday_counts.get("Saturday", 0)) + _to_int(
        weekday_counts.get("Sunday", 0)
    )
    weekend_share = _pct(weekend_total, total)
    items.append({"label": "Weekend share", "value": f"{weekend_share:.1f}% of encounters"})
    return items


def _build_encounters_chart_insights(df_all: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    if df_all.empty:
        return {}

    insights: dict[str, list[dict[str, object]]] = {}

    if "encounter_date" in df_all.columns:
        dates = pd.to_datetime(df_all["encounter_date"], errors="coerce").dropna()
        if not dates.empty:
            monthly_items = _encounters_monthly_insights(dates)
            if monthly_items:
                insights["encounters_counts_monthly"] = monthly_items
            weekday_items = _encounters_weekday_insights(dates)
            if weekday_items:
                insights["encounters_counts_weekday"] = weekday_items

    for field, label in [
        ("pcp_agency", "PCP Agency"),
        ("encounter_type_cat1", "Encounter Type (Primary)"),
        ("encounter_type_cat2", "Encounter Type (Secondary)"),
        ("encounter_type_cat3", "Encounter Type (Tertiary)"),
    ]:
        tips = _insights_categorical(df_all, field, label)
        if tips:
            insights[field] = tips

    return insights


@login_required
def user_profile(request):
    user = request.user
    if request.method == "POST":
        form = ProfileForm(request.POST, request.FILES, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "Your profile has been updated.")
            return redirect("dashboard:user_profile")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = ProfileForm(instance=user)

    context = {
        "form": form,
        "profile_user": user,
        "theme": get_theme_from_request(request),
    }
    return render(request, "dashboard/profile.html", context)


def authentication(request):
    return render(request, "dashboard/authentication.html")


def _build_historical_hargrove_metrics(year_metrics: dict, str_q: str) -> list[dict[str, object]]:
    """Build metrics from historical JSON data."""
    q_data = year_metrics[str_q]
    metrics = q_data.get("metrics", {})
    narratives = q_data.get("narratives", [])

    # 1. Patient Demographics
    demo_rows = []
    if "individuals" in metrics:
        demo_rows.extend(metrics["individuals"])
    if "Health Insurance Type" in metrics:
        demo_rows.extend(metrics["Health Insurance Type"])
    if "zip_codes" in metrics:
        demo_rows.extend(metrics["zip_codes"])

    # 4. Outcomes
    outcome_rows = []
    if "services" in metrics:
        outcome_rows.extend(metrics["services"])
    if "objectives" in metrics:
        outcome_rows.extend(metrics["objectives"])

    return [
        {
            "title": "1. Patient Demographics",
            "type": "table",
            "columns": ["ID", "Metric", "Value", "Notes"],
            "rows": demo_rows if demo_rows else [],
        },
        {
            "title": "2. Substance Use Disorder (SUD)",
            "type": "table",
            "columns": ["ID", "Metric", "Value", "Notes"],
            "rows": [],
        },
        {
            "title": "3. Behavioral Health",
            "type": "table",
            "columns": ["ID", "Metric", "Value", "Notes"],
            "rows": [],
        },
        {
            "title": "4. Outcomes",
            "type": "table",
            "columns": ["ID", "Metric", "Value", "Notes"],
            "rows": outcome_rows if outcome_rows else [],
        },
        {
            "title": "5. Narrative",
            "type": "narrative",
            "questions_responses": narratives,
        },
    ]


def _build_2025_hargrove_metrics(year: int, q: int) -> list[dict[str, object]]:
    """Build dynamic metrics for 2025 from database."""
    from datetime import timedelta

    start_date = date(year, (q - 1) * 3 + 1, 1)
    end_date = date(year, 12, 31) if q == 4 else date(year, q * 3 + 1, 1) - timedelta(days=1)

    # 1. Patient Demographics
    served_patient_ids = (
        Encounters.objects.filter(encounter_date__range=(start_date, end_date))
        .values_list("patient_ID", flat=True)
        .distinct()
    )
    total_served = len(served_patient_ids)
    new_enrollments = Patients.objects.filter(created_date__range=(start_date, end_date)).count()
    served_patients = Patients.objects.filter(id__in=served_patient_ids)

    # Insurance
    insurance_counts = served_patients.values("insurance").annotate(count=Count("id"))
    insurance_rows = [
        {
            "id": "-",
            "metric": f"Insurance: {item['insurance'] or 'Unknown'}",
            "value": str(item["count"]),
            "notes": "",
        }
        for item in insurance_counts
    ]

    # Zip Codes
    zip_counts = served_patients.values("zip_code").annotate(count=Count("id"))
    zip_rows = [
        {
            "id": "-",
            "metric": f"Zip: {item['zip_code'] or 'Unknown'}",
            "value": str(item["count"]),
            "notes": "",
        }
        for item in zip_counts
    ]

    demo_rows = (
        [
            {
                "id": "-",
                "metric": "Total Patients Served",
                "value": str(total_served),
                "notes": "Patients with encounters in quarter",
            },
            {
                "id": "-",
                "metric": "New Enrollments",
                "value": str(new_enrollments),
                "notes": "Created in quarter",
            },
        ]
        + insurance_rows
        + zip_rows
    )

    # 2. SUD
    sud_patients = served_patients.filter(sud=True).count()
    sud_referrals = ODReferrals.objects.filter(
        od_date__range=(start_date, end_date), referral_to_sud_agency=True
    ).count()
    sud_rows = [
        {
            "id": "-",
            "metric": "SUD Flagged Patients",
            "value": str(sud_patients),
            "notes": "Subset of served patients",
        },
        {
            "id": "-",
            "metric": "Referrals to SUD Agency",
            "value": str(sud_referrals),
            "notes": "From OD Referrals",
        },
    ]

    # 3. Behavioral Health
    bh_patients = served_patients.filter(behavioral_health=True).count()
    bh_rows = [
        {
            "id": "-",
            "metric": "Behavioral Health Flagged",
            "value": str(bh_patients),
            "notes": "Subset of served patients",
        },
    ]

    # 4. Outcomes
    total_encounters = Encounters.objects.filter(
        encounter_date__range=(start_date, end_date)
    ).count()
    total_referrals = Referrals.objects.filter(date_received__range=(start_date, end_date)).count()
    outcome_rows = [
        {
            "id": "-",
            "metric": "Total Encounters",
            "value": str(total_encounters),
            "notes": "",
        },
        {"id": "-", "metric": "Referrals Made", "value": str(total_referrals), "notes": ""},
    ]

    return [
        {
            "title": "1. Patient Demographics",
            "type": "table",
            "columns": ["ID", "Metric", "Value", "Notes"],
            "rows": demo_rows,
        },
        {
            "title": "2. Substance Use Disorder (SUD)",
            "type": "table",
            "columns": ["ID", "Metric", "Value", "Notes"],
            "rows": sud_rows,
        },
        {
            "title": "3. Behavioral Health",
            "type": "table",
            "columns": ["ID", "Metric", "Value", "Notes"],
            "rows": bh_rows,
        },
        {
            "title": "4. Outcomes",
            "type": "table",
            "columns": ["ID", "Metric", "Value", "Notes"],
            "rows": outcome_rows,
        },
        {
            "title": "5. Narrative",
            "type": "narrative",
            "questions_responses": [
                {
                    "question": "Narrative",
                    "response": "Data aggregation for the current year is automated. Narrative inputs are not yet available.",
                }
            ],
        },
    ]


def _build_hargrove_accordions() -> list[dict[str, object]]:
    """Build accordion data for Hargrove Grant reporting."""
    # Load historical data
    historical_data = {}
    try:
        with HARGROVE_METRICS_PATH.open(encoding="utf-8") as fp:
            historical_data = json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    years_data = []
    # Years to display: 2025 down to 2019
    target_years = range(2025, 2018, -1)

    for year in target_years:
        str_year = str(year)
        quarters = []

        # Determine if we have data for this year
        year_metrics = historical_data.get(str_year)

        # Generate 4 quarters for each year
        for q in range(1, 5):
            str_q = str(q)
            label = f"Q{q}"
            date_range = ""
            if q == 1:
                date_range = f"Jan - Mar {year}"
            elif q == 2:
                date_range = f"Apr - Jun {year}"
            elif q == 3:
                date_range = f"Jul - Sep {year}"
            elif q == 4:
                date_range = f"Oct - Dec {year}"

            sections = []

            if year_metrics and str_q in year_metrics:
                sections = _build_historical_hargrove_metrics(year_metrics, str_q)
            elif year == 2025:
                sections = _build_2025_hargrove_metrics(year, q)
            else:
                # Placeholder for missing historical years (2019-2021)
                sections = [
                    {
                        "title": "Data Archive",
                        "type": "table",
                        "columns": ["ID", "Metric", "Value", "Notes"],
                        "rows": [
                            {
                                "id": "-",
                                "metric": "Status",
                                "value": "Archived",
                                "notes": "Data for this period is archived.",
                            }
                        ],
                    }
                ]

            quarters.append({"label": label, "date_range": date_range, "sections": sections})

        years_data.append({"year": year, "is_current": year == 2025, "quarters": quarters})

    return years_data


def hargrove_grant(request):
    years_data = _build_hargrove_accordions()
    updated_on = date(2025, 11, 30)
    context = {
        "years_data": years_data,
        "page_header_updated_at": updated_on,
        "page_header_updated_at_iso": updated_on.isoformat(),
        "page_header_read_time": "5 min read",
    }
    return render(request, "dashboard/hargrove_grant.html", context)


@login_required
def age_chart_variations_demo(request):
    """Demo view to compare all 5 enhanced age chart variations."""
    theme = get_theme_from_request(request)
    variations = build_all_age_chart_variations(theme)

    context = {
        "theme": theme,
        "variations": [
            {
                "key": "option_a_annotated",
                "title": "Option A: Annotated Bar Chart",
                "description": "Enhanced with benchmark lines, target zones, and contextual insights",
                "html": variations["option_a_annotated"],
            },
            {
                "key": "option_b_pyramid",
                "title": "Option B: Population Pyramid",
                "description": "Gender comparison with mirrored horizontal bars",
                "html": variations["option_b_pyramid"],
            },
            {
                "key": "option_c_small_multiples",
                "title": "Option C: Small Multiples (Temporal)",
                "description": "Side-by-side comparison across time periods with trend indicators",
                "html": variations["option_c_small_multiples"],
            },
            {
                "key": "option_d_sankey",
                "title": "Option D: Sankey Flow Diagram",
                "description": "Visual flow from age groups to service pathways",
                "html": variations["option_d_sankey"],
            },
            {
                "key": "option_e_combination",
                "title": "Option E: Combination Chart",
                "description": "Bars for counts with cumulative percentage line overlay",
                "html": variations["option_e_combination"],
            },
        ],
    }
    return render(request, "dashboard/age_chart_variations_demo.html", context)
