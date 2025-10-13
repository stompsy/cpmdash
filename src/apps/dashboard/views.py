import json
from contextlib import suppress
from typing import TypedDict

import pandas as pd
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render
from django.utils import timezone

from apps.accounts.forms import ProfileForm
from utils.theme import get_theme_from_request

from ..charts.od_utils import get_quarterly_patient_counts
from ..charts.odreferrals.odreferrals_field_charts import build_odreferrals_field_charts
from ..charts.overdose.od_all_cases_scatter import (
    build_chart_all_cases_scatter,  # noqa: F401 - re-export for tests monkeypatch
)
from ..charts.overdose.od_density_heatmap import (  # noqa: F401 - re-export for tests monkeypatch
    build_chart_od_density_heatmap,
)
from ..charts.overdose.od_hist_monthly import build_chart_od_hist_monthly
from ..charts.overdose.od_hourly_breakdown import (  # noqa: F401 - re-export for tests monkeypatch
    build_chart_day_of_week_totals,
    build_chart_od_hourly_breakdown,
)
from ..charts.overdose.od_repeats_scatter import (  # noqa: F401 - re-export for tests monkeypatch
    build_chart_repeats_scatter,
)
from ..charts.overdose.od_shift_scenarios import (  # noqa: F401 - re-export for tests monkeypatch
    build_chart_cost_benefit_analysis,
    build_chart_shift_scenarios,
    calculate_coverage_scenarios,
)
from ..charts.patients.patient_field_charts import build_patients_field_charts
from ..charts.referral.referrals_field_charts import build_referrals_field_charts
from ..core.models import Encounters, ODReferrals, Patients, Referrals


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
    return render(request, "dashboard/overview.html")


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


def _build_repeat_overdose_stats() -> list[dict[str, int | float]]:
    records = list(
        ODReferrals.objects.exclude(od_date__isnull=True).values("od_date", "patient_id")
    )
    if not records:
        return []

    df = pd.DataFrame(records)
    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    df = df.dropna(subset=["od_date", "patient_id"])
    if df.empty:
        return []

    df["year"] = df["od_date"].dt.year.astype(int)
    df["patient_id"] = df["patient_id"].astype(int)

    stats: list[dict[str, int | float]] = []
    for year, group in df.groupby("year"):
        total = int(group.shape[0])
        per_patient = group.groupby("patient_id").size()
        repeat_mask = per_patient > 1
        repeat_overdoses = int(per_patient[repeat_mask].sum())
        repeat_patients = int(repeat_mask.sum())
        percent_repeat = round((repeat_overdoses / total) * 100, 1) if total else 0.0
        stats.append(
            {
                "year": _to_int(year),
                "total_overdoses": total,
                "repeat_overdoses": repeat_overdoses,
                "repeat_patients": repeat_patients,
                "percent_repeat": percent_repeat,
            }
        )

    stats.sort(key=lambda item: item["year"])
    return stats


def _insights_sud(df_all: pd.DataFrame) -> list[dict[str, object]] | None:
    if "sud" not in df_all.columns:
        return None
    s2 = df_all["sud"].map({True: "Yes", False: "No"}).fillna("Unknown")
    vc = s2.value_counts()
    total = int(vc.sum())
    yes = int(vc.get("Yes", 0))
    no = int(vc.get("No", 0))
    unk = int(vc.get("Unknown", 0))
    known = yes + no
    prevalence_known = _pct(yes, known) if known else 0.0
    items: list[dict[str, object]] = [
        {"label": "SUD Yes", "value": f"{yes} ({_pct(yes, total)}%)"},
        {"label": "SUD No", "value": f"{no} ({_pct(no, total)}%)"},
        {"label": "Unknown", "value": f"{unk} ({_pct(unk, total)}%)"},
        {"label": "Prevalence (known)", "value": f"{prevalence_known}%"},
    ]
    return items


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


def _build_patients_chart_insights(
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
                    f"Referrals accompany PORT work in { _pct(int((od_linked_df['referrals'] > 0).sum()), int(od_linked_df.shape[0])) }% of these cases.",
                ],
                "examples": _patient_examples(od_linked_df),
            }
        )

    return sections


def _prepare_patients_insights_context() -> dict[str, object]:
    datasets = _load_patient_touchpoint_datasets()
    patients_df = datasets.get("patients", pd.DataFrame())
    if patients_df.empty:
        return {
            "data_is_empty": True,
            "summary_cards": [],
            "top_patients": [],
            "insight_sections": [],
        }

    enriched = _enrich_patient_touchpoints(datasets)
    if enriched.empty or not enriched["has_touchpoint"].any():
        return {
            "data_is_empty": True,
            "summary_cards": [],
            "top_patients": [],
            "insight_sections": [],
        }

    summary_cards = _build_patient_summary_cards(enriched)
    top_patients = _build_patient_top_patients(enriched)
    insight_sections = _build_patient_insight_sections(enriched)

    return {
        "data_is_empty": False,
        "summary_cards": summary_cards,
        "top_patients": top_patients,
        "insight_sections": insight_sections,
    }


def patients_insights(request):
    context = _prepare_patients_insights_context()
    return render(request, "dashboard/patients_insights.html", context)


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


REFERRALS_RATIONALE_MAP: dict[str, str] = {
    "age": (
        "Age informs outreach intensity and linkage speed. Younger referrals may need school or family coordination; older adults often need fall risk and medication review."
    ),
    "insurance": (
        "Coverage reveals access barriers and guides which benefits coordinations to deploy (e.g., Medicaid enrollment assistance, charity care)."
    ),
    "referral_agency": (
        "Knowing who refers helps strengthen partnerships and feedback loops. It highlights where education or outreach could boost appropriate referrals."
    ),
    "zipcode": (
        "ZIP code patterns surface geographic access challenges and help route mobile services and targeted outreach."
    ),
    "referral_closed_reason": (
        "Closed reasons explain throughput and friction (unable to contact, declined, no longer in area). Tracking them improves workflows and follow-up scripts."
    ),
    "sex": (
        "Sex distribution helps scan for imbalances and tailor messaging or scheduling to avoid access gaps."
    ),
    "encounter_type_cat1": (
        "Encounter types show what’s driving referrals initially; use this to align resources like behavioral health or primary care slots."
    ),
    "encounter_type_cat2": (
        "Secondary encounter categorization reveals co-occurring needs that impact care plans and handoffs."
    ),
    "encounter_type_cat3": (
        "Tertiary categories catch nuances that inform documentation, reporting, and quality improvement."
    ),
    "referrals_counts_quarterly": (
        "Quarterly volume benchmarks outreach effectiveness and partner engagement, informing staffing and grant reporting."
    ),
}


def patients(request):
    theme = get_theme_from_request(request)
    charts = build_patients_field_charts(theme=theme)

    # Build DataFrame for insights
    try:
        base_fields = [
            "age",
            "insurance",
            "pcp_agency",
            "race",
            "sex",
            "sud",
            "zip_code",
            "marital_status",
            "veteran_status",
        ]
        df_all = pd.DataFrame(list(Patients.objects.all().values(*base_fields)))
    except Exception:
        df_all = pd.DataFrame()

    age_insights = _compute_age_insights(df_all)
    chart_insights = _build_patients_chart_insights(df_all, age_insights)

    # Rationale text per field
    rationale_map: dict[str, str] = {
        "age": (
            "Age guides risk stratification, eligibility for services, and care planning. Older adults often need "
            "falls prevention, medication reconciliation, and chronic disease support, while younger cohorts may "
            "benefit more from behavioral health linkage and injury prevention."
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
            "ZIP codes help identify geographic clusters, travel barriers, and SDOH needs. They inform mobile "
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
            "SUD screening is central to harm reduction and linkage to treatment. Tracking prevalence and screening "
            "completeness helps target naloxone, MAT referrals, and proactive outreach."
        ),
        "patient_counts_quarterly": (
            "Quarterly volume shows throughput, staffing needs, and seasonality. It supports grant reporting, "
            "capacity planning, and demonstrates program growth or stabilization."
        ),
        "sex_age_boxplot": (
            "Age distribution by sex can reveal cohort differences relevant to scheduling, education, and outreach. "
            "Monitoring gaps helps tailor services equitably."
        ),
        "race_age_boxplot": (
            "Age distribution by race helps detect potential disparities and informs culturally responsive, "
            "community-informed outreach strategies."
        ),
    }

    label_overrides = {
        "pcp_agency": "Primary Care Agency",
        "zip_code": "ZIP Code",
        "patient_counts_quarterly": "Patients by Quarter",
        "sex_age_boxplot": "Age Distribution by Sex",
        "race_age_boxplot": "Age Distribution by Race",
    }

    charts_list = []

    # Reorder so donut pairs render side-by-side in the 2-col grid
    def _move_after(lst: list[str], item: str, after_item: str) -> None:
        if item in lst and after_item in lst:
            lst.remove(item)
            try:
                idx = lst.index(after_item)
            except ValueError:
                return
            lst.insert(idx + 1, item)

    ordered_fields = list(charts.keys())
    # Ensure Insurance and SUD are adjacent (SUD immediately after Insurance)
    _move_after(ordered_fields, "sud", "insurance")
    # Ensure Marital Status and Veteran Status are adjacent (Veteran immediately after Marital)
    _move_after(ordered_fields, "veteran_status", "marital_status")

    for field in ordered_fields:
        chart = charts[field]
        charts_list.append(
            {
                "field": field,
                "label": label_overrides.get(field, field.replace("_", " ").title()),
                "chart": chart,
                "insights": chart_insights.get(field),
                "rationale": rationale_map.get(field),
            }
        )

    context = {
        "charts": charts,
        "charts_list": charts_list,
        "theme": theme,
        "age_insights": age_insights,
    }
    return render(request, "dashboard/patients.html", context)


# Referrals
def referrals(request):
    theme = get_theme_from_request(request)
    charts = build_referrals_field_charts(theme=theme)

    label_overrides = {
        "zipcode": "ZIP Code",
        "referral_agency": "Referral Agency",
        "referrals_counts_quarterly": "Referrals by Quarter",
    }

    # Order donuts: Insurance then Closed Reason
    ordered_fields = list(charts.keys())

    def _move_after(lst: list[str], item: str, after_item: str) -> None:
        if item in lst and after_item in lst:
            lst.remove(item)
            try:
                idx = lst.index(after_item)
            except ValueError:
                return
            lst.insert(idx + 1, item)

    _move_after(ordered_fields, "referral_closed_reason", "insurance")

    # Build DataFrame to compute insights for referrals
    try:
        from ..core.models import Referrals  # local import to avoid circular import in tests

        df_all = pd.DataFrame(
            list(
                Referrals.objects.all().values(
                    "age",
                    "sex",
                    "referral_agency",
                    "encounter_type_cat1",
                    "encounter_type_cat2",
                    "encounter_type_cat3",
                    "referral_closed_reason",
                    "zipcode",
                    "insurance",
                    "date_received",
                )
            )
        )
    except Exception:
        df_all = pd.DataFrame()

    chart_insights = _build_referrals_chart_insights(df_all)

    charts_list = []
    for field in ordered_fields:
        charts_list.append(
            {
                "field": field,
                "label": label_overrides.get(field, field.replace("_", " ").title()),
                "chart": charts[field],
                "insights": chart_insights.get(field),
                "rationale": REFERRALS_RATIONALE_MAP.get(field),
            }
        )

    context = {"charts": charts, "charts_list": charts_list, "theme": theme}
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
    return render(request, "dashboard/odreferrals_shift_coverage.html", context)


def odreferrals_repeat_overdoses(request):
    theme = get_theme_from_request(request)

    fig_repeats_scatter = _chart_html(build_chart_repeats_scatter(theme=theme))
    repeat_stats = _build_repeat_overdose_stats()

    context = {
        "repeat_stats_by_year": repeat_stats,
        "fig_repeats_scatter": fig_repeats_scatter,
        "has_repeat_data": bool(repeat_stats),
    }
    return render(request, "dashboard/odreferrals_repeat_overdoses.html", context)


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


def referrals_insights(request):
    context = _prepare_referrals_insights_context()
    return render(request, "dashboard/referrals_insights.html", context)


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


def encounters_insights(request):
    context = _prepare_encounters_insights_context()
    return render(request, "dashboard/encounters_insights.html", context)


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
        ("referral_agency", "Referral Agency"),
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
        "referral_agency",
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
    "referral_agency": "Referral agency",
    "suspected_drug": "Suspected substance",
    "cpm_disposition": "CPM disposition",
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
    "referral_agency": (
        "Top referring agencies indicate where collaboration is strong and where additional training could boost reporting."
    ),
    "suspected_drug": (
        "Substance mix guides harm reduction supplies and treatment referrals tailored to current trends."
    ),
    "cpm_disposition": (
        "Disposition outcomes demonstrate how CPM teams close loops and prioritize follow-up."
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


OD_REFERRALS_COL_SPAN: dict[str, int] = {
    "odreferrals_counts_monthly": 2,
    "odreferrals_counts_weekday": 2,
    "narcan_given": 1,
    "referral_to_sud_agency": 1,
}


def odreferrals(request):
    theme = get_theme_from_request(request)
    charts = build_odreferrals_field_charts(theme=theme)

    fields = [
        "od_date",
        "referral_source",
        "referral_agency",
        "suspected_drug",
        "cpm_disposition",
        "living_situation",
        "engagement_location",
        "narcan_given",
        "referral_to_sud_agency",
    ]
    try:
        df_all = pd.DataFrame(list(ODReferrals.objects.all().values(*fields)))
    except Exception:
        df_all = pd.DataFrame(columns=fields)

    chart_insights = _build_odreferrals_chart_insights(df_all)

    charts_list: list[dict[str, object]] = []
    for field, chart in charts.items():
        charts_list.append(
            {
                "field": field,
                "label": OD_REFERRALS_LABEL_MAP.get(field, field.replace("_", " ").title()),
                "chart": chart,
                "insights": chart_insights.get(field),
                "rationale": OD_REFERRALS_RATIONALE_MAP.get(field),
                "col_span": OD_REFERRALS_COL_SPAN.get(field, 2),
            }
        )

    context = {
        "charts": charts,
        "charts_list": charts_list,
        "theme": theme,
    }
    return render(request, "dashboard/odreferrals.html", context)


def odreferrals_insights(request):
    context = _prepare_odreferrals_insights_context()
    return render(request, "dashboard/odreferrals_insights.html", context)


# Removed overdoses_by_case view per request


def odreferrals_monthly(request):
    theme = get_theme_from_request(request)

    # Get chart
    od_monthly = build_chart_od_hist_monthly(theme=theme)

    odreferrals = ODReferrals.objects.all()
    od_records = list(
        odreferrals.values(
            "disposition",
            "od_date",
            "patient_id",
            "narcan_given",
            "suspected_drug",
            "living_situation",
            "cpm_disposition",
            "referral_to_sud_agency",
            "referral_source",
        )
    )
    df = pd.DataFrame.from_records(od_records)
    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["od_date"])

    # Total overdoses
    total_overdoses = len(df)

    # Set month for trend calculations
    df["month"] = df["od_date"].dt.to_period("M")

    # Fatality Rate
    fatal_overdoses = len(df[df["disposition"].isin(["CPR attempted", "DOA"])])

    # Repeat overdoses (patients with more than one overdose)
    repeat_counts = df["patient_id"].value_counts()
    repeat_patients = len(repeat_counts[repeat_counts > 1])
    repeat_overdoses = repeat_counts[repeat_counts > 1].sum()
    percent_repeat = (
        round((repeat_overdoses / total_overdoses) * 100, 1) if total_overdoses > 0 else 0
    )

    # Calculate referral success rate based on 'referral_to_sud_agency', excluding 'Other' cases.
    # Filter out cases where referral_source is 'Other'
    referral_df = df[df["referral_source"] != "Other"]

    if len(referral_df) > 0:
        successful_referrals = referral_df[
            "referral_to_sud_agency"
        ].sum()  # This works for boolean True values
        total_eligible_referrals = len(referral_df)
        referral_success_rate = round((successful_referrals / total_eligible_referrals) * 100, 1)
    else:
        referral_success_rate = 0.0

    # Calculate density stats for time regions
    def calculate_density_stats():
        early_morning_mask = df["od_date"].dt.hour < 8  # 00:00-07:59
        working_hours_mask = (
            df["od_date"].dt.hour.between(8, 15)  # 08:00-15:59
            & df["od_date"].dt.weekday.isin([0, 1, 2, 3, 4])  # Mon–Fri
        )
        weekend_daytime_mask = (
            df["od_date"].dt.hour.between(8, 15)  # 08:00-15:59
            & df["od_date"].dt.weekday.isin([5, 6])  # Sat–Sun
        )
        early_evening_mask = (
            df["od_date"].dt.hour.between(16, 18)  # 16:00-18:59
            & df["od_date"].dt.weekday.isin([0, 1, 2, 3, 4])  # Mon–Fri
        )
        weekend_early_evening_mask = (
            df["od_date"].dt.hour.between(16, 18)  # 16:00-18:59
            & df["od_date"].dt.weekday.isin([5, 6])  # Sat–Sun
        )
        late_evening_mask = df["od_date"].dt.hour >= 19  # 19:00-23:59

        # Calculate counts and percentages
        early_morning_count = early_morning_mask.sum()
        working_hours_count = working_hours_mask.sum()
        weekend_daytime_count = weekend_daytime_mask.sum()
        early_evening_count = early_evening_mask.sum()
        weekend_early_evening_count = weekend_early_evening_mask.sum()
        late_evening_count = late_evening_mask.sum()

        def percent(x):
            return round((x / total_overdoses) * 100, 1) if total_overdoses else 0

        return {
            "early_morning": {
                "count": int(early_morning_count),
                "pct": percent(early_morning_count),
            },
            "working_hours": {
                "count": int(working_hours_count),
                "pct": percent(working_hours_count),
            },
            "weekend_daytime": {
                "count": int(weekend_daytime_count),
                "pct": percent(weekend_daytime_count),
            },
            "early_evening": {
                "count": int(early_evening_count),
                "pct": percent(early_evening_count),
            },
            "weekend_early_evening": {
                "count": int(weekend_early_evening_count),
                "pct": percent(weekend_early_evening_count),
            },
            "late_evening": {"count": int(late_evening_count), "pct": percent(late_evening_count)},
        }

    density_stats = calculate_density_stats()

    context = {
        "od_monthly": od_monthly,
        "total_overdoses": total_overdoses,
        "fatal_overdoses": fatal_overdoses,
        "repeat_overdoses": repeat_overdoses,
        "repeat_patients": repeat_patients,
        "percent_repeat": percent_repeat,
        "referral_success_rate": referral_success_rate,
        "density_stats": density_stats,
        "theme": theme,
    }

    return render(request, "dashboard/monthly.html", context)


# Encounters
def encounters(request):
    return render(request, "dashboard/encounters.html")


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
