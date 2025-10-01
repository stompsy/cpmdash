import pandas as pd
from django.shortcuts import render

from utils.theme import get_theme_from_request

from ..charts.od_utils import get_quarterly_patient_counts
from ..charts.overdose.od_all_cases_scatter import (
    build_chart_all_cases_scatter,  # noqa: F401 - re-export for tests monkeypatch
)
from ..charts.overdose.od_hist_monthly import build_chart_od_hist_monthly
from ..charts.patients.patient_field_charts import build_patients_field_charts
from ..charts.referral.referrals_field_charts import build_referrals_field_charts
from ..core.models import ODReferrals, Patients


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


# OD Referrals
def odreferrals(request):
    return render(request, "dashboard/odreferrals.html")


# Removed overdoses_by_case view per request


def odreferrals_monthly(request):
    theme = get_theme_from_request(request)

    # Get chart
    od_monthly = build_chart_od_hist_monthly(theme=theme)

    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(
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


def user_profile(request):
    return render(request, "dashboard/profile.html")


def authentication(request):
    return render(request, "dashboard/authentication.html")
