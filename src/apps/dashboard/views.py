import pandas as pd
from django.shortcuts import render

from utils.theme import get_theme_from_request

from ..charts.overdose.od_all_cases_scatter import build_chart_all_cases_scatter
from ..charts.overdose.od_hist_monthly import build_chart_od_hist_monthly
from ..core.models import ODReferrals


def overview(request):
    return render(request, "dashboard/overview.html")


# Patients
def patients(request):
    return render(request, "dashboard/patients.html")


# Referrals
def referrals(request):
    return render(request, "dashboard/referrals.html")


# OD Referrals
def odreferrals(request):
    return render(request, "dashboard/odreferrals.html")


def overdoses_by_case(request):
    theme = get_theme_from_request(request)
    all_cases_scatter_plot = build_chart_all_cases_scatter(theme=theme)
    context = {
        "all_cases_scatter_plot": all_cases_scatter_plot,
    }
    return render(request, "dashboard/overdoses_by_case.html", context)


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
