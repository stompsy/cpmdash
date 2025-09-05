import json

import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render

from utils.theme import get_theme_from_request

from ..charts.overdose.od_density_heatmap import build_chart_od_density_heatmap
from ..charts.overdose.od_hourly_breakdown import (
    build_chart_day_of_week_totals,
    build_chart_od_hourly_breakdown,
)
from ..charts.overdose.od_repeats_scatter import build_chart_repeats_scatter
from ..charts.overdose.od_shift_scenarios import (
    build_chart_cost_benefit_analysis,
    build_chart_shift_scenarios,
    calculate_coverage_scenarios,
)
from ..core.models import ODReferrals


def cases(request):
    title = "PORT Referrals"
    description = "Case Studies - OP Shielding Hope"
    context = {
        "title": title,
        "description": description,
    }
    return render(request, "cases/opshieldinghope.html", context=context)


def opshield(request):
    title = "PORT Referrals"
    description = "Case Studies - OP Shielding Hope"
    context = {
        "title": title,
        "description": description,
    }
    return render(request, "cases/opshieldinghope.html", context=context)


def shiftcoverage(request):
    theme = get_theme_from_request(request)
    title = "PORT Referrals"
    description = "Case Studies - Shift Coverage Analysis"

    # Charts for operations page
    fig_density_map = build_chart_od_density_heatmap(theme=theme)

    # Build new detailed analytics
    fig_hourly_breakdown = build_chart_od_hourly_breakdown(theme=theme)
    fig_day_of_week_totals = build_chart_day_of_week_totals(theme=theme)
    fig_shift_scenarios = build_chart_shift_scenarios(theme=theme)
    fig_cost_benefit_analysis = build_chart_cost_benefit_analysis(theme=theme)

    # Get shift scenario data for summary cards
    scenarios_data = calculate_coverage_scenarios()

    # Calculate statistics for both old and new time boundaries
    def calculate_region_stats(use_old_boundaries=False):
        # Get all overdose data
        odreferrals = ODReferrals.objects.all()
        df = pd.DataFrame.from_records(odreferrals.values("od_date"))

        # Gracefully handle empty dataset (no overdose records yet)
        if df.empty or "od_date" not in df.columns:
            return {
                "early_morning": {"count": 0, "pct": 0},
                "working_hours": {"count": 0, "pct": 0},
                "weekend_daytime": {"count": 0, "pct": 0},
                "early_evening": {"count": 0, "pct": 0},
                "weekend_early_evening": {"count": 0, "pct": 0},
                "late_evening": {"count": 0, "pct": 0},
            }

        df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
        df = df.dropna(subset=["od_date"])

        total_count = len(df)

        if use_old_boundaries:
            # Old time boundaries (08:00-16:00 working hours)
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
        else:
            # New time boundaries (09:00-17:00 working hours) - same as heatmap
            early_morning_mask = df["od_date"].dt.hour < 9  # 00:00-08:59
            working_hours_mask = (
                df["od_date"].dt.hour.between(9, 16)  # 09:00-16:59
                & df["od_date"].dt.weekday.isin([0, 1, 2, 3, 4])  # Mon–Fri
            )
            weekend_daytime_mask = (
                df["od_date"].dt.hour.between(9, 16)  # 09:00-16:59
                & df["od_date"].dt.weekday.isin([5, 6])  # Sat–Sun
            )
            early_evening_mask = (
                df["od_date"].dt.hour.between(17, 18)  # 17:00-18:59
                & df["od_date"].dt.weekday.isin([0, 1, 2, 3, 4])  # Mon–Fri
            )
            weekend_early_evening_mask = (
                df["od_date"].dt.hour.between(17, 18)  # 17:00-18:59
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
            return round((x / total_count) * 100, 1) if total_count else 0

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

    # Calculate stats for both boundary definitions
    stats_old = calculate_region_stats(use_old_boundaries=True)
    stats_new = calculate_region_stats(use_old_boundaries=False)

    return render(
        request,
        "cases/shift_coverage.html",
        {
            "title": title,
            "description": description,
            "fig_density_map": fig_density_map,
            "fig_hourly_breakdown": fig_hourly_breakdown,
            "fig_day_of_week_totals": fig_day_of_week_totals,
            "fig_shift_scenarios": fig_shift_scenarios,
            "fig_cost_benefit_analysis": fig_cost_benefit_analysis,
            "scenarios_data": json.dumps(scenarios_data) if scenarios_data else "{}",
            "current_coverage": stats_old["working_hours"]["pct"],
            "missed_opportunities": round(100 - stats_old["working_hours"]["pct"], 1),
            "proposed_coverage": round(
                stats_new["working_hours"]["pct"] + stats_new["early_evening"]["pct"], 1
            ),
            "theme": theme,
        },
    )


def repeatods(request):
    theme = get_theme_from_request(request)

    # Get the repeat overdoses chart
    fig_repeats_scatter = build_chart_repeats_scatter(theme=theme)

    # Calculate repeat overdose statistics
    odreferrals = ODReferrals.objects.all()
    df = pd.DataFrame.from_records(odreferrals.values("disposition", "od_date", "patient_id"))
    df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")
    df = df.dropna(subset=["od_date"])

    # Calculate year-over-year statistics
    repeat_stats_by_year = []

    for year in sorted(df["od_date"].dt.year.unique()):
        year_df = df[df["od_date"].dt.year == year]

        # Count repeat patients (patients with more than one overdose in this year)
        year_repeat_counts = year_df["patient_id"].value_counts()
        repeat_patients = len(year_repeat_counts[year_repeat_counts > 1])
        repeat_overdoses = year_repeat_counts[year_repeat_counts > 1].sum()
        total_overdoses_year = len(year_df)

        percent_repeat = (
            round((repeat_overdoses / total_overdoses_year) * 100, 1)
            if total_overdoses_year > 0
            else 0
        )

        repeat_stats_by_year.append(
            {
                "year": int(year),
                "repeat_overdoses": int(repeat_overdoses),
                "repeat_patients": int(repeat_patients),
                "percent_repeat": percent_repeat,
            }
        )

    context = {
        "fig_repeats_scatter": fig_repeats_scatter,
        "repeat_stats_by_year": repeat_stats_by_year,
        "theme": theme,
    }

    return render(request, "cases/repeatods.html", context)


def costsavings(request):
    """Render the Cost Savings Analysis page"""
    from ..charts.od_utils import get_cost_savings_metrics

    title = "Cost Savings Analysis"
    description = "Financial impact of Community Paramedic interventions"

    # Get dynamic cost savings metrics
    cost_metrics = get_cost_savings_metrics()

    # Calculate rounded total savings to nearest $500,000 for grant justification
    total_savings = cost_metrics["total_savings"]
    rounded_savings = round(total_savings / 500000) * 500000

    context = {
        "title": title,
        "description": description,
        "cost_metrics": cost_metrics,
        "rounded_total_savings": rounded_savings,
    }
    return render(request, "cases/costsavings.html", context)


# HTMX Chart Update Views


def htmx_heatmap_chart(request):
    """Return just the heatmap chart HTML for HTMX updates"""
    theme = get_theme_from_request(request)
    fig_density_map, _ = build_chart_od_density_heatmap(theme=theme)
    return HttpResponse(fig_density_map)


def htmx_hourly_breakdown_chart(request):
    """Return just the hourly breakdown chart HTML for HTMX updates"""
    theme = get_theme_from_request(request)
    fig_hourly_breakdown = build_chart_od_hourly_breakdown(theme=theme)
    return HttpResponse(fig_hourly_breakdown)


def htmx_shift_scenarios_chart(request):
    """Return just the shift scenarios chart HTML for HTMX updates"""
    theme = get_theme_from_request(request)
    fig_shift_scenarios = build_chart_shift_scenarios(theme=theme)
    return HttpResponse(fig_shift_scenarios)


def htmx_cost_benefit_chart(request):
    """Return just the cost benefit analysis chart HTML for HTMX updates"""
    theme = get_theme_from_request(request)
    fig_cost_benefit_analysis = build_chart_cost_benefit_analysis(theme=theme)
    return HttpResponse(fig_cost_benefit_analysis)
