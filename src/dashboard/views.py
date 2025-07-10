from django.shortcuts import render
import json

from .models import *
from django.db.models import Count

from .utils.theme import get_theme_from_request
from .charts.od_utils import *

# General statistics
from .charts.overdose.od_age_race import *
from .charts.overdose.od_age_sex import *
from .charts.overdose.od_stack_livingsituation import *
from .charts.overdose.od_stack_insurance import *

# Trends over time
from .charts.overdose.od_hist_monthly import *
from .charts.overdose.od_density_heatmap import *

# Geographic
from .charts.overdose.od_map import *

# Operation metrics
from .charts.overdose.od_repeats_scatter import *
from .charts.overdose.od_bar_workhours import *
from .charts.overdose.od_hourly_breakdown import *
from .charts.overdose.od_shift_scenarios import *

# CPM specific metrics
from .charts.overdose.od_referral_delay import *
from .charts.overdose.od_cpm_notification import *
from .charts.overdose.od_cpm_disposition import *

# Substance specific
from .charts.overdose.od_sus_drug import *

# Emergency response metrics
from .charts.overdose.od_cpr_admin import *
from .charts.overdose.od_call_disposition import *

# Referrals
from .charts.referral.od_agency_treemap import build_chart_od_agency_treemap


def dashboard(request):

    title = "Dashboard"
    description = "This is a Dashboard page"

    context = {
        "title": title,
        "description": description,
        # "odreferrals_count": odreferrals_count,
    }

    return render(request, "dashboard/index.html", context)


def patients(request):
    patients = Patients.objects.all()
    title = "Patients"
    description = "This is a Patients page"

    context = {
        "title": title,
        "description": description,
        "patients": patients,
    }

    return render(request, "dashboard/patients.html", context)


def encounters(request):
    encounters = Encounters.objects.all()
    title = "Encounters"
    description = "This is a Encounters page"

    context = {
        "title": title,
        "description": description,
        "encounters": encounters,
    }

    return render(request, "dashboard/encounters.html", context)


def referrals(request):
    theme = get_theme_from_request(request)
    title = "Referrals"
    description = "This is a Referrals page"

    fig_agency_treemap      = build_chart_od_agency_treemap(theme=theme)

    return render(
        request,
        "dashboard/referrals.html",
        {
            "title":                        title,
            "description":                  description,
            "fig_agency_treemap":           fig_agency_treemap,
            "theme": theme,
        },
    )


def odreferrals(request):
    title = "PORT Referrals"
    description = "Key metrics for analyzing Post Overdose Response Team data"

    return render(
        request,
        "dashboard/odreferrals.html",
        {
            "title":                        title,
            "description":                  description,
        },
    )


def odreferrals_trends(request):
    theme = get_theme_from_request(request)
    title = "PORT Referrals"
    description = "Key metrics - Trends"

    # Trends over time
    fig_od_monthly                  = build_chart_od_hist_monthly(theme=theme)
    fig_density_map, density_stats  = build_chart_od_density_heatmap(theme=theme)
    fig_od_work_hours               = build_chart_od_work_hours(theme=theme)
    fig_repeats_scatter             = build_chart_repeats_scatter(theme=theme)

    # Total overdoses (all time)
    total_overdoses = ODReferrals.objects.count()

    # Fatal overdoses (all time)
    fatal_overdoses = ODReferrals.objects.filter(
        disposition__in=["CPR attempted", "DOA"]
    ).count()

    # All-time repeat overdose statistics for header cards
    all_time_repeats = (
        ODReferrals.objects.exclude(patient_id__isnull=True)
        .values("patient_id")
        .annotate(num=Count("patient_id"))
        .filter(num__gt=1)
    )

    # Count of repeat patients (all time)
    repeat_patients = all_time_repeats.count()

    # Sum of repeat overdoses excluding each patient's first (all time)
    repeat_overdoses = sum([r["num"] - 1 for r in all_time_repeats])

    # Percentage of repeats (all time)
    percent_repeat = round((repeat_overdoses / total_overdoses) * 100, 1) if total_overdoses > 0 else 0
    
    # Referral success rate (all time) - handle both boolean True and string representations
    successful_referrals = ODReferrals.objects.filter(
        referral_to_sud_agency=True
    ).count()
    
    # Also count string representations that might exist in SQLite
    string_true_referrals = ODReferrals.objects.extra(
        where=["referral_to_sud_agency = 'true' OR referral_to_sud_agency = 'True' OR referral_to_sud_agency = '1'"]
    ).count()
    
    # Total successful referrals
    total_successful = successful_referrals + string_true_referrals
    referral_success_rate = round((total_successful / total_overdoses) * 100, 1) if total_overdoses > 0 else 0
    
    # List of years to compare
    years_to_compare = [2024, 2025]
    repeat_stats_by_year = []

    for year in years_to_compare:
        qs_year = ODReferrals.objects.filter(od_date__year=year).exclude(patient_id__isnull=True)

        total = qs_year.count()

        grouped = (
            qs_year
            .values("patient_id")
            .annotate(num=Count("patient_id"))
            .filter(num__gt=1)
        )

        repeat_overdoses_year = sum([r["num"] - 1 for r in grouped])
        repeat_patients_year = grouped.count()
        percent_repeat_year = round((repeat_overdoses_year / total) * 100, 1) if total > 0 else 0

        repeat_stats_by_year.append({
            "year": year,
            "total": total,
            "repeat_overdoses": repeat_overdoses_year,
            "repeat_patients": repeat_patients_year,
            "percent_repeat": percent_repeat_year,
        })

    return render(
        request,
        "dashboard/trends.html",
        {
            "title": title,
            "description": description,
            "od_monthly": fig_od_monthly,
            "fig_density_map": fig_density_map,
            "density_stats": density_stats,
            "od_work_hours": fig_od_work_hours,
            "fig_repeats_scatter": fig_repeats_scatter,
            "fatal_overdoses": fatal_overdoses,
            "repeat_overdoses": repeat_overdoses,
            "repeat_patients": repeat_patients,
            "percent_repeat": percent_repeat,
            "referral_success_rate": referral_success_rate,
            "total_overdoses": total_overdoses,
            "repeat_stats_by_year": repeat_stats_by_year,
            "theme": theme,
        },
    )


def odreferrals_geographic(request):
    
    theme = get_theme_from_request(request)
    title = "PORT Referrals"
    description = "Key metrics - Geographic"
    
    # Geographic
    fig_od_map              = build_chart_od_map(theme=theme)
    
    return render(
        request,
        "dashboard/geographic.html",
        {
            "title":                        title,
            "description":                  description,
            
            "fig_od_map":                   fig_od_map,
            
            "theme": theme,
        },
    )


def odreferrals_substances(request):
    
    theme = get_theme_from_request(request)
    title = "PORT Referrals"
    description = "Key metrics - Substances"
    
    # Substance specific
    fig_od_sus_drug         = build_chart_sus_drug(theme=theme)
    
    return render(
        request,
        "dashboard/substances.html",
        {
            "title":                        title,
            "description":                  description,
            
            "fig_od_sus_drug":              fig_od_sus_drug,
            
            "theme": theme,
        },
    )


def odreferrals_operations(request):
    
    theme = get_theme_from_request(request)
    title = "PORT Referrals"
    description = "Key metrics - Operation Metrics"

    # Charts for operations page
    fig_density_map, density_stats  = build_chart_od_density_heatmap(theme=theme)
    fig_density_map_interactive_static = build_chart_od_density_heatmap_interactive_static(theme=theme)
    
    # ML-based predictions for different volumes
    fig_density_map_ml_1000 = build_chart_od_density_heatmap_ml_predicted(theme=theme, target_volume=1000, method="ml_ensemble")
    fig_density_map_ml_5000 = build_chart_od_density_heatmap_ml_predicted(theme=theme, target_volume=5000, method="ml_ensemble")
    
    # Build new detailed analytics
    fig_hourly_breakdown = build_chart_od_hourly_breakdown(theme=theme)
    fig_day_of_week_totals = build_chart_day_of_week_totals(theme=theme)
    fig_shift_scenarios = build_chart_shift_scenarios(theme=theme)
    fig_cost_benefit_analysis = build_chart_cost_benefit_analysis(theme=theme)
    
    # Get shift scenario data for summary cards
    scenarios_data = calculate_coverage_scenarios()

    # Calculate statistics for both old and new time boundaries
    def calculate_region_stats(use_old_boundaries=False):
        import pandas as pd
        from .models import ODReferrals
        
        # Get all overdose data
        odreferrals = ODReferrals.objects.all()
        df = pd.DataFrame.from_records(odreferrals.values("od_date"))
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
        
        percent = lambda x: round((x / total_count) * 100, 1) if total_count else 0
        
        return {
            "early_morning": {"count": int(early_morning_count), "pct": percent(early_morning_count)},
            "working_hours": {"count": int(working_hours_count), "pct": percent(working_hours_count)},
            "weekend_daytime": {"count": int(weekend_daytime_count), "pct": percent(weekend_daytime_count)},
            "early_evening": {"count": int(early_evening_count), "pct": percent(early_evening_count)},
            "weekend_early_evening": {"count": int(weekend_early_evening_count), "pct": percent(weekend_early_evening_count)},
            "late_evening": {"count": int(late_evening_count), "pct": percent(late_evening_count)},
        }
    
    # Calculate stats for both boundary definitions
    stats_old = calculate_region_stats(use_old_boundaries=True)
    stats_new = calculate_region_stats(use_old_boundaries=False)

    return render(
        request,
        "dashboard/operations.html",
        {
            "title": title,
            "description": description,
            "fig_density_map": fig_density_map,
            "fig_density_map_interactive_static": fig_density_map_interactive_static,
            "fig_density_map_ml_1000": fig_density_map_ml_1000,
            "fig_density_map_ml_5000": fig_density_map_ml_5000,
            "fig_hourly_breakdown": fig_hourly_breakdown,
            "fig_day_of_week_totals": fig_day_of_week_totals,
            "fig_shift_scenarios": fig_shift_scenarios,
            "fig_cost_benefit_analysis": fig_cost_benefit_analysis,
            "scenarios_data": json.dumps(scenarios_data) if scenarios_data else "{}",
            "theme": theme,
            # Summary statistics - use old boundaries for current coverage
            "current_coverage": stats_old["working_hours"]["pct"],
            "missed_opportunities": round(100 - stats_old["working_hours"]["pct"], 1),
            "proposed_coverage": round(stats_new["working_hours"]["pct"] + stats_new["early_evening"]["pct"], 1),
        },
    )


def odreferrals_response(request):
    
    theme = get_theme_from_request(request)
    title = "PORT Referrals"
    description = "Key metrics - Emergency Response Metrics"

    # Emergency response metrics
    fig_cpr_admin                   = build_chart_cpr_admin(theme=theme)
    fig_call_disposition            = build_chart_call_disposition(theme=theme)

    return render(
        request,
        "dashboard/response.html",
        {
            "title":                        title,
            "description":                  description,
            
            "fig_cpr_admin":                fig_cpr_admin,
            "fig_call_disposition":         fig_call_disposition,
            
            "theme": theme,
        },
    )


def odreferrals_cpm(request):
    
    theme = get_theme_from_request(request)
    title = "PORT Referrals"
    description = "Key metrics - CPM Specific Metrics"

    # CPM specific metrics
    fig_referral_delay      = build_chart_referral_delay(theme=theme)
    fig_cpm_notification    = build_chart_cpm_notification(theme=theme)
    fig_cpm_disposition     = build_chart_cpm_disposition(theme=theme)
    
    return render(
        request,
        "dashboard/cpm.html",
        {
            "title":                        title,
            "description":                  description,
            
            "od_referral_delay":            fig_referral_delay,
            "fig_cpm_notification":         fig_cpm_notification,
            "fig_cpm_disposition":          fig_cpm_disposition,
            
            "theme": theme,
        },
    )


def odreferrals_socioeconomic(request):

    theme = get_theme_from_request(request)
    title = "PORT Referrals"
    description = "Key metrics - Social & Economic Factors"
    # od_counts = get_odreferral_counts()

    # General statistics
    fig_od_age_race         = build_chart_od_age_race(theme=theme)
    fig_od_age_sex          = build_chart_od_age_sex(theme=theme)

    # Socioeconomic factors
    fig_od_living_sit       = build_chart_od_stack_livingsituation(theme=theme)
    fig_od_insurance        = build_chart_od_stack_insurance(theme=theme)

    return render(
        request,
        "dashboard/socioeconomic.html",
        {
            "title":                        title,
            "description":                  description,            
            
            "od_age_race":                  fig_od_age_race,
            "od_age_sex":                   fig_od_age_sex,
            "od_living_sit":                fig_od_living_sit,
            "od_insurance":                 fig_od_insurance,
            
            "theme": theme,
        },
    )


# HTMX Chart Update Views
from django.http import HttpResponse

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

