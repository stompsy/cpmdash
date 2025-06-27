from django.shortcuts import render
from django.utils.timezone import now
from datetime import datetime

from .models import *
from django.db.models import Count

from .utils.theme import get_theme_from_request
from .charts.od_utils import *
from collections import defaultdict

# General statistics
from .charts.overdose.od_age_race import *
from .charts.overdose.od_age_sex import *
from .charts.overdose.od_stack_livingsituation import *
from .charts.overdose.od_stack_insurance import *

# Trends over time
from .charts.overdose.od_hist_monthly import *
from .charts.overdose.od_density_heatmap import *
from .charts.overdose.od_bar_workhours import *
from .charts.overdose.od_hist_hourly import *
from .charts.overdose.od_repeats_scatter import *

# Geographic
from .charts.overdose.od_map import *

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

    fig_agency_treemap      = build_chart_od_agency_treemap(theme="dark")

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
    fig_od_monthly                  = build_chart_od_hist_monthly(theme="dark")
    fig_density_map, density_stats  = build_chart_od_density_heatmap(theme="dark")
    fig_od_work_hours               = build_chart_od_work_hours(theme=theme)
    fig_od_hist_hourly              = build_chart_od_hist_hourly(theme=theme)
    fig_repeats_scatter             = build_chart_repeats_scatter(theme="dark")

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
    
    # Referral success rate (all time) - based on referral_to_sud_agency field
    successful_referrals = ODReferrals.objects.filter(referral_to_sud_agency=True).count()
    referral_success_rate = round((successful_referrals / total_overdoses) * 100, 1) if total_overdoses > 0 else 0
    
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
            "chart_od_hist_hourly": fig_od_hist_hourly,
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
    fig_od_map              = build_chart_od_map(theme="dark")
    
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
    fig_od_sus_drug         = build_chart_sus_drug(theme="dark")
    
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


def odreferrals_response(request):
    
    theme = get_theme_from_request(request)
    title = "PORT Referrals"
    description = "Key metrics - Emergency Response Metrics"

    # Emergency response metrics
    fig_cpr_admin           = build_chart_cpr_admin(theme="dark")
    fig_call_disposition    = build_chart_call_disposition(theme="dark")
    
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
    fig_referral_delay      = build_chart_referral_delay(theme="dark")
    fig_cpm_notification    = build_chart_cpm_notification(theme="dark")
    fig_cpm_disposition     = build_chart_cpm_disposition(theme="dark")
    
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
    fig_od_age_race         = build_chart_od_age_race(theme="dark")
    fig_od_age_sex          = build_chart_od_age_sex(theme="dark")

    # Socioeconomic factors
    fig_od_living_sit       = build_chart_od_stack_livingsituation(theme="dark")
    fig_od_insurance        = build_chart_od_stack_insurance(theme="dark")

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
