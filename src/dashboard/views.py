from django.shortcuts import render
from .utils.theme import get_theme_from_request

from .models import *

from .charts.od_utils import *
from .charts.od_age_sex import *
from .charts.od_age_race import *
from .charts.od_pie_fatal_nonfatal import *
from .charts.od_hist_monthly import *
from .charts.od_density_heatmap import *
from .charts.od_bar_workhours import *
from .charts.od_hist_hourly import *
from .charts.od_line_hourly import *
from .charts.od_map import *
from .charts.od_stack_livingsituation import *
from .charts.od_stack_insurance import *
from .charts.od_fatality_charts import *
from .charts.od_repeats_scatter import *
from .charts.od_referral_delay import build_chart_referral_delay


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
    referrals = Referrals.objects.all()
    title = "Referrals"
    description = "This is a Referrals page"

    context = {
        "title": title,
        "description": description,
        "referrals": referrals,
    }

    return render(request, "dashboard/referrals.html", context)


def odreferrals(request):
    
    theme = get_theme_from_request(request)
    title = "PORT Referrals"
    description = "Key metrics for analyzing Post Overdose Response Team data"
    
    od_counts = get_odreferral_counts()
    copa_population = 20_000
    fatal_dispositions = ["cpr attempted", "doa"]
    years = [2024, 2025]
    
    od_stats = { y: get_od_metrics(y, copa_population) for y in years }

    od_fatality_rate_2025 = get_od_fatality_rate_year(
        year=2025,
        fatal_dispositions=fatal_dispositions,
        copa_population=copa_population
    )
    
    od_fatality_rate_2024 = get_od_fatality_rate_year(
        year=2024,
        fatal_dispositions=fatal_dispositions,
        copa_population=copa_population
    )
        
    
    fig_od_age_sex = build_chart_od_age_sex(theme)
    fig_od_age_race = build_chart_od_age_race(theme)
    fig_od_work_hours = build_chart_od_work_hours(theme)
    fig_od_hist_hourly = build_chart_od_hist_hourly(theme)
    fig_referral_delay = build_chart_referral_delay(theme)

    return render(
        request,
        "dashboard/odreferrals.html",
        {
            "title":                        title,
            "description":                  description,
            # "chart_od_fatal_nfatal": chart_od_fatal_nfatal,
            # "chart_od_per_month": chart_od_per_month,
            # "chart_od_density_heatmap": chart_od_density_heatmap,
            # "chart_od_line_hourly": chart_od_line_hourly,
            # "chart_od_map": chart_od_map,
            # "chart_od_stack_livingsituation": chart_od_stack_livingsituation,
            # "chart_od_stack_insurance": chart_od_stack_insurance,
            # "chart_od_repeats_scatter": chart_od_repeats_scatter,
            
            'total_odreferrals':            od_counts['total'],
            'odreferrals_by_year':          od_counts['by_year'],
            "od_stats":                     od_stats,
            "od_fatality_rate_2025":        od_fatality_rate_2025,
            "od_fatality_rate_2024":        od_fatality_rate_2024,
            
            "od_age_sex":                   fig_od_age_sex,
            "od_age_race":                  fig_od_age_race,
            "od_work_hours":                fig_od_work_hours,
            "chart_od_hist_hourly":         fig_od_hist_hourly,
            "od_referral_delay":            fig_referral_delay,
            "theme": theme,
        },
    )
