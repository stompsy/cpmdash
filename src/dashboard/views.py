from django.shortcuts import render
from .utils.theme import get_theme_from_request

from .models import *

from .charts.od_utils import *

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
    fig_od_monthly          = build_chart_od_hist_monthly(theme="dark")
    fig_density_map         = build_chart_od_density_heatmap(theme="dark")
    fig_od_work_hours       = build_chart_od_work_hours(theme=theme)
    fig_od_hist_hourly      = build_chart_od_hist_hourly(theme=theme)
    fig_repeats_scatter     = build_chart_repeats_scatter(theme=theme)

    return render(
        request,
        "dashboard/trends.html",
        {
            "title":                        title,
            "description":                  description,

            "od_monthly":                   fig_od_monthly,
            "fig_density_map":              fig_density_map,
            "od_work_hours":                fig_od_work_hours,
            "chart_od_hist_hourly":         fig_od_hist_hourly,
            "fig_repeats_scatter":          fig_repeats_scatter,
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
