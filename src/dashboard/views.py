from django.shortcuts import render

# import pandas as pd
# from plotly.offline import plot
# from .utils.theme import get_theme_from_request

from dashboard.models import *

# from .charts.od_age_sex import build_chart_od_age_sex
# from .charts.od_age_race import build_chart_od_age_race
# from .charts.od_pie_fatal_nonfatal import build_chart_od_fatal_nonfatal
# from .charts.od_hist_monthly import build_chart_od_hist_monthly
# from .charts.od_density_heatmap import build_chart_od_density_heatmap
# from .charts.od_bar_workhours import build_chart_od_work_hours
# from .charts.od_hist_hourly import build_chart_od_hist_hourly
# from .charts.od_line_hourly import build_chart_od_line_hourly
# from .charts.od_map import build_chart_od_map
# from .charts.od_stack_livingsituation import build_chart_od_stack_livingsituation
# from .charts.od_stack_insurance import build_chart_od_stack_insurance
# from .charts.od_fatality_charts import get_fatality_charts
# from .charts.od_repeats_scatter import build_chart_repeats_scatter
# from .charts.od_referral_delay import build_chart_referral_delay

# odreferrals_count = ODReferrals.objects.count()
# odreferrals_count_2024 = ODReferrals.objects.filter(od_date__year=2024).count()
# odreferrals_count_2025 = ODReferrals.objects.filter(od_date__year=2025).count()

# od_rate = (
#     odreferrals_count_2024 / odreferrals_count * 100
# ) if odreferrals_count else 0

# od_fatality_rate = (
#     ODReferrals.objects.filter(
#         od_date__year=2024, disposition__icontains="fatal"
#     ).count()
#     / odreferrals_count_2024
#     * 100
# ) if odreferrals_count_2024 else 0

# # Calculate the OD fatality rate for 2025
# od_fatality_rate_2025 = (
#     ODReferrals.objects.filter(
#         od_date__year=2025, disposition__icontains="fatal"
#     ).count()
#     / odreferrals_count_2025
#     * 100
# ) if odreferrals_count_2025 else 0

# # Calculate the OD fatality rate for 2024
# od_fatality_rate_2024 = (
#     ODReferrals.objects.filter(
#         od_date__year=2024, disposition__icontains="fatal"
#     ).count()
#     / odreferrals_count_2024
#     * 100
# ) if odreferrals_count_2024 else 0


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
    # patients = Patients.objects.all()
    title = "Patients"
    description = "This is a Patients page"

    context = {
        "title": title,
        "description": description,
        # "patients": patients,
    }

    return render(request, "dashboard/patients.html", context)


def encounters(request):
    # encounters = Encounters.objects.all()
    title = "Encounters"
    description = "This is a Encounters page"

    context = {
        "title": title,
        "description": description,
        # "encounters": encounters,
    }

    return render(request, "dashboard/encounters.html", context)


def referrals(request):
    # referrals = Referrals.objects.all()
    title = "Referrals"
    description = "This is a Referrals page"

    context = {
        "title": title,
        "description": description,
        # "referrals": referrals,
    }

    return render(request, "dashboard/referrals.html", context)


def odreferrals(request):
    # theme = get_theme_from_request(request)
    title = "PORT Referrals"
    description = "Key metrics for analyzing Post Overdose Response Team data"

#     # Get OD records and create DataFrame
#     odreferrals = ODReferrals.objects.all()
#     df = pd.DataFrame.from_records(
#         odreferrals.values(
#             "patient_age",
#             "patient_sex",
#             "patient_race",
#             "disposition",
#             "od_date",
#         )
#     )

#     fig_od_age_sex = build_chart_od_age_sex(df, theme=theme)
#     chart_od_age_sex = plot(
#         fig_od_age_sex, output_type="div", config=fig_od_age_sex._config
#     )

#     fig_od_age_race = build_chart_od_age_race(df, theme=theme)
#     chart_od_age_race = plot(
#         fig_od_age_race, output_type="div", config=fig_od_age_race._config
#     )

#     fig_od_fatal_nfatal = build_chart_od_fatal_nonfatal(df, theme=theme)
#     chart_od_fatal_nfatal = plot(
#         fig_od_fatal_nfatal, output_type="div", config=fig_od_fatal_nfatal._config
#     )

#     fig_od_monthly = build_chart_od_hist_monthly(df, theme=theme)
#     chart_od_per_month = plot(
#         fig_od_monthly, output_type="div", config=fig_od_monthly._config
#     )

#     fig_od_density_heatmap = build_chart_od_density_heatmap(df, theme=theme)
#     chart_od_density_heatmap = plot(
#         fig_od_density_heatmap, output_type="div", config=fig_od_density_heatmap._config
#     )

#     fig_od_working_hours = build_chart_od_work_hours(df, theme=theme)
#     chart_od_working_hours = plot(
#         fig_od_working_hours, output_type="div", config=fig_od_working_hours._config
#     )

#     fig_od_most_ods_hourly = build_chart_od_hour_most_ods(df, theme=theme)
#     chart_od_most_ods_hourly = plot(
#         fig_od_most_ods_hourly, output_type="div", config=fig_od_most_ods_hourly._config
#     )

#     fig_od_hist_hourly = build_chart_od_hist_hourly(df, theme=theme)
#     chart_od_hist_hourly = plot(
#         fig_od_hist_hourly, output_type="div", config=fig_od_hist_hourly._config
#     )

#     fig_od_line_hourly = build_chart_od_line_hourly(df, theme=theme)
#     chart_od_line_hourly = plot(
#         fig_od_line_hourly, output_type="div", config=fig_od_line_hourly._config
#     )

#     fig_od_map = build_chart_od_map(df, theme=theme)
#     chart_od_map = plot(fig_od_map, output_type="div", config=fig_od_map._config)

#     fig_od_stack_insurance = build_chart_od_stack_insurance(df, theme=theme)
#     chart_od_stack_insurance = plot(
#         fig_od_stack_insurance,
#         output_type="div",
#         config=fig_od_stack_insurance._config,
#     )

#     fig_od_stack_livingsituation = build_chart_od_stack_livingsituation(df, theme=theme)
#     chart_od_stack_livingsituation = plot(
#         fig_od_stack_livingsituation,
#         output_type="div",
#         config=fig_od_stack_livingsituation._config,
#     )

#     fig_dict = get_fatality_charts(theme)
#     charts = {
#         name: plot(fig, output_type="div", config=fig._config)
#         for name, fig in fig_dict.items()
#     }

#     fig_od_repeats_scatter = build_chart_repeats_scatter(df, theme=theme)
#     chart_od_repeats_scatter = plot(
#         fig_od_repeats_scatter, output_type="div", config=fig_od_repeats_scatter._config
#     )

#     fig_referral_delay = build_chart_referral_delay(theme)

    return render(
        request,
        "dashboard/odreferrals.html",
        {
            "title": title,
            "description": description,
            # "odreferrals": odreferrals,
            # "odreferrals_count": odreferrals_count,
            # "odreferrals_count_2024": odreferrals_count_2024,
            # "odreferrals_count_2025": odreferrals_count_2025,
            # "od_rate": od_rate,
            # "od_fatality_rate": od_fatality_rate,
            # "chart_od_age_sex": chart_od_age_sex,
            # "chart_od_age_race": chart_od_age_race,
            # "chart_od_fatal_nfatal": chart_od_fatal_nfatal,
            # "chart_od_per_month": chart_od_per_month,
            # "chart_od_density_heatmap": chart_od_density_heatmap,
            # "chart_od_working_hours": chart_od_working_hours,
            # "chart_od_most_ods_hourly": chart_od_most_ods_hourly,
            # "chart_od_hist_hourly": chart_od_hist_hourly,
            # "chart_od_line_hourly": chart_od_line_hourly,
            # "chart_od_map": chart_od_map,
            # "chart_od_stack_livingsituation": chart_od_stack_livingsituation,
            # "chart_od_stack_insurance": chart_od_stack_insurance,
            # "chart_od_repeats_scatter": chart_od_repeats_scatter,
            # "chart_od_referral_delay": fig_referral_delay,
            # "charts": charts,
            # "theme": theme,
        },
    )