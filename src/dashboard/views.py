from django.shortcuts import render

from .models import *
from .charts.od_utils import *
from .utils.theme import get_theme_from_request

# General statistics
from .charts.overdose.od_age_race import *
from .charts.overdose.od_age_sex import *
from .charts.overdose.od_stack_livingsituation import *
from .charts.overdose.od_stack_insurance import *

# Trends over time
from .charts.overdose.od_hist_monthly import *

# Geographic
from .charts.overdose.od_map import *

# Operation metrics
from .charts.overdose.od_repeats_scatter import *
from .charts.overdose.od_bar_workhours import *

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


def overview(request):
    return render(request, "dashboard/overview.html")


def patients(request):
    return render(request, "dashboard/patients.html")


def referrals(request):
    return render(request, "dashboard/referrals.html")

def odreferrals(request):
    return render(request, "dashboard/odreferrals.html")


def encounters(request):
    return render(request, "dashboard/encounters.html")


def user_profile(request):
    return render(request, "dashboard/profile.html")


def authentication(request):
    return render(request, "dashboard/authentication.html")
