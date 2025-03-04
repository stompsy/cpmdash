from datetime import datetime
from dateutil.relativedelta import relativedelta
from django.utils.timezone import make_aware
from django.shortcuts import render
from .models import *  # Assuming Referral is your model name


def dashboard(request):
    odreferrals = ODReferrals.objects.all()
    referrals = Referrals.objects.all()
    patients = Patients.objects.all()
    encounters = Encounters.objects.all()

    today = datetime.today()
    current_quarter_start = datetime(today.year, 3 * ((today.month - 1) // 3) + 1, 1)
    last_quarter_start = current_quarter_start - relativedelta(months=3)
    last_quarter_end = current_quarter_start - relativedelta(days=1)

    # Convert to timezone-aware if using timezone support
    current_quarter_start = make_aware(current_quarter_start)
    last_quarter_start = make_aware(last_quarter_start)
    last_quarter_end = make_aware(last_quarter_end)

    # Get total patients count
    patients_count = patients.count

    # Get referral this/last quarter counts
    referrals_count = referrals.count
    this_quarter_referral_count = referrals.filter(
        date_received__gte=current_quarter_start
    ).count()
    last_quarter_referral_count = referrals.filter(
        date_received__range=(last_quarter_start, last_quarter_end)
    ).count()

    # Get total OD referrals count
    odreferrals_count = odreferrals.count
    this_quarter_odreferral_count = odreferrals.filter(
        referral_date__gte=current_quarter_start
    ).count()
    last_quarter_odreferral_count = odreferrals.filter(
        referral_date__range=(last_quarter_start, last_quarter_end)
    ).count()

    # Get total encounters count
    encounters_count = encounters.count
    this_quarter_encounter_count = encounters.filter(
        encounter_date__gte=current_quarter_start
    ).count()
    last_quarter_encounter_count = encounters.filter(
        encounter_date__range=(last_quarter_start, last_quarter_end)
    ).count()

    # Calculate referral difference
    referral_difference = this_quarter_referral_count - last_quarter_referral_count
    referral_percentage_difference = (
        (referral_difference / last_quarter_referral_count * 100)
        if last_quarter_referral_count > 0
        else 0
    )

    # Calculate OD referral difference
    odreferral_difference = (
        this_quarter_odreferral_count - last_quarter_odreferral_count
    )
    odreferral_percentage_difference = (
        (odreferral_difference / last_quarter_odreferral_count * 100)
        if last_quarter_odreferral_count > 0
        else 0
    )

    # Calculate encounter difference
    encounter_difference = this_quarter_encounter_count - last_quarter_encounter_count
    encounter_percentage_difference = (
        (encounter_difference / last_quarter_encounter_count * 100)
        if last_quarter_encounter_count > 0
        else 0
    )

    # Determine referral trend symbol
    if referral_difference > 0:
        referral_trend_symbol = "↗︎"  # Increase
    elif referral_difference < 0:
        referral_trend_symbol = "↘︎"  # Decrease
    else:
        referral_trend_symbol = "→"  # No change

    # Determine od referral trend symbol
    if odreferral_difference > 0:
        odreferral_trend_symbol = "↗︎"  # Increase
    elif odreferral_difference < 0:
        odreferral_trend_symbol = "↘︎"  # Decrease
    else:
        odreferral_trend_symbol = "→"  # No change

    # Determine encounter trend symbol
    if encounter_difference > 0:
        encounter_trend_symbol = "↗︎"  # Increase
    elif encounter_difference < 0:
        encounter_trend_symbol = "↘︎"  # Decrease
    else:
        encounter_trend_symbol = "→"  # No change

    title = "Dashboard"
    description = "This is a Dashboard page"

    context = {
        "title": title,
        "description": description,
        "patients_count": patients_count,
        "referrals_count": referrals_count,
        "odreferrals_count": odreferrals_count,
        "encounters_count": encounters_count,
        "this_quarter_referral_count": this_quarter_referral_count,
        "last_quarter_referral_count": last_quarter_referral_count,
        "this_quarter_odreferral_count": this_quarter_odreferral_count,
        "last_quarter_odreferral_count": last_quarter_odreferral_count,
        "this_quarter_encounter_count": this_quarter_encounter_count,
        "last_quarter_encounter_count": last_quarter_encounter_count,
        "referral_difference": referral_difference,
        "referral_percentage_difference": round(referral_percentage_difference, 1),
        "odreferral_difference": odreferral_difference,
        "odreferral_percentage_difference": round(odreferral_percentage_difference, 1),
        "encounter_difference": encounter_difference,
        "encounter_percentage_difference": round(encounter_percentage_difference, 1),
        "referral_trend_symbol": referral_trend_symbol,
        "odreferral_trend_symbol": odreferral_trend_symbol,
        "encounter_trend_symbol": encounter_trend_symbol,
    }

    return render(request, "dashboard/index.html", context)
