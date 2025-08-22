import pandas as pd
from django.db.models import Q

from ..core.models import ODReferrals


def get_odreferral_counts():
    total = ODReferrals.objects.count()
    dates = ODReferrals.objects.values_list("od_date", flat=True)
    df = pd.DataFrame({"od_date": dates}).dropna(subset=["od_date"])

    if df.empty:
        by_year = {}
    else:
        df["od_date"] = pd.to_datetime(df["od_date"])
        df["year"] = df["od_date"].dt.year
        by_year = (
            df["year"]
            .value_counts()
            .sort_index(ascending=False)  # descending
            .to_dict()
        )

    return {"total": total, "by_year": by_year}


def get_od_metrics(year: int, population: int = 20_000) -> dict:
    """
    Returns a dict with:
        - count: total OD referrals in `year`
        - rate_per_100k: referrals per 100 000 population
        - percentage: % of population with at least one referral
    """
    # fetch count
    count = ODReferrals.objects.filter(od_date__year=year).count()

    # guard zero‐pop
    if not population:
        return {"count": count, "rate_per_100k": 0.0, "percentage": 0.0}

    # compute
    rate_per_100k = (count / population) * 100_000
    percentage = (count / population) * 100

    return {
        "count": count,
        "rate_per_100k": rate_per_100k,
        "percentage": percentage,
    }


def get_od_fatality_rate_year(
    year: int, fatal_dispositions: list[str], copa_population: int = 20_000
) -> float:
    """
    Returns the fatal overdose referral rate per 100 000 population for `year`,
    where 'disposition' contains any of the provided strings.
    """
    if not copa_population:
        return 0.0

    # Build a Q object matching any of the dispositions
    q_filter = Q()
    for disp in fatal_dispositions:
        q_filter |= Q(disposition__icontains=disp)

    fatal_count = ODReferrals.objects.filter(od_date__year=year).filter(q_filter).count()

    return (fatal_count / copa_population) * 100_000
