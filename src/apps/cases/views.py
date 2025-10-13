from django.http import HttpResponse
from django.shortcuts import redirect, render

from utils.theme import get_theme_from_request

from ..charts.overdose.od_density_heatmap import build_chart_od_density_heatmap
from ..charts.overdose.od_hourly_breakdown import build_chart_od_hourly_breakdown
from ..charts.overdose.od_shift_scenarios import (
    build_chart_cost_benefit_analysis,
    build_chart_shift_scenarios,
)


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
    return redirect("dashboard:odreferrals_shift_coverage")


def repeatods(request):
    return redirect("dashboard:odreferrals_repeat_overdoses")


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


def timeline(request):
    """Render the Community Paramedicine program timeline."""

    timeline_entries = [
        {
            "year": "2019",
            "tagline": "Program pilot launched",
            "lead": (
                "Implemented the Community Paramedic program by reallocating a firefighter/paramedic from line staff as a "
                "one-year pilot with union support."
            ),
            "bullets": [
                "Decrease overutilization of 9-1-1 for non-emergent medical needs",
                "Decrease overutilization of Emergency Department for non-emergent medical needs",
                "Improve overall health and wellness among high system utilizers",
            ],
        },
        {
            "year": "2020",
            "tagline": "Strengthening sustainability",
            "lead": (
                "North Olympic Healthcare Network and PAFD collaborated through United Healthcare Fund grant dollars to "
                "augment and sustain the program."
            ),
            "bullets": [
                "Secured funding to hire three community paramedics",
                "Expanded in-home medical assessments and care coordination with primary care providers",
            ],
        },
        {
            "year": "2021",
            "tagline": "Scaling capacity and outreach",
            "lead": "Expanded clinical capabilities and outreach partnerships while continuing pandemic support.",
            "bullets": [
                "Hired Brian Gerdes in January with multi-year grant support",
                "Delivered regional vaccine clinics for first responders, health professionals, and homebound residents (~600 doses)",
                "Launched Treat & Release model enabling point-of-care labs, IV therapy, and tailored care plans",
                "Gained read-only access to EPIC EHR for coordinated care",
                "Hired Kristin Fox in July (two-year grant funding)",
                "Partnered with OPCC and Rediscovery outreach teams to fill care gaps for unhoused and unsheltered residents, including wound care and medication administration",
            ],
        },
        {
            "year": "2022",
            "tagline": "Maintaining community health support",
            "lead": "Served as mental health support for residents isolated by the pandemic while continuing preventive care.",
            "bullets": [
                "Administered ~800 additional vaccinations, adding influenza, shingles, and other immunizations",
            ],
        },
        {
            "year": "2023",
            "tagline": "Responding to overdose escalation",
            "lead": "Began investigating the rise in fatal overdoses across the service area.",
            "sections": [
                {
                    "title": "Jan–Jun 2023 snapshot",
                    "items": [
                        "102 overdose responses",
                        "60% of survivors refused treatment and transport",
                        "0% MOUD or MAT referrals",
                        "Elevated repeat overdose rates",
                        "Compassion fatigue increasing among first responders",
                    ],
                },
                {
                    "title": "Funding milestone",
                    "items": [
                        "Operation Shielding Hope awarded opioid settlement funding",
                    ],
                },
            ],
        },
        {
            "year": "2024",
            "tagline": "Launching the Post Overdose Response Team",
            "lead": (
                "Established the PORT pilot and deepened co-response partnerships to expand CPM scope of practice."
            ),
            "bullets": [
                "Formed partnerships between CPM, community policing, and co-response teams",
                "PORT pilot approved by Washington DOH, expanding CPM scope",
                "Authorized buprenorphine administration",
                "CPs began responding to every overdose during shift hours",
                "PAFD crews now refer all overdose survivors to the CPM office for follow-up",
                "CPM provides warm handoffs to co-response partners",
                "Deployed i-Stat point-of-care testing devices and Butterfly ultrasound handhelds",
            ],
        },
        {
            "year": "2025",
            "tagline": "Scaling PORT infrastructure",
            "lead": (
                "Hired additional community EMTs and invested opioid settlement and grant funding into response capacity, "
                "training, and outreach."
            ),
            "bullets": [
                "January 2025: onboarded Tatiana and Heather",
                "UW & CROA grants awarded to accelerate program growth",
            ],
            "sections": [
                {
                    "title": "Personnel & equipment investments",
                    "items": [
                        "Two full-time community EMTs",
                        "MOUD Overdose Response Unit transport and equipment",
                        "UTV for outreach, remote response, and patient extrication",
                        "Advanced life support equipment including cardiac monitors",
                        "Field blood analyzer (i-Stat) and specialized responder bags",
                    ],
                },
                {
                    "title": "Clinical supply readiness",
                    "items": [
                        "Co-response unit stocked with medical, trauma, airway, and outreach kits",
                        "Personal protective equipment and disposable supplies",
                    ],
                },
                {
                    "title": "Outreach & engagement",
                    "items": [
                        "Public education materials and referral service collateral",
                        "Behavioral health and SUD support supplies for community engagement",
                    ],
                },
                {
                    "title": "Training & workforce support",
                    "items": [
                        "Co-responder MOUD and behavioral health training",
                        "All FD staff continuing education and monthly co-responder run reviews",
                        "Train-the-trainer exercises with dedicated wages, benefits, and travel support",
                        "Two focused 8-hour behavioral health training sessions",
                        "Investments in multimedia resources and facility rentals for instruction",
                    ],
                },
            ],
        },
    ]

    context = {
        "title": "PAFD Community Paramedicine Program Timeline",
        "description": "Milestones that have shaped the Port Angeles Fire Department's community paramedicine evolution.",
        "timeline_entries": timeline_entries,
    }

    return render(request, "timeline/index.html", context)


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
