from datetime import date

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.shortcuts import render

from utils.ed_pricing import get_ed_pricing_reference_data


def _get_opshielding_context() -> dict[str, object]:
    updated_on = date(2026, 2, 19)
    return {
        "title": "PORT Referrals",
        "description": "Case Studies - OP Shielding Hope",
        "page_header_updated_at": updated_on,
        "page_header_updated_at_iso": updated_on.isoformat(),
    }


def cases(request):
    context = _get_opshielding_context()
    return render(request, "cases/opshieldinghope.html", context=context)


def opshield(request):
    context = _get_opshielding_context()
    return render(request, "cases/opshieldinghope.html", context=context)


def costsavings(request):
    """Render the Cost Savings Analysis page"""
    from ..charts.od_utils import get_cost_savings_metrics

    title = "Cost Savings Analysis"
    description = "Financial impact of Community Paramedic interventions"

    # Get dynamic cost savings metrics
    cost_metrics = get_cost_savings_metrics()
    ed_cost_reference = get_ed_pricing_reference_data()

    # Calculate rounded total savings to nearest $500,000 for grant justification
    total_savings = cost_metrics["total_savings"]
    rounded_savings = round(total_savings / 500000) * 500000

    updated_on = date(2026, 2, 19)

    context = {
        "title": title,
        "description": description,
        "cost_metrics": cost_metrics,
        "ed_cost_reference": ed_cost_reference,
        "rounded_total_savings": rounded_savings,
        "page_header_updated_at": updated_on,
        "page_header_updated_at_iso": updated_on.isoformat(),
    }
    return render(request, "cases/costsavings.html", context)


@login_required
def ed_pricing_explorer(request):
    """Render a searchable ED pricing explorer for staff users."""
    if not request.user.is_staff:
        return render(request, "access_denied.html")

    pricing_data = get_ed_pricing_reference_data()
    all_rows = pricing_data.get("ed_rows", [])

    facility = request.GET.get("facility", "all").strip()
    code_type = request.GET.get("code_type", "all").strip().upper()
    query = request.GET.get("q", "").strip().lower()
    sort_key = request.GET.get("sort", "gross_charge").strip()
    sort_dir = request.GET.get("dir", "desc").strip().lower()

    filtered_rows = all_rows
    if facility and facility != "all":
        filtered_rows = [row for row in filtered_rows if row["facility"] == facility]

    if code_type and code_type != "ALL":
        filtered_rows = [row for row in filtered_rows if row["code_type"] == code_type]

    if query:
        filtered_rows = [
            row
            for row in filtered_rows
            if query in row["procedure_description"].lower()
            or query in row["procedure_code"].lower()
            or query in row["code_type"].lower()
        ]

    sort_options = {
        "facility": lambda row: row["facility"],
        "procedure_code": lambda row: row["procedure_code"],
        "procedure_description": lambda row: row["procedure_description"],
        "gross_charge": lambda row: row["gross_charge"] if row["gross_charge"] is not None else -1,
        "discounted_cash": lambda row: (
            row["discounted_cash"] if row["discounted_cash"] is not None else -1
        ),
        "setting": lambda row: row["settings"],
    }
    sort_key_fn = sort_options.get(sort_key, sort_options["gross_charge"])
    reverse = sort_dir != "asc"
    filtered_rows = sorted(filtered_rows, key=sort_key_fn, reverse=reverse)

    code_type_options = sorted({row["code_type"] for row in all_rows if row.get("code_type")})

    paginator = Paginator(filtered_rows, 50)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    updated_on = date(2026, 7, 27)

    context = {
        "title": "ED Procedure Pricing Explorer",
        "description": "Search emergency department procedure costs from public hospital standard charge files.",
        "page_obj": page_obj,
        "facility": facility,
        "code_type": code_type,
        "query": request.GET.get("q", "").strip(),
        "facility_options": pricing_data.get("facilities", []),
        "code_type_options": code_type_options,
        "sort": sort_key,
        "dir": sort_dir,
        "total_results": len(filtered_rows),
        "total_ed_rows": len(all_rows),
        "ed_cost_reference": pricing_data,
        "page_header_updated_at": updated_on,
        "page_header_updated_at_iso": updated_on.isoformat(),
    }
    return render(request, "cases/ed_pricing_explorer.html", context)


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

    updated_on = date(2026, 2, 19)

    context = {
        "title": "PAFD Community Paramedicine Program Timeline",
        "description": "Milestones that have shaped the Port Angeles Fire Department's community paramedicine evolution.",
        "timeline_entries": timeline_entries,
        "page_header_updated_at": updated_on,
        "page_header_updated_at_iso": updated_on.isoformat(),
        "page_header_read_time": "5 min read",
    }

    return render(request, "timeline/index.html", context)
