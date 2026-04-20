"""
Co-occurring disorder charts for the Patients and OD Referrals pages.

Charts built from the de-identified od_sud_aud.json dataset (03/2024-03/2025).
Each builder follows the standard cpmdash pattern: accept ``theme``, return
Plotly HTML via ``plotly.offline.plot()``.
"""

from __future__ import annotations

from collections import Counter

import plotly.graph_objects as go
from plotly.offline import plot

from utils.chart_colors import CHART_COLORS_VIBRANT, CHART_COLORS_WARM
from utils.plotly import get_theme_colors, style_plotly_layout
from utils.tailwind_colors import TAILWIND_COLORS

from .data_loader import get_opioid_patients

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plot_html(fig: go.Figure) -> str:
    return plot(
        fig,
        output_type="div",
        config={"displayModeBar": False},
    )


# Map conditions to higher-level categories for color-coding
_BH_CATEGORY = {
    "Depression": "Mood",
    "Bipolar": "Mood",
    "Seasonal Affective Disorder": "Mood",
    "Anxiety": "Anxiety",
    "PTSD": "Anxiety",
    "Paranoia": "Anxiety",
    "Schizophrenia": "Psychotic",
    "Schizoaffective": "Psychotic",
    "Psychosis": "Psychotic",
    "Delusional Disorder": "Psychotic",
    "Schizotypal": "Psychotic",
    "Suicidal Ideation": "Crisis",
    "Suicide Attempt": "Crisis",
    "ADHD": "Neurodevelopmental",
    "Autism": "Neurodevelopmental",
    "ODD": "Neurodevelopmental",
    "PPD": "Other",
    "Mood Disorder": "Mood",
    "REM Behavioral Disorder": "Other",
}

_CATEGORY_COLORS = {
    "Mood": TAILWIND_COLORS["violet-500"],
    "Anxiety": TAILWIND_COLORS["amber-500"],
    "Psychotic": TAILWIND_COLORS["rose-500"],
    "Crisis": TAILWIND_COLORS["red-600"],
    "Neurodevelopmental": TAILWIND_COLORS["cyan-500"],
    "Other": TAILWIND_COLORS["slate-400"],
}


# ---------------------------------------------------------------------------
# PATIENTS PAGE: BH Prevalence Bar
# ---------------------------------------------------------------------------


def build_bh_prevalence_bar(theme: str = "dark") -> str:
    """Horizontal bar chart of behavioral health conditions among OUD patients.

    Why this chart matters:
    -   Shows the sheer *density* of co-occurring behavioral health conditions
        in the opioid overdose population.  Nearly 70% have at least one
        diagnosed BH condition, but the breakdowns reveal clustering that
        simple prevalence numbers hide.
    -   Color-coded by condition category (mood, anxiety, psychotic, crisis,
        neurodevelopmental) so viewers can instantly see that mood + anxiety
        disorders dominate, but psychotic-spectrum disorders are a
        surprisingly large third bucket.
    """
    patients = get_opioid_patients()
    counter: Counter[str] = Counter()
    for p in patients:
        for cond in p["bh_conditions"]:
            counter[cond] += 1

    # Filter to conditions appearing 3+ times for readability
    items = [(cond, cnt) for cond, cnt in counter.most_common() if cnt >= 3]
    if not items:
        return ""

    conditions = [c for c, _ in items]
    counts = [n for _, n in items]
    total = len(patients)
    pcts = [round(100 * n / total, 1) for n in counts]

    # Color by category
    colors = [
        _CATEGORY_COLORS.get(_BH_CATEGORY.get(c, "Other"), TAILWIND_COLORS["slate-400"])
        for c in conditions
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=conditions,
            x=counts,
            orientation="h",
            marker_color=colors,
            text=[f"{n}  ({p}%)" for n, p in zip(counts, pcts, strict=False)],
            textposition="outside",
            textfont=dict(size=13),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Patients: %{x}<br>"
                "Share of OUD cohort: %{customdata}%"
                "<extra></extra>"
            ),
            customdata=pcts,
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=420,
        margin=dict(t=30, l=180, r=80, b=30),
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed", categoryorder="total ascending"),
        xaxis=dict(title=None, showticklabels=False, showgrid=False),
        bargap=0.35,
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# PATIENTS PAGE: Chronic Illness Treemap
# ---------------------------------------------------------------------------


def build_chronic_illness_treemap(theme: str = "dark") -> str:
    """Treemap of chronic illnesses grouped by body system.

    Why a treemap:
    -   There are 50+ distinct chronic conditions in the dataset.  A bar chart
        would be noisy.  A treemap lets you see *relative burden* at the
        category level (hepatic, cardiac, pain, respiratory) while still
        showing individual conditions within each group.
    -   Hepatitis C dominates — not surprising given IV drug use — but the
        cardiac cluster (CHF, CAD, HTN, afib) is arguably more lethal and
        gets less attention in SUD treatment planning.
    """
    patients = get_opioid_patients()
    counter: Counter[str] = Counter()
    for p in patients:
        for cond in p["chronic_conditions"]:
            counter[cond] += 1

    # Group into body systems
    _SYSTEM_MAP = {
        "Hepatitis C": "Hepatic/Infectious",
        "Hepatitis B": "Hepatic/Infectious",
        "HIV": "Hepatic/Infectious",
        "COPD": "Respiratory",
        "Diabetes (Type 2)": "Metabolic",
        "Diabetes": "Metabolic",
        "Hypertension": "Cardiovascular",
        "Heart Failure": "Cardiovascular",
        "Coronary Artery Disease": "Cardiovascular",
        "Atrial Fibrillation": "Cardiovascular",
        "Peripheral Artery Disease": "Cardiovascular",
        "Chronic Kidney Disease": "Renal",
        "End-Stage Renal Disease": "Renal",
        "Seizures": "Neurological",
        "Epilepsy": "Neurological",
        "Migraines": "Neurological",
        "Traumatic Brain Injury": "Neurological",
        "Rheumatoid Arthritis": "Pain/Musculoskeletal",
        "Arthritis": "Pain/Musculoskeletal",
        "Chronic Back Pain": "Pain/Musculoskeletal",
        "Chronic Pain": "Pain/Musculoskeletal",
        "Back Pain": "Pain/Musculoskeletal",
        "Spondylosis": "Pain/Musculoskeletal",
        "Fibromyalgia": "Pain/Musculoskeletal",
        "Kyphosis": "Pain/Musculoskeletal",
        "Spinal Stenosis": "Pain/Musculoskeletal",
    }

    # Filter to conditions appearing 2+ times
    items = [(cond, cnt) for cond, cnt in counter.most_common() if cnt >= 2]
    if not items:
        return ""

    labels = []
    parents = []
    values = []
    colors_list = []

    system_color_map = {
        "Hepatic/Infectious": TAILWIND_COLORS["amber-500"],
        "Cardiovascular": TAILWIND_COLORS["rose-500"],
        "Pain/Musculoskeletal": TAILWIND_COLORS["violet-500"],
        "Neurological": TAILWIND_COLORS["cyan-500"],
        "Respiratory": TAILWIND_COLORS["teal-500"],
        "Metabolic": TAILWIND_COLORS["emerald-500"],
        "Renal": TAILWIND_COLORS["blue-500"],
        "Other": TAILWIND_COLORS["slate-500"],
    }

    # Collect system totals
    system_totals: Counter[str] = Counter()
    for cond, cnt in items:
        system = _SYSTEM_MAP.get(cond, "Other")
        system_totals[system] += cnt

    # Add system-level parents
    for system, total in system_totals.most_common():
        labels.append(system)
        parents.append("")
        values.append(total)
        colors_list.append(system_color_map.get(system, TAILWIND_COLORS["slate-500"]))

    # Add individual conditions as children
    for cond, cnt in items:
        system = _SYSTEM_MAP.get(cond, "Other")
        labels.append(cond)
        parents.append(system)
        values.append(cnt)
        base_color = system_color_map.get(system, TAILWIND_COLORS["slate-500"])
        colors_list.append(base_color)

    tc = get_theme_colors(theme)

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors_list),
            textfont=dict(size=14, color=tc["font_color"]),
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Patients: %{value}<extra></extra>",
            branchvalues="total",
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=400,
        margin=dict(t=20, l=10, r=10, b=10),
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# OD REFERRALS PAGE: Repeat OD × BH Comparison
# ---------------------------------------------------------------------------


def build_repeat_od_bh_comparison(theme: str = "dark") -> str:
    """Grouped bar comparing BH prevalence in repeat-OD vs single-OD patients.

    Why this chart matters:
    -   This is the single most actionable correlation in the dataset.
        79.2% of repeat-OD patients have a diagnosed behavioral health
        disorder compared to 60.4% of single-OD patients.  That ~19-point
        gap is a screaming signal that untreated mental health is a driver
        of recurrent overdose.
    -   The implication: post-reversal care that ignores BH screening is
        leaving the biggest risk factor on the table.
    """
    patients = get_opioid_patients()

    repeat = [p for p in patients if p["repeat_od_count"] > 0]
    single = [p for p in patients if p["repeat_od_count"] == 0]

    repeat_bh = sum(1 for p in repeat if p["has_bh"])
    single_bh = sum(1 for p in single if p["has_bh"])

    repeat_pct = round(100 * repeat_bh / len(repeat), 1) if repeat else 0
    single_pct = round(100 * single_bh / len(single), 1) if single else 0

    tc = get_theme_colors(theme)

    fig = go.Figure()

    categories = ["Repeat Overdose<br>Patients", "Single Overdose<br>Patients"]

    # BH present
    fig.add_trace(
        go.Bar(
            x=categories,
            y=[repeat_pct, single_pct],
            name="With BH Disorder",
            marker_color=TAILWIND_COLORS["violet-500"],
            text=[f"{repeat_pct}%", f"{single_pct}%"],
            textposition="outside",
            textfont=dict(size=16, color=tc["font_color"]),
            hovertemplate=("<b>%{x}</b><br>With BH disorder: %{y}%<br><extra></extra>"),
        )
    )

    # BH absent
    repeat_no_pct = round(100 - repeat_pct, 1)
    single_no_pct = round(100 - single_pct, 1)
    fig.add_trace(
        go.Bar(
            x=categories,
            y=[repeat_no_pct, single_no_pct],
            name="No BH Disorder",
            marker_color=TAILWIND_COLORS["slate-400"],
            text=[f"{repeat_no_pct}%", f"{single_no_pct}%"],
            textposition="outside",
            textfont=dict(size=16, color=tc["font_color"]),
            hovertemplate=("<b>%{x}</b><br>No BH disorder: %{y}%<br><extra></extra>"),
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=380,
        show_legend=True,
        margin=dict(t=40, l=50, r=30, b=60),
        y_title="Percentage of Group",
    )
    fig.update_layout(
        barmode="group",
        bargap=0.3,
        bargroupgap=0.15,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color=tc["font_color"]),
        ),
        yaxis=dict(range=[0, 100], dtick=20, ticksuffix="%"),
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# OD REFERRALS PAGE: BH × Sex Breakdown
# ---------------------------------------------------------------------------


def build_bh_by_sex(theme: str = "dark") -> str:
    """Stacked bar showing BH prevalence by sex among OUD patients.

    Women in this cohort carry a disproportionate BH burden (73% vs 60%
    for men), despite being only 28% of the overdose population.  This
    has real implications for treatment design — women-specific SUD
    programs need integrated BH services even more urgently than the
    general SUD population.
    """
    patients = get_opioid_patients()
    males = [p for p in patients if p["sex"] == "Male"]
    females = [p for p in patients if p["sex"] == "Female"]

    m_bh = sum(1 for p in males if p["has_bh"])
    f_bh = sum(1 for p in females if p["has_bh"])
    m_no = len(males) - m_bh
    f_no = len(females) - f_bh

    m_pct = round(100 * m_bh / len(males), 1) if males else 0
    f_pct = round(100 * f_bh / len(females), 1) if females else 0

    tc = get_theme_colors(theme)
    categories = [f"Male (n={len(males)})", f"Female (n={len(females)})"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=categories,
            y=[m_bh, f_bh],
            name="With BH Disorder",
            marker_color=TAILWIND_COLORS["violet-500"],
            text=[f"{m_pct}%", f"{f_pct}%"],
            textposition="inside",
            textfont=dict(size=15, color="white"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=categories,
            y=[m_no, f_no],
            name="No BH Disorder",
            marker_color=TAILWIND_COLORS["slate-500"],
            text=[f"{100 - m_pct}%", f"{100 - f_pct}%"],
            textposition="inside",
            textfont=dict(size=15, color="white"),
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=350,
        show_legend=True,
        margin=dict(t=40, l=50, r=30, b=50),
        y_title="Patients",
    )
    fig.update_layout(
        barmode="stack",
        bargap=0.4,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color=tc["font_color"]),
        ),
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# OD REFERRALS PAGE: Co-Occurrence Heatmap
# ---------------------------------------------------------------------------


def _heatmap_colorscale(theme: str) -> list[list[str | float]]:
    """Return a violet colorscale appropriate for the given theme."""
    if theme == "dark":
        return [
            [0, "rgba(30, 41, 59, 0.3)"],
            [0.25, TAILWIND_COLORS["violet-900"]],
            [0.5, TAILWIND_COLORS["violet-700"]],
            [0.75, TAILWIND_COLORS["violet-500"]],
            [1, TAILWIND_COLORS["violet-400"]],
        ]
    return [
        [0, TAILWIND_COLORS["slate-50"]],
        [0.25, TAILWIND_COLORS["violet-100"]],
        [0.5, TAILWIND_COLORS["violet-300"]],
        [0.75, TAILWIND_COLORS["violet-500"]],
        [1, TAILWIND_COLORS["violet-700"]],
    ]


def build_bh_cooccurrence_heatmap(theme: str = "dark") -> str:
    """Heatmap showing how often BH conditions appear together in the same patient.

    This is the chart that makes clinicians lean forward.  It reveals that
    depression + anxiety is the most common pairing (not surprising), but
    schizophrenia + suicidal ideation and PTSD + depression are strong
    runners-up.  These pairings define the *complexity* of the patient
    population and argue against single-diagnosis treatment models.
    """
    patients = get_opioid_patients()

    # Only consider conditions with 5+ occurrences for a readable matrix
    all_conds: Counter[str] = Counter()
    for p in patients:
        for c in p["bh_conditions"]:
            all_conds[c] += 1

    top_conds = [c for c, n in all_conds.most_common() if n >= 5]
    if len(top_conds) < 2:
        return ""

    # Build co-occurrence matrix
    n = len(top_conds)
    matrix = [[0] * n for _ in range(n)]
    for p in patients:
        conds_in = [c for c in p["bh_conditions"] if c in top_conds]
        for i, c1 in enumerate(top_conds):
            for j, c2 in enumerate(top_conds):
                if c1 in conds_in and c2 in conds_in:
                    matrix[i][j] += 1

    tc = get_theme_colors(theme)
    colorscale = _heatmap_colorscale(theme)

    # Build text annotations for cells
    text_matrix = []
    for row in matrix:
        text_row = []
        for val in row:
            text_row.append(str(val) if val > 0 else "")
        text_matrix.append(text_row)

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=top_conds,
            y=top_conds,
            colorscale=colorscale,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=14, color=tc["font_color"]),
            hovertemplate=("<b>%{x}</b> × <b>%{y}</b><br>Co-occurrences: %{z}<extra></extra>"),
            showscale=False,
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=420,
        margin=dict(t=20, l=140, r=20, b=120),
    )
    fig.update_layout(
        xaxis=dict(
            side="bottom",
            tickangle=-45,
            showgrid=False,
        ),
        yaxis=dict(
            autorange="reversed",
            showgrid=False,
        ),
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# PATIENTS PAGE: Age × BH Status Box Plot
# ---------------------------------------------------------------------------


def build_age_bh_boxplot(theme: str = "dark") -> str:
    """Box plot comparing age distributions for patients with/without BH disorders.

    This answers a natural question: are BH-burdened patients younger (more
    years of exposure ahead) or older (more accumulated damage)?  The answer
    has implications for prevention vs. harm-reduction resource allocation.
    """
    patients = get_opioid_patients()
    with_bh = [p["age"] for p in patients if p["has_bh"] and p["age"]]
    without_bh = [p["age"] for p in patients if not p["has_bh"] and p["age"]]

    if not with_bh or not without_bh:
        return ""

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=with_bh,
            name="With BH Disorder",
            marker_color=TAILWIND_COLORS["violet-500"],
            boxmean=True,
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Box(
            y=without_bh,
            name="No BH Disorder",
            marker_color=TAILWIND_COLORS["slate-400"],
            boxmean=True,
            line=dict(width=2),
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=380,
        margin=dict(t=30, l=50, r=30, b=50),
        y_title="Age (years)",
    )
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False),
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# REFERRALS PAGE: AUD Co-Occurrence Donut
# ---------------------------------------------------------------------------


def build_aud_cooccurrence_donut(theme: str = "dark") -> str:
    """Donut chart showing AUD prevalence among OUD patients.

    24.35% of opioid overdose patients also have diagnosed AUD.  That's
    nearly 1 in 4, and it matters because alcohol + opioid polysubstance
    use dramatically increases overdose lethality — alcohol potentiates
    respiratory depression.  Referral pathways need to address both.
    """
    patients = get_opioid_patients()
    aud_yes = sum(1 for p in patients if p.get("aud") == "Yes")
    aud_no = len(patients) - aud_yes
    aud_pct = round(100 * aud_yes / len(patients), 1) if patients else 0

    tc = get_theme_colors(theme)

    fig = go.Figure(
        go.Pie(
            labels=["AUD Co-Occurring", "OUD Only"],
            values=[aud_yes, aud_no],
            hole=0.55,
            marker=dict(
                colors=[
                    TAILWIND_COLORS["amber-500"],
                    TAILWIND_COLORS["slate-500"],
                ]
            ),
            textinfo="percent+label",
            textfont=dict(size=14, color=tc["font_color"]),
            hovertemplate=(
                "<b>%{label}</b><br>Patients: %{value}<br>Share: %{percent}<extra></extra>"
            ),
        )
    )

    # Center annotation
    fig.add_annotation(
        text=f"<b>{aud_pct}%</b><br>AUD",
        showarrow=False,
        font=dict(size=22, color=tc["font_color"]),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=340,
        margin=dict(t=20, l=20, r=20, b=20),
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# REFERRALS PAGE: SUD Substance Breakdown (Section 2 data)
# ---------------------------------------------------------------------------


def build_sud_substance_breakdown(theme: str = "dark") -> str:
    """Bar chart of substance types from the broader SUD cohort (non-opioid).

    The second section of the dataset captures patients whose primary
    impression was non-opioid substance use — methamphetamine dominates,
    followed by cannabis and polysubstance.  This fills a blind spot: the
    opioid-centric view misses the meth epidemic happening in parallel.
    """
    from .data_loader import get_sud_patients

    patients = get_sud_patients()

    counter: Counter[str] = Counter()
    for p in patients:
        sub = p.get("substance")
        if not sub or sub == "None":
            continue
        # Normalize multi-substance entries
        for s in sub.split(","):
            s = s.strip()
            if "?" in s:
                s = s.replace("?", "").strip()
            if s.lower() in ("unkn", "unknown"):
                s = "Unknown"
            if s:
                counter[s.title()] += 1

    items = [(s, n) for s, n in counter.most_common() if n >= 1]
    if not items:
        return ""

    substances = [s for s, _ in items]
    counts = [n for _, n in items]

    fig = go.Figure(
        go.Bar(
            x=substances,
            y=counts,
            marker_color=CHART_COLORS_WARM[: len(substances)],
            text=counts,
            textposition="outside",
            textfont=dict(size=14),
            hovertemplate="<b>%{x}</b><br>Patients: %{y}<extra></extra>",
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=340,
        margin=dict(t=30, l=50, r=30, b=60),
        y_title="Patients",
    )
    fig.update_layout(
        xaxis=dict(showgrid=False),
        bargap=0.35,
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# NEW: BH Prevalence by Age Bracket
# ---------------------------------------------------------------------------


def build_bh_by_age_bracket(theme: str = "dark") -> str:
    """Grouped bar showing BH prevalence percentage by age bracket.

    The data mining revealed a non-linear pattern: BH prevalence peaks at
    35-44 (77.4%) and 25-34 (70.3%), then dips at 45-54 (52.6%) before
    climbing again at 55-64 (61.5%).  That mid-life dip is worth investigating
    — it could reflect survivor bias (severely ill patients dying younger)
    or a generational gap in BH diagnosis rates.
    """
    patients = get_opioid_patients()

    brackets = [
        ("18-24", 18, 25),
        ("25-34", 25, 35),
        ("35-44", 35, 45),
        ("45-54", 45, 55),
        ("55-64", 55, 65),
        ("65+", 65, 200),
    ]

    tc = get_theme_colors(theme)
    labels = []
    bh_pcts = []
    no_bh_pcts = []
    hover_texts = []

    for label, lo, hi in brackets:
        in_bracket = [p for p in patients if p["age"] and lo <= p["age"] < hi]
        if not in_bracket:
            continue
        bh = sum(1 for p in in_bracket if p["has_bh"])
        pct = round(100 * bh / len(in_bracket), 1)
        labels.append(label)
        bh_pcts.append(pct)
        no_bh_pcts.append(round(100 - pct, 1))
        hover_texts.append(f"{bh}/{len(in_bracket)} patients")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=bh_pcts,
            name="With BH Disorder",
            marker_color=TAILWIND_COLORS["violet-500"],
            text=[f"{p}%" for p in bh_pcts],
            textposition="outside",
            textfont=dict(size=13, color=tc["font_color"]),
            customdata=hover_texts,
            hovertemplate="<b>Age %{x}</b><br>BH prevalence: %{y}%<br>%{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=no_bh_pcts,
            name="No BH Disorder",
            marker_color=TAILWIND_COLORS["slate-400"],
            text=[f"{p}%" for p in no_bh_pcts],
            textposition="outside",
            textfont=dict(size=13, color=tc["font_color"]),
            hovertemplate="<b>Age %{x}</b><br>No BH: %{y}%<extra></extra>",
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=380,
        show_legend=True,
        margin=dict(t=40, l=50, r=30, b=50),
        y_title="Percentage",
        x_title="Age Bracket",
    )
    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        yaxis=dict(range=[0, 100], dtick=20, ticksuffix="%"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color=tc["font_color"]),
        ),
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# NEW: AUD by Age + Sex
# ---------------------------------------------------------------------------


def build_aud_by_age_sex(theme: str = "dark") -> str:
    """Grouped bar showing AUD prevalence by age bracket, split by sex.

    Males have 2x the AUD rate (24.7% vs 13.5%), but the age trajectory
    is dramatic: AUD climbs from 7.7% in 18-24 to 38.5% in 55-64.
    This isn't just about younger people partying — it's about decades of
    escalating alcohol dependency that compounds the opioid risk.
    """
    patients = get_opioid_patients()

    brackets = [
        ("18-24", 18, 25),
        ("25-34", 25, 35),
        ("35-44", 35, 45),
        ("45-54", 45, 55),
        ("55-64", 55, 65),
        ("65+", 65, 200),
    ]

    tc = get_theme_colors(theme)
    labels = []
    male_pcts = []
    female_pcts = []

    for label, lo, hi in brackets:
        in_bracket = [p for p in patients if p["age"] and lo <= p["age"] < hi]
        if not in_bracket:
            continue
        males = [p for p in in_bracket if p["sex"] == "Male"]
        females = [p for p in in_bracket if p["sex"] == "Female"]
        m_aud = sum(1 for p in males if p.get("aud") == "Yes")
        f_aud = sum(1 for p in females if p.get("aud") == "Yes")
        labels.append(label)
        male_pcts.append(round(100 * m_aud / len(males), 1) if males else 0)
        female_pcts.append(round(100 * f_aud / len(females), 1) if females else 0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=male_pcts,
            name="Male",
            marker_color=TAILWIND_COLORS["blue-500"],
            text=[f"{p}%" for p in male_pcts],
            textposition="outside",
            textfont=dict(size=12, color=tc["font_color"]),
            hovertemplate="<b>Age %{x} — Male</b><br>AUD: %{y}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=female_pcts,
            name="Female",
            marker_color=TAILWIND_COLORS["rose-400"],
            text=[f"{p}%" for p in female_pcts],
            textposition="outside",
            textfont=dict(size=12, color=tc["font_color"]),
            hovertemplate="<b>Age %{x} — Female</b><br>AUD: %{y}%<extra></extra>",
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=380,
        show_legend=True,
        margin=dict(t=40, l=50, r=30, b=50),
        y_title="AUD Prevalence (%)",
        x_title="Age Bracket",
    )
    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        yaxis=dict(range=[0, 60], dtick=10, ticksuffix="%"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color=tc["font_color"]),
        ),
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# NEW: Patient Complexity Funnel
# ---------------------------------------------------------------------------


def build_complexity_funnel(theme: str = "dark") -> str:
    """Horizontal bar showing patient complexity layers — from no conditions
    through single-burden to the dreaded triple burden (BH + Chronic + AUD).

    Only 23.8% of OD patients have *zero* comorbidities.  The rest are
    distributed across escalating complexity tiers.  This chart makes the
    case for integrated care better than any narrative paragraph could.
    """
    patients = get_opioid_patients()
    total = len(patients)

    # Classify each patient
    none_count = 0
    bh_only = 0
    chronic_only = 0
    bh_chronic = 0
    triple = 0
    aud_only = 0

    for p in patients:
        has_bh = p["has_bh"]
        has_chr = p["has_chronic"]
        has_aud = p.get("aud") == "Yes"

        if has_bh and has_chr and has_aud:
            triple += 1
        elif has_bh and has_chr:
            bh_chronic += 1
        elif has_bh and has_aud:
            bh_chronic += 1  # BH + AUD also counts as dual
        elif has_chr and has_aud:
            bh_chronic += 1  # Chronic + AUD also counts as dual
        elif has_bh:
            bh_only += 1
        elif has_chr:
            chronic_only += 1
        elif has_aud:
            aud_only += 1
        else:
            none_count += 1

    # Build from most complex to least
    labels = [
        "Triple Burden<br>(BH + Chronic + AUD)",
        "Dual Burden<br>(Any two conditions)",
        "Single Burden<br>(BH, Chronic, or AUD only)",
        "No Comorbidities",
    ]
    values = [
        triple,
        bh_chronic,
        bh_only + chronic_only + aud_only,
        none_count,
    ]
    colors = [
        TAILWIND_COLORS["red-500"],
        TAILWIND_COLORS["amber-500"],
        TAILWIND_COLORS["violet-400"],
        TAILWIND_COLORS["slate-400"],
    ]
    pcts = [round(100 * v / total, 1) for v in values]

    tc = get_theme_colors(theme)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=colors,
            text=[f"{v} ({p}%)" for v, p in zip(values, pcts, strict=False)],
            textposition="outside",
            textfont=dict(size=14, color=tc["font_color"]),
            hovertemplate="<b>%{y}</b><br>Patients: %{x}<br>Share: %{customdata}%<extra></extra>",
            customdata=pcts,
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=320,
        margin=dict(t=20, l=200, r=90, b=30),
    )
    fig.update_layout(
        yaxis=dict(showgrid=False),
        xaxis=dict(title=None, showticklabels=False, showgrid=False),
        bargap=0.3,
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# NEW: Multiple BH Conditions Distribution
# ---------------------------------------------------------------------------


def build_multi_bh_histogram(theme: str = "dark") -> str:
    """Bar chart showing how many BH conditions each BH-positive patient has.

    70% of BH-positive patients have MULTIPLE conditions (mean 2.14, max 5).
    This chart kills the fantasy that we're dealing with single-diagnosis
    patients.  The modal BH patient has 2 conditions, and a non-trivial
    chunk has 4-5.  That's not depression OR anxiety — it's depression
    AND anxiety AND PTSD AND substance-induced psychosis.
    """
    patients = get_opioid_patients()
    bh_patients = [p for p in patients if p["has_bh"]]

    if not bh_patients:
        return ""

    counts = Counter(len(p["bh_conditions"]) for p in bh_patients)
    max_conds = max(counts.keys())
    x_vals = list(range(1, max_conds + 1))
    y_vals = [counts.get(x, 0) for x in x_vals]

    tc = get_theme_colors(theme)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(x) for x in x_vals],
            y=y_vals,
            marker_color=CHART_COLORS_VIBRANT[: len(x_vals)],
            text=y_vals,
            textposition="outside",
            textfont=dict(size=14, color=tc["font_color"]),
            hovertemplate=("<b>%{x} condition(s)</b><br>Patients: %{y}<br><extra></extra>"),
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=340,
        margin=dict(t=30, l=50, r=30, b=50),
        y_title="Patients",
        x_title="Number of BH Conditions",
    )
    fig.update_layout(bargap=0.3)
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# NEW: SUD Substance × BH Rate
# ---------------------------------------------------------------------------


def build_sud_bh_by_substance(theme: str = "dark") -> str:
    """Bar chart showing BH prevalence by primary substance in the SUD cohort.

    Meth patients have 67% BH prevalence.  Meth+Cannabis is 100%.  This
    chart exposes the substance-specific mental health burden that generic
    SUD treatment programs tend to flatten into a single bucket.
    """
    from .data_loader import get_sud_patients

    patients = get_sud_patients()

    by_sub: dict[str, dict[str, int]] = {}
    for p in patients:
        sub = p.get("substance", "Unknown")
        if not sub or sub == "None":
            continue
        sub = sub.strip().title()
        if sub not in by_sub:
            by_sub[sub] = {"total": 0, "bh": 0}
        by_sub[sub]["total"] += 1
        if p["has_bh"]:
            by_sub[sub]["bh"] += 1

    # Sort by total descending, filter to substances with 2+ patients
    items = sorted(
        [(s, v) for s, v in by_sub.items() if v["total"] >= 2],
        key=lambda x: x[1]["total"],
        reverse=True,
    )
    if not items:
        return ""

    substances = [s for s, _ in items]
    pcts = [round(100 * v["bh"] / v["total"], 1) for _, v in items]
    totals = [v["total"] for _, v in items]

    tc = get_theme_colors(theme)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=substances,
            y=pcts,
            marker_color=CHART_COLORS_WARM[: len(substances)],
            text=[f"{p}%" for p in pcts],
            textposition="outside",
            textfont=dict(size=14, color=tc["font_color"]),
            customdata=totals,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "BH prevalence: %{y}%<br>"
                "Total patients: %{customdata}"
                "<extra></extra>"
            ),
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=340,
        margin=dict(t=30, l=50, r=30, b=60),
        y_title="BH Prevalence (%)",
    )
    fig.update_layout(
        yaxis=dict(range=[0, 110], dtick=25, ticksuffix="%"),
        xaxis=dict(showgrid=False),
        bargap=0.35,
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# NEW: Intentional OD Profile
# ---------------------------------------------------------------------------


def build_intentional_od_profile(theme: str = "dark") -> str:
    """Horizontal bar showing key characteristics of intentional OD patients.

    ALL 6 intentional OD patients have BH disorders, and 4 out of 6 have
    suicidal ideation.  This isn't recreational overdose — it's self-harm
    via a pharmacological vector.  The chart makes the clinical reality
    impossible to ignore.
    """
    patients = get_opioid_patients()
    intentional = [p for p in patients if p.get("intentional_od") == "Yes"]

    if not intentional:
        return ""

    total = len(intentional)

    # Calculate key metrics
    has_bh = sum(1 for p in intentional if p["has_bh"])
    has_si = sum(1 for p in intentional if "Suicidal Ideation" in p["bh_conditions"])
    has_chronic = sum(1 for p in intentional if p["has_chronic"])
    has_aud = sum(1 for p in intentional if p.get("aud") == "Yes")
    female = sum(1 for p in intentional if p["sex"] == "Female")

    labels = [
        "Behavioral Health<br>Disorder",
        "Suicidal Ideation",
        "Chronic Illness",
        "Alcohol Use<br>Disorder",
        "Female",
    ]
    values = [has_bh, has_si, has_chronic, has_aud, female]
    pcts = [round(100 * v / total) for v in values]
    colors = [
        TAILWIND_COLORS["violet-500"],
        TAILWIND_COLORS["red-500"],
        TAILWIND_COLORS["amber-500"],
        TAILWIND_COLORS["amber-600"],
        TAILWIND_COLORS["rose-400"],
    ]

    tc = get_theme_colors(theme)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=colors,
            text=[f"{v}/{total} ({p}%)" for v, p in zip(values, pcts, strict=False)],
            textposition="outside",
            textfont=dict(size=13, color=tc["font_color"]),
            hovertemplate="<b>%{y}</b><br>%{x} of %{customdata} patients<extra></extra>",
            customdata=[total] * len(values),
        )
    )

    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=300,
        margin=dict(t=20, l=150, r=100, b=30),
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(
            title=None,
            showticklabels=False,
            showgrid=False,
            range=[0, total + 1],
        ),
        bargap=0.35,
    )
    return _plot_html(fig)


# ---------------------------------------------------------------------------
# Quick stats builders (for template rendering)
# ---------------------------------------------------------------------------


def build_patients_cooccurring_stats() -> list[dict[str, str]]:
    """Summary stat cards for the Patients page co-occurring section."""
    patients = get_opioid_patients()
    total = len(patients)
    bh_count = sum(1 for p in patients if p["has_bh"])
    chronic_count = sum(1 for p in patients if p["has_chronic"])
    both_count = sum(1 for p in patients if p["has_bh"] and p["has_chronic"])
    # Average number of BH conditions per patient who has any
    bh_conds_per = (
        sum(len(p["bh_conditions"]) for p in patients if p["has_bh"]) / bh_count if bh_count else 0
    )

    return [
        {"label": "OUD Cohort", "value": str(total)},
        {"label": "With BH Disorder", "value": f"{round(100 * bh_count / total)}%"},
        {"label": "Chronic Illness", "value": f"{round(100 * chronic_count / total)}%"},
        {"label": "BH + Chronic", "value": f"{round(100 * both_count / total)}%"},
        {"label": "Avg BH Conditions", "value": f"{bh_conds_per:.1f}"},
    ]


def build_odreferrals_cooccurring_stats() -> list[dict[str, str]]:
    """Summary stat cards for the OD Referrals page co-occurring section."""
    patients = get_opioid_patients()
    repeat = [p for p in patients if p["repeat_od_count"] > 0]
    single = [p for p in patients if p["repeat_od_count"] == 0]
    repeat_bh_pct = (
        round(100 * sum(1 for p in repeat if p["has_bh"]) / len(repeat)) if repeat else 0
    )
    single_bh_pct = (
        round(100 * sum(1 for p in single if p["has_bh"]) / len(single)) if single else 0
    )
    intentional = sum(1 for p in patients if p.get("intentional_od") == "Yes")
    aud_count = sum(1 for p in patients if p.get("aud") == "Yes")

    return [
        {"label": "Repeat OD + BH", "value": f"{repeat_bh_pct}%"},
        {"label": "Single OD + BH", "value": f"{single_bh_pct}%"},
        {"label": "BH Gap", "value": f"+{repeat_bh_pct - single_bh_pct}pts"},
        {"label": "Intentional OD", "value": str(intentional)},
        {"label": "OUD + AUD", "value": str(aud_count)},
    ]


def build_referrals_cooccurring_stats() -> list[dict[str, str]]:
    """Summary stat cards for the Referrals page co-occurring section."""
    patients = get_opioid_patients()
    from .data_loader import get_sud_patients

    sud = get_sud_patients()
    aud_count = sum(1 for p in patients if p.get("aud") == "Yes")
    aud_pct = round(100 * aud_count / len(patients)) if patients else 0
    sud_with_bh = sum(1 for p in sud if p["has_bh"])
    sud_bh_pct = round(100 * sud_with_bh / len(sud)) if sud else 0

    return [
        {"label": "OUD + AUD", "value": f"{aud_pct}%"},
        {"label": "AUD Patients", "value": str(aud_count)},
        {"label": "Non-Opioid SUD", "value": str(len(sud))},
        {"label": "SUD + BH", "value": f"{sud_bh_pct}%"},
    ]


def build_deep_dive_hero_stats() -> list[dict[str, str]]:
    """Top-level hero stats for the co-occurring deep dive page."""
    patients = get_opioid_patients()
    from .data_loader import get_sud_patients

    sud = get_sud_patients()
    total_combined = len(patients) + len(sud)
    bh_all = sum(1 for p in patients if p["has_bh"]) + sum(1 for p in sud if p["has_bh"])
    bh_pct = round(100 * bh_all / total_combined) if total_combined else 0
    chronic_count = sum(1 for p in patients if p["has_chronic"])
    triple = sum(1 for p in patients if p["has_bh"] and p["has_chronic"] and p.get("aud") == "Yes")
    none_at_all = sum(
        1 for p in patients if not p["has_bh"] and not p["has_chronic"] and p.get("aud") != "Yes"
    )
    intentional = sum(1 for p in patients if p.get("intentional_od") == "Yes")

    return [
        {"label": "Combined Cohort", "value": str(total_combined)},
        {"label": "Overall BH Rate", "value": f"{bh_pct}%"},
        {
            "label": "Chronic Illness (OUD)",
            "value": f"{round(100 * chronic_count / len(patients))}%",
        },
        {"label": "Triple Burden", "value": str(triple)},
        {"label": "Zero Comorbidities", "value": f"{round(100 * none_at_all / len(patients))}%"},
        {"label": "Intentional OD", "value": str(intentional)},
    ]
