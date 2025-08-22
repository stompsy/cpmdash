"""
Shift scenario comparison analytics for optimizing operational coverage
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from utils.plotly import get_color_palette, style_plotly_layout

from ...core.models import ODReferrals


def calculate_coverage_scenarios():  # noqa: C901
    """
    Calculate coverage percentages for selected shift scenarios.
    Properly handles annualization and crew overlap.
    """

    # Get all overdose data with timestamps
    overdoses = ODReferrals.objects.exclude(od_date__isnull=True)

    # Convert to pandas for easier analysis
    data = []
    for od in overdoses:
        data.append(
            {
                "datetime": od.od_date,
                "hour": od.od_date.hour,
                "weekday": od.od_date.weekday(),  # 0=Monday, 6=Sunday
                "is_weekend": od.od_date.weekday() >= 5,
            }
        )

    df = pd.DataFrame(data)
    if df.empty:
        return {}

    # Calculate total time span for annualization
    min_date = df["datetime"].min()
    max_date = df["datetime"].max()
    total_days = (max_date - min_date).days + 1
    total_years = total_days / 365.25

    total_overdoses = len(df)

    scenarios = {}

    def calculate_covered_hours_per_week(crew1_hours, crew2_hours):
        """
        Calculate total covered hours per week, avoiding double-counting overlaps.
        crew1_hours and crew2_hours are lists of (day, start_hour, end_hour) tuples.
        """
        # Create a set to track all covered hour slots
        covered_slots = set()

        # Add crew 1 hours
        for day, start, end in crew1_hours:
            for hour in range(start, end):
                covered_slots.add((day, hour))

        # Add crew 2 hours (overlaps are automatically handled by set)
        for day, start, end in crew2_hours:
            for hour in range(start, end):
                covered_slots.add((day, hour))

        return len(covered_slots)

    def get_scenario_mask_and_hours(crew1_schedule, crew2_schedule):
        """
        Get both the coverage mask for cases and actual covered hours.
        """
        # Calculate crew masks for cases
        crew1_mask = pd.Series(False, index=df.index)
        crew2_mask = pd.Series(False, index=df.index)

        for day, start_hour, end_hour in crew1_schedule:
            if day == 6:  # Sunday
                mask = (df["hour"] >= start_hour) & (df["hour"] < end_hour) & (df["weekday"] == 6)
            else:
                mask = (df["hour"] >= start_hour) & (df["hour"] < end_hour) & (df["weekday"] == day)
            crew1_mask |= mask

        for day, start_hour, end_hour in crew2_schedule:
            if day == 6:  # Sunday
                mask = (df["hour"] >= start_hour) & (df["hour"] < end_hour) & (df["weekday"] == 6)
            else:
                mask = (df["hour"] >= start_hour) & (df["hour"] < end_hour) & (df["weekday"] == day)
            crew2_mask |= mask

        # Combined mask (any crew covering)
        combined_mask = crew1_mask | crew2_mask

        # Calculate covered hours per week
        covered_hours = calculate_covered_hours_per_week(crew1_schedule, crew2_schedule)

        return combined_mask, covered_hours

    # Current scenario: 5x8 weekdays only (single crew working Mon-Fri 08:00-16:00)
    crew1_current = [(0, 8, 16), (1, 8, 16), (2, 8, 16), (3, 8, 16), (4, 8, 16)]  # Mon-Fri
    crew2_current = []  # No second crew in current scenario
    current_mask, current_hours = get_scenario_mask_and_hours(crew1_current, crew2_current)
    covered_cases = len(df[current_mask])

    scenarios["5x8 (Current)"] = {
        "coverage": (covered_cases / total_overdoses) * 100,
        "cases_per_year": (covered_cases / total_years),
        "description": "Single Crew: Mon-Fri 08:00-16:00",
        "hours_per_week": current_hours,
        "shifts": "Single crew: 5 days × 8 hours",
        "color": "primary",
    }

    # 3x14 Staggered: Crew 1 (Mon–Wed 07:00–21:00), Crew 2 (Thu–Sat 07:00–21:00)
    crew1_3x14 = [(0, 7, 21), (1, 7, 21), (2, 7, 21)]  # Mon-Wed
    crew2_3x14 = [(3, 7, 21), (4, 7, 21), (5, 7, 21)]  # Thu-Sat
    staggered_3x14_mask, staggered_3x14_hours = get_scenario_mask_and_hours(crew1_3x14, crew2_3x14)
    covered_cases = len(df[staggered_3x14_mask])
    scenarios["3×14 Staggered"] = {
        "coverage": (covered_cases / total_overdoses) * 100,
        "cases_per_year": (covered_cases / total_years),
        "description": "Crew 1: Mon–Wed 07:00–21:00, Crew 2: Thu–Sat 07:00–21:00",
        "hours_per_week": staggered_3x14_hours,
        "shifts": "Crew 1: 3×14h (Mon–Wed), Crew 2: 3×14h (Thu–Sat)",
        "color": "warning",
    }

    # 3x12 Staggered: Crew 1 (Mon–Wed 08:00–20:00), Crew 2 (Thu–Sat 08:00–20:00)
    crew1_3x12 = [(0, 8, 20), (1, 8, 20), (2, 8, 20)]  # Mon-Wed
    crew2_3x12 = [(3, 8, 20), (4, 8, 20), (5, 8, 20)]  # Thu-Sat
    staggered_3x12_mask, staggered_3x12_hours = get_scenario_mask_and_hours(crew1_3x12, crew2_3x12)
    covered_cases = len(df[staggered_3x12_mask])
    scenarios["3×12 Staggered"] = {
        "coverage": (covered_cases / total_overdoses) * 100,
        "cases_per_year": (covered_cases / total_years),
        "description": "Crew 1: Mon–Wed 08:00–20:00, Crew 2: Thu–Sat 08:00–20:00",
        "hours_per_week": staggered_3x12_hours,
        "shifts": "Crew 1: 3×12h (Mon–Wed), Crew 2: 3×12h (Thu–Sat)",
        "color": "info",
    }

    # 4x10 Staggered A: Crew 1 (Mon–Thu 07:00–17:00), Crew 2 (Tue–Fri 07:00–17:00)
    crew1_4x10a = [(0, 7, 17), (1, 7, 17), (2, 7, 17), (3, 7, 17)]  # Mon-Thu
    crew2_4x10a = [(1, 7, 17), (2, 7, 17), (3, 7, 17), (4, 7, 17)]  # Tue-Fri
    staggered_4x10a_mask, staggered_4x10a_hours = get_scenario_mask_and_hours(
        crew1_4x10a, crew2_4x10a
    )
    covered_cases = len(df[staggered_4x10a_mask])
    scenarios["4×10 Staggered A"] = {
        "coverage": (covered_cases / total_overdoses) * 100,
        "cases_per_year": (covered_cases / total_years),
        "description": "Crew 1: Mon–Thu 07:00–17:00, Crew 2: Tue–Fri 07:00–17:00",
        "hours_per_week": staggered_4x10a_hours,
        "shifts": "Crew 1: 4×10h (Mon–Thu), Crew 2: 4×10h (Tue–Fri)",
        "color": "success",
    }

    # 4x10 Staggered B: Crew 1 (Mon–Thu 07:00–17:00), Crew 2 (Tue–Fri 09:00–19:00)
    crew1_4x10b = [(0, 7, 17), (1, 7, 17), (2, 7, 17), (3, 7, 17)]  # Mon-Thu
    crew2_4x10b = [(1, 9, 19), (2, 9, 19), (3, 9, 19), (4, 9, 19)]  # Tue-Fri
    staggered_4x10b_mask, staggered_4x10b_hours = get_scenario_mask_and_hours(
        crew1_4x10b, crew2_4x10b
    )
    covered_cases = len(df[staggered_4x10b_mask])
    scenarios["4×10 Staggered B"] = {
        "coverage": (covered_cases / total_overdoses) * 100,
        "cases_per_year": (covered_cases / total_years),
        "description": "Crew 1: Mon–Thu 07:00–17:00, Crew 2: Tue–Fri 09:00–19:00",
        "hours_per_week": staggered_4x10b_hours,
        "shifts": "Crew 1: 4×10h (Mon–Thu), Crew 2: 4×10h (Tue–Fri)",
        "color": "success",
    }

    # 4x10 Staggered C: Crew 1 (Mon–Thu 08:00–18:00), Crew 2 (Tue–Fri 08:00–18:00)
    crew1_4x10c = [(0, 8, 18), (1, 8, 18), (2, 8, 18), (3, 8, 18)]  # Mon-Thu
    crew2_4x10c = [(1, 8, 18), (2, 8, 18), (3, 8, 18), (4, 8, 18)]  # Tue-Fri
    staggered_4x10c_mask, staggered_4x10c_hours = get_scenario_mask_and_hours(
        crew1_4x10c, crew2_4x10c
    )
    covered_cases = len(df[staggered_4x10c_mask])
    scenarios["4×10 Staggered C"] = {
        "coverage": (covered_cases / total_overdoses) * 100,
        "cases_per_year": (covered_cases / total_years),
        "description": "Crew 1: Mon–Thu 08:00–18:00, Crew 2: Tue–Fri 08:00–18:00",
        "hours_per_week": staggered_4x10c_hours,
        "shifts": "Crew 1: 4×10h (Mon–Thu), Crew 2: 4×10h (Tue–Fri)",
        "color": "success",
    }

    # 4x10 Staggered D: Crew 1 (Mon–Thu 09:00–19:00), Crew 2 (Tue–Fri 09:00–19:00)
    crew1_4x10d = [(0, 9, 19), (1, 9, 19), (2, 9, 19), (3, 9, 19)]  # Mon-Thu
    crew2_4x10d = [(1, 9, 19), (2, 9, 19), (3, 9, 19), (4, 9, 19)]  # Tue-Fri
    staggered_4x10d_mask, staggered_4x10d_hours = get_scenario_mask_and_hours(
        crew1_4x10d, crew2_4x10d
    )
    covered_cases = len(df[staggered_4x10d_mask])
    scenarios["4×10 Staggered D"] = {
        "coverage": (covered_cases / total_overdoses) * 100,
        "cases_per_year": (covered_cases / total_years),
        "description": "Crew 1: Mon–Thu 09:00–19:00, Crew 2: Tue–Fri 09:00–19:00",
        "hours_per_week": staggered_4x10d_hours,
        "shifts": "Crew 1: 4×10h (Mon–Thu), Crew 2: 4×10h (Tue–Fri)",
        "color": "success",
    }

    # 4x10 Weekend Overlap: Crew 1 (Sun–Wed 09:00–19:00), Crew 2 (Wed–Sat 09:00–19:00)
    crew1_4x10e = [(6, 9, 19), (0, 9, 19), (1, 9, 19), (2, 9, 19)]  # Sun-Wed
    crew2_4x10e = [(2, 9, 19), (3, 9, 19), (4, 9, 19), (5, 9, 19)]  # Wed-Sat
    weekend_4x10_mask, weekend_4x10_hours = get_scenario_mask_and_hours(crew1_4x10e, crew2_4x10e)
    covered_cases = len(df[weekend_4x10_mask])
    scenarios["4×10 Weekend Overlap"] = {
        "coverage": (covered_cases / total_overdoses) * 100,
        "cases_per_year": (covered_cases / total_years),
        "description": "Crew 1: Sun–Wed 09:00–19:00, Crew 2: Wed–Sat 09:00–19:00",
        "hours_per_week": weekend_4x10_hours,
        "shifts": "Crew 1: 4×10h (Sun–Wed), Crew 2: 4×10h (Wed–Sat)",
        "color": "purple",
    }

    # When building scenarios, add a 'short_name' for each scenario
    scenarios["5x8 (Current)"]["short_name"] = "5x8 (Current)"
    scenarios["3×14 Staggered"]["short_name"] = "3x14 Staggered"
    scenarios["3×12 Staggered"]["short_name"] = "3x12 Staggered"
    scenarios["4×10 Staggered A"]["short_name"] = "4x10 Staggered A"
    scenarios["4×10 Staggered B"]["short_name"] = "4x10 Staggered B"
    scenarios["4×10 Staggered C"]["short_name"] = "4x10 Staggered C"
    scenarios["4×10 Staggered D"]["short_name"] = "4x10 Staggered D"
    scenarios["4×10 Weekend Overlap"]["short_name"] = "4x10 Weekend Overlap"

    return scenarios


def build_chart_shift_scenarios(theme):
    """
    Build shift scenario comparison chart
    """

    scenarios = calculate_coverage_scenarios()

    if not scenarios:
        # Return empty chart if no data with proper styling
        fig = go.Figure()
        fig.add_annotation(
            text="No overdose data available for scenario analysis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig = style_plotly_layout(
            fig,
            theme=theme,
            scroll_zoom=False,
            x_title=None,
            y_title=None,
            margin=dict(t=45, l=10, r=10, b=45),
            hovermode_unified=False,
        )

        chart_config = fig._config.copy()
        chart_config.update(
            {
                "responsive": True,
                "displaylogo": False,
                "displayModeBar": "hover",  # Show modebar only on hover
                "staticPlot": False,  # Ensure interactivity
            }
        )

        return plot(fig, output_type="div", config=chart_config)

    # Get color palette
    colors = get_color_palette(theme)

    # Prepare data for visualization
    scenario_names = list(scenarios.keys())
    short_names = [scenarios[name].get("short_name", name) for name in scenario_names]
    coverages = [scenarios[name]["coverage"] for name in scenario_names]
    cases_per_year = [scenarios[name]["cases_per_year"] for name in scenario_names]
    hours_per_week = [scenarios[name]["hours_per_week"] for name in scenario_names]

    # Color mapping
    color_map = {
        "primary": colors["primary"],
        "success": colors["success"],
        "info": colors["info"],
        "warning": colors["warning"],
        "danger": colors["danger"],
        "purple": "#8B5CF6",
    }

    bar_colors = [color_map[scenarios[name]["color"]] for name in scenario_names]

    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Coverage Percentage by Scenario", "Operational Efficiency Analysis"),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5],
    )

    # Top chart: Coverage percentages
    fig.add_trace(
        go.Bar(
            x=short_names,
            y=coverages,
            name="Coverage %",
            marker_color=bar_colors,
            text=[f"{c:.1f}%" for c in coverages],
            textposition="outside",
            hovertemplate="<b>%{customdata}</b><br>Coverage: %{y:.1f}%<br>Cases/Year: %{hovertext}<extra></extra>",
            customdata=scenario_names,
            hovertext=[f"{cpy:.1f}" for cpy in cases_per_year],
        ),
        row=1,
        col=1,
    )

    # Bottom chart: Efficiency as vertical bar chart (coverage efficiency per hour)
    efficiency = [c / h if h > 0 else 0 for c, h in zip(coverages, hours_per_week, strict=False)]

    # Create hover text manually for better control
    hover_texts = []
    for i, name in enumerate(scenario_names):
        hover_text = f"<b>{name}</b><br>"
        hover_text += f"Efficiency: {efficiency[i]:.3f} coverage % per hour<br>"
        hover_text += f"Coverage: {coverages[i]:.1f}%<br>"
        hover_text += f"Cases/Year: {cases_per_year[i]:.1f}<br>"
        hover_text += f"Hours/Week: {hours_per_week[i]}"
        hover_texts.append(hover_text)

    fig.add_trace(
        go.Bar(
            x=short_names,
            y=efficiency,
            name="Efficiency",
            marker_color=bar_colors,
            text=[f"{e:.2f}" for e in efficiency],
            textposition="outside",
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_texts,
        ),
        row=2,
        col=1,
    )

    # Configure modebar to show only on hover
    fig.update_layout(
        modebar=dict(
            orientation="v", bgcolor="rgba(255,255,255,0)", color="gray", activecolor="black"
        ),
        showlegend=False,
    )

    # Update axes
    fig.update_xaxes(title_text="Shift Scenario", row=1, col=1)
    fig.update_xaxes(title_text="Shift Scenario", row=2, col=1)
    fig.update_yaxes(title_text="Coverage Percentage (%)", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (Coverage % per Hour)", row=2, col=1)

    # Make x-axis labels vertical for better mobile readability
    fig.update_xaxes(
        tickangle=90,  # Vertical labels
        tickmode="array",
        tickvals=list(range(len(short_names))),
        ticktext=short_names,  # Use original short names without line breaks
        row=1,
        col=1,
    )

    fig.update_xaxes(
        tickangle=90,  # Vertical labels
        tickmode="array",
        tickvals=list(range(len(short_names))),
        ticktext=short_names,  # Use original short names without line breaks
        row=2,
        col=1,
    )

    # Update layout using style_plotly_layout
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=500,
        scroll_zoom=False,
        x_title=None,
        y_title=None,
        margin=dict(t=45, l=10, r=10, b=45),
        hovermode_unified=False,
    )

    chart_config = fig._config.copy()
    chart_config.update(
        {
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": "hover",  # Show modebar only on hover
            "staticPlot": False,  # Ensure interactivity
        }
    )

    return plot(fig, output_type="div", config=chart_config)


def build_chart_cost_benefit_analysis(theme):  # noqa: C901
    """
    Build cost-benefit analysis chart for different scenarios
    """

    scenarios = calculate_coverage_scenarios()

    if not scenarios:
        # Return empty chart if no data with proper styling
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for cost-benefit analysis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        # Update layout using style_plotly_layout
        fig = style_plotly_layout(
            fig,
            theme=theme,
            height=500,
            scroll_zoom=False,
            x_title=None,
            y_title=None,
            margin=dict(t=45, l=45, r=40, b=45),
            hovermode_unified=False,
        )

        chart_config = fig._config.copy()
        chart_config.update(
            {
                "responsive": True,
                "displaylogo": False,
                "displayModeBar": "hover",  # Show modebar only on hover
                "staticPlot": False,  # Ensure interactivity
            }
        )

        return plot(fig, output_type="div", config=chart_config)

    # Calculate efficiency metrics
    efficiency_data = []
    for name, data in scenarios.items():
        efficiency = data["coverage"] / data["hours_per_week"] if data["hours_per_week"] > 0 else 0
        efficiency_data.append({"scenario": name, "efficiency": efficiency})

    # Sort by efficiency descending
    efficiency_data = sorted(efficiency_data, key=lambda x: x["efficiency"], reverse=True)

    # Get color palette
    colors = get_color_palette(theme)

    scenario_names = [d["scenario"] for d in efficiency_data]
    efficiencies = [d["efficiency"] for d in efficiency_data]

    bar_colors = [
        colors["primary"]
        if n == "5x8 (Current)"
        else "#9CA3AF"
        if "3×14" in n
        # Grey for legally restricted 14-hour shifts
        else "#9CA3AF"
        if "3×12" in n
        # Grey for legally restricted 12-hour shifts
        else colors["success"]
        if "4×10" in n and "Weekend" not in n
        else "#8B5CF6"
        if "Weekend" in n
        else colors["primary"]
        for n in scenario_names
    ]

    # Create vertical bar chart for efficiency with improved labels
    fig = go.Figure()

    # Create shorter, more readable labels for x-axis
    short_labels = []
    for name in scenario_names:
        if name == "5x8 (Current)":
            short_labels.append("5x8\n(Current)")
        elif name == "3×14 Staggered":
            short_labels.append("3×14\nStaggered")
        elif name == "3×12 Staggered":
            short_labels.append("3×12\nStaggered")
        elif name == "4×10 Staggered A":
            short_labels.append("4×10\nStaggered A")
        elif name == "4×10 Staggered B":
            short_labels.append("4×10\nStaggered B")
        elif name == "4×10 Staggered C":
            short_labels.append("4×10\nStaggered C")
        elif name == "4×10 Staggered D":
            short_labels.append("4×10\nStaggered D")
        elif name == "4×10 Weekend Overlap":
            short_labels.append("4×10\nWeekend")
        else:
            # Fallback to first few words
            short_labels.append(name.split()[0] + "\n" + " ".join(name.split()[1:3]))

    # Create hover texts with scenario information
    hover_texts = []
    for i, name in enumerate(scenario_names):
        scenario = scenarios[name]
        short_name = scenario.get("short_name", name)
        hover_text = f"<b>{short_name}</b><br>"
        hover_text += f"Schedule: {scenario['description']}<br>"
        hover_text += f"Efficiency: {efficiencies[i]:.3f} coverage % per hour<br>"
        hover_text += f"Cases/Year: {scenario['cases_per_year']:.1f}"
        hover_texts.append(hover_text)

    fig.add_trace(
        go.Bar(
            x=short_labels,  # Use shorter labels for x-axis
            y=efficiencies,
            marker_color=bar_colors,
            text=[f"{e:.2f}" for e in efficiencies],
            textposition="outside",
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_texts,
        )
    )

    # Set x-axis labels to be vertical for mobile readability
    fig.update_xaxes(
        tickangle=90,  # Vertical labels for better mobile readability
        tickmode="array",
        tickvals=list(range(len(short_labels))),
        ticktext=short_labels,
        title=None,  # Remove x-axis title
        tickfont=dict(size=14, family="Roboto"),  # Consistent font size and family
        ticklabelstandoff=10,  # Set specific standoff distance
        showgrid=False,  # Remove vertical grid lines
        showline=True,  # Show the x-axis line
        linewidth=1,  # Set line width
        linecolor="lightgray",  # Set line color
        zeroline=True,  # No zero line needed for this chart
    )

    # Set y-axis label standoff and font
    fig.update_yaxes(
        title=None,  # Remove y-axis title
        tickfont=dict(size=14, family="Roboto"),  # Consistent font size and family
        ticklabelstandoff=10,  # Set specific standoff distance
        showgrid=True,  # Keep horizontal grid lines
        range=[0, 1],  # Set y-axis range from 0 to 0.8
        dtick=0.1,  # Show tick marks every 0.1
    )

    # Update layout using style_plotly_layout
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=400,
        scroll_zoom=False,
        x_title=None,
        y_title=None,
        margin=dict(t=45, l=45, r=40, b=45),
        hovermode_unified=False,
    )

    # Explicitly remove vertical grid lines after styling
    fig.update_layout(
        xaxis=dict(showgrid=False),  # Remove vertical grid lines
        yaxis=dict(showgrid=True),  # Keep horizontal grid lines
    )

    chart_config = fig._config.copy()
    chart_config.update(
        {
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": "hover",  # Show modebar only on hover
            "staticPlot": False,  # Ensure interactivity
        }
    )

    return plot(fig, output_type="div", config=chart_config)
