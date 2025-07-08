"""
Shift scenario comparison analytics for optimizing operational coverage
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
from django.db.models import Count
from ...models import ODReferrals
from ...utils.plotly import get_plotly_theme, get_color_palette, style_plotly_layout

def calculate_coverage_scenarios():
    """
    Calculate coverage percentages for selected shift scenarios.
    """

    # Get all overdose data with timestamps
    overdoses = ODReferrals.objects.exclude(od_date__isnull=True)

    # Convert to pandas for easier analysis
    data = []
    for od in overdoses:
        data.append({
            'datetime': od.od_date,
            'hour': od.od_date.hour,
            'weekday': od.od_date.weekday(),  # 0=Monday, 6=Sunday
            'is_weekend': od.od_date.weekday() >= 5
        })

    df = pd.DataFrame(data)
    if df.empty:
        return {}
    total_overdoses = len(df)
    scenarios = {}

    # Current scenario: 8x5 weekdays only
    crew1_current_mask = (df['hour'] >= 8) & (df['hour'] < 16) & (~df['is_weekend'])
    crew2_current_mask = (df['hour'] >= 8) & (df['hour'] < 16) & (~df['is_weekend'])
    dual_crew_current_mask = crew1_current_mask | crew2_current_mask
    scenarios['Current (8x5)'] = {
        'coverage': len(df[dual_crew_current_mask]) / total_overdoses * 100,
        'description': 'Crew 1: Mon-Fri: 08:00-16:00, Crew 2: Mon-Fri: 08:00-16:00',
        'hours_per_week': 80,
        'shifts': 'Crew 1: 5 days × 8 hours, Crew 2: 5 days × 8 hours',
        'color': 'primary'
    }

    # 3x14 Staggered: Crew 1 (Mon–Wed 07:00–21:00), Crew 2 (Thu–Sat 07:00–21:00)
    crew1_3x14_mask = (df['hour'] >= 7) & (df['hour'] < 21) & (df['weekday'] >= 0) & (df['weekday'] <= 2)  # Mon–Wed
    crew2_3x14_mask = (df['hour'] >= 7) & (df['hour'] < 21) & (df['weekday'] >= 3) & (df['weekday'] <= 5)  # Thu–Sat
    dual_crew_3x14_mask = crew1_3x14_mask | crew2_3x14_mask
    scenarios['3×14 Staggered'] = {
        'coverage': len(df[dual_crew_3x14_mask]) / total_overdoses * 100,
        'description': 'Crew 1: Mon–Wed 07:00–21:00, Crew 2: Thu–Sat 07:00–21:00',
        'hours_per_week': 84,  # 2 crews × 3×14h = 2×42
        'shifts': 'Crew 1: 3×14h (Mon–Wed), Crew 2: 3×14h (Thu–Sat)',
        'color': 'warning'
    }

    # 3x12 Staggered: Crew 1 (Mon–Wed 08:00–20:00), Crew 2 (Thu–Sat 08:00–20:00)
    crew1_3x12_mask = (df['hour'] >= 8) & (df['hour'] < 20) & (df['weekday'] >= 0) & (df['weekday'] <= 2)  # Mon–Wed
    crew2_3x12_mask = (df['hour'] >= 8) & (df['hour'] < 20) & (df['weekday'] >= 3) & (df['weekday'] <= 5)  # Thu–Sat
    dual_crew_3x12_mask = crew1_3x12_mask | crew2_3x12_mask
    scenarios['3×12 Staggered'] = {
        'coverage': len(df[dual_crew_3x12_mask]) / total_overdoses * 100,
        'description': 'Crew 1: Mon–Wed 08:00–20:00, Crew 2: Thu–Sat 08:00–20:00',
        'hours_per_week': 72,  # 2 crews × 3×12h = 2×36
        'shifts': 'Crew 1: 3×12h (Mon–Wed), Crew 2: 3×12h (Thu–Sat)',
        'color': 'info'
    }

    # 4x10 Staggered A: Crew 1 (Mon–Thu 07:00–17:00), Crew 2 (Tue–Fri 07:00–17:00)
    crew1_4x10a_mask = (df['hour'] >= 7) & (df['hour'] < 17) & (df['weekday'] >= 0) & (df['weekday'] <= 3)  # Mon–Thu
    crew2_4x10a_mask = (df['hour'] >= 7) & (df['hour'] < 17) & (df['weekday'] >= 1) & (df['weekday'] <= 4)  # Tue–Fri
    dual_crew_4x10a_mask = crew1_4x10a_mask | crew2_4x10a_mask
    scenarios['4×10 Staggered A'] = {
        'coverage': len(df[dual_crew_4x10a_mask]) / total_overdoses * 100,
        'description': 'Crew 1: Mon–Thu 07:00–17:00, Crew 2: Tue–Fri 07:00–17:00',
        'hours_per_week': 80,  # 2 crews × 4×10h
        'shifts': 'Crew 1: 4×10h (Mon–Thu), Crew 2: 4×10h (Tue–Fri)',
        'color': 'success'
    }

    # 4x10 Staggered B: Crew 1 (Mon–Thu 07:00–17:00), Crew 2 (Tue–Fri 09:00–19:00)
    crew1_4x10b_mask = (df['hour'] >= 7) & (df['hour'] < 17) & (df['weekday'] >= 0) & (df['weekday'] <= 3)  # Mon–Thu
    crew2_4x10b_mask = (df['hour'] >= 9) & (df['hour'] < 19) & (df['weekday'] >= 1) & (df['weekday'] <= 4)  # Tue–Fri
    dual_crew_4x10b_mask = crew1_4x10b_mask | crew2_4x10b_mask
    scenarios['4×10 Staggered B'] = {
        'coverage': len(df[dual_crew_4x10b_mask]) / total_overdoses * 100,
        'description': 'Crew 1: Mon–Thu 07:00–17:00, Crew 2: Tue–Fri 09:00–19:00',
        'hours_per_week': 80,  # 2 crews × 4×10h
        'shifts': 'Crew 1: 4×10h (Mon–Thu), Crew 2: 4×10h (Tue–Fri)',
        'color': 'success'
    }

    # 4x10 Staggered C: Crew 1 (Mon–Thu 08:00–18:00), Crew 2 (Tue–Fri 08:00–18:00)
    crew1_4x10c_mask = (df['hour'] >= 8) & (df['hour'] < 18) & (df['weekday'] >= 0) & (df['weekday'] <= 3)  # Mon–Thu
    crew2_4x10c_mask = (df['hour'] >= 8) & (df['hour'] < 18) & (df['weekday'] >= 1) & (df['weekday'] <= 4)  # Tue–Fri
    dual_crew_4x10c_mask = crew1_4x10c_mask | crew2_4x10c_mask
    scenarios['4×10 Staggered C'] = {
        'coverage': len(df[dual_crew_4x10c_mask]) / total_overdoses * 100,
        'description': 'Crew 1: Mon–Thu 08:00–18:00, Crew 2: Tue–Fri 08:00–18:00',
        'hours_per_week': 80,  # 2 crews × 4×10h
        'shifts': 'Crew 1: 4×10h (Mon–Thu), Crew 2: 4×10h (Tue–Fri)',
        'color': 'success'
    }

    # 4x10 Staggered D: Crew 1 (Mon–Thu 09:00–19:00), Crew 2 (Tue–Fri 09:00–19:00)
    crew1_4x10d_mask = (df['hour'] >= 9) & (df['hour'] < 19) & (df['weekday'] >= 0) & (df['weekday'] <= 3)  # Mon–Thu
    crew2_4x10d_mask = (df['hour'] >= 9) & (df['hour'] < 19) & (df['weekday'] >= 1) & (df['weekday'] <= 4)  # Tue–Fri
    dual_crew_4x10d_mask = crew1_4x10d_mask | crew2_4x10d_mask
    scenarios['4×10 Staggered D'] = {
        'coverage': len(df[dual_crew_4x10d_mask]) / total_overdoses * 100,
        'description': 'Crew 1: Mon–Thu 09:00–19:00, Crew 2: Tue–Fri 09:00–19:00',
        'hours_per_week': 80,  # 2 crews × 4×10h
        'shifts': 'Crew 1: 4×10h (Mon–Thu), Crew 2: 4×10h (Tue–Fri)',
        'color': 'success'
    }

    # 4x10 Weekend Overlap: Crew 1 (Sun–Wed 09:00–19:00), Crew 2 (Wed–Sat 09:00–19:00)
    crew1_4x10e_mask = (df['hour'] >= 9) & (df['hour'] < 19) & ((df['weekday'] == 6) | (df['weekday'] <= 2))  # Sun–Wed
    crew2_4x10e_mask = (df['hour'] >= 9) & (df['hour'] < 19) & (df['weekday'] >= 2) & (df['weekday'] <= 5)  # Wed–Sat
    dual_crew_4x10e_mask = crew1_4x10e_mask | crew2_4x10e_mask
    scenarios['4×10 Weekend Overlap'] = {
        'coverage': len(df[dual_crew_4x10e_mask]) / total_overdoses * 100,
        'description': 'Crew 1: Sun–Wed 09:00–19:00, Crew 2: Wed–Sat 09:00–19:00',
        'hours_per_week': 80,  # 2 crews × 4×10h
        'shifts': 'Crew 1: 4×10h (Sun–Wed), Crew 2: 4×10h (Wed–Sat)',
        'color': 'purple'
    }

    # When building scenarios, add a 'short_name' for each scenario
    scenarios['Current (8x5)']['short_name'] = '8x5 Current'
    scenarios['3×14 Staggered']['short_name'] = '3x14 Staggered'
    scenarios['3×12 Staggered']['short_name'] = '3x12 Staggered'
    scenarios['4×10 Staggered A']['short_name'] = '4x10 Staggered A'
    scenarios['4×10 Staggered B']['short_name'] = '4x10 Staggered B'
    scenarios['4×10 Staggered C']['short_name'] = '4x10 Staggered C'
    scenarios['4×10 Staggered D']['short_name'] = '4x10 Staggered D'
    scenarios['4×10 Weekend Overlap']['short_name'] = '4x10 Weekend Overlap'

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
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
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
        chart_config.update({
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": "hover",  # Show modebar only on hover
            "staticPlot": False,  # Ensure interactivity
        })

        return plot(fig, output_type="div", config=chart_config)

    # Get color palette
    colors = get_color_palette(theme)

    # Prepare data for visualization
    scenario_names = list(scenarios.keys())
    short_names = [scenarios[name].get('short_name', name) for name in scenario_names]
    coverages = [scenarios[name]['coverage'] for name in scenario_names]
    hours_per_week = [scenarios[name]['hours_per_week'] for name in scenario_names]
    descriptions = [scenarios[name]['description'] for name in scenario_names]

    # Color mapping
    color_map = {
        'primary': colors['primary'],
        'success': colors['success'],
        'info': colors['info'],
        'warning': colors['warning'],
        'danger': colors['danger'],
        'purple': '#8B5CF6'
    }

    bar_colors = [color_map[scenarios[name]['color']] for name in scenario_names]

    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Coverage Percentage by Scenario', 'Operational Efficiency Analysis'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )

    # Top chart: Coverage percentages
    fig.add_trace(
        go.Bar(
            x=short_names,
            y=coverages,
            name='Coverage %',
            marker_color=bar_colors,
            text=[f'{c:.1f}%' for c in coverages],
            textposition='outside',
            hovertemplate='<b>%{customdata}</b><br>Coverage: %{y:.1f}%<br>%{text}<extra></extra>',
            customdata=scenario_names
        ),
        row=1, col=1
    )

    # Bottom chart: Efficiency as vertical bar chart (coverage efficiency per hour)
    efficiency = [c / h if h > 0 else 0 for c, h in zip(coverages, hours_per_week)]

    # Create hover text manually for better control
    hover_texts = []
    for i, name in enumerate(scenario_names):
        hover_text = f"<b>{name}</b><br>"
        hover_text += f"Efficiency: {efficiency[i]:.3f} coverage % per hour<br>"
        hover_text += f"Coverage: {coverages[i]:.1f}%<br>"
        hover_text += f"Hours/Week: {hours_per_week[i]}"
        hover_texts.append(hover_text)

    fig.add_trace(
        go.Bar(
            x=short_names,
            y=efficiency,
            name='Efficiency',
            marker_color=bar_colors,
            text=[f'{e:.2f}' for e in efficiency],
            textposition='outside',
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_texts
        ),
        row=2, col=1
    )

    # Configure modebar to show only on hover
    fig.update_layout(
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0)',
            color='gray',
            activecolor='black'
        ),
        showlegend=False
    )

    # Update axes
    fig.update_xaxes(title_text="Shift Scenario", row=1, col=1)
    fig.update_xaxes(title_text="Shift Scenario", row=2, col=1)
    fig.update_yaxes(title_text="Coverage Percentage (%)", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (Coverage % per Hour)", row=2, col=1)

    # Make x-axis labels vertical for better mobile readability
    fig.update_xaxes(
        tickangle=90,  # Vertical labels
        tickmode='array',
        tickvals=list(range(len(short_names))),
        ticktext=short_names,  # Use original short names without line breaks
        row=1, col=1
    )

    fig.update_xaxes(
        tickangle=90,  # Vertical labels
        tickmode='array', 
        tickvals=list(range(len(short_names))),
        ticktext=short_names,  # Use original short names without line breaks
        row=2, col=1
    )

    # Update layout using style_plotly_layout    
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=800,
        scroll_zoom=False,
        x_title=None,
        y_title=None,
        margin=dict(t=45, l=10, r=10, b=45),
        hovermode_unified=False,
    )

    chart_config = fig._config.copy()
    chart_config.update({
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",  # Show modebar only on hover
        "staticPlot": False,  # Ensure interactivity
    })

    return plot(fig, output_type="div", config=chart_config)

def build_chart_cost_benefit_analysis(theme):
    """
    Build cost-benefit analysis chart for different scenarios
    """

    scenarios = calculate_coverage_scenarios()

    if not scenarios:
        # Return empty chart if no data with proper styling
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for cost-benefit analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
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
        chart_config.update({
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": "hover",  # Show modebar only on hover
            "staticPlot": False,  # Ensure interactivity
        })

        return plot(fig, output_type="div", config=chart_config)

    # Calculate efficiency metrics
    efficiency_data = []
    for name, data in scenarios.items():
        efficiency = data['coverage'] / data['hours_per_week'] if data['hours_per_week'] > 0 else 0
        efficiency_data.append({
            'scenario': name,
            'coverage': data['coverage'],
            'hours': data['hours_per_week'],
            'efficiency': efficiency,
            'description': data['description']
        })

    # Sort by efficiency descending
    efficiency_data = sorted(efficiency_data, key=lambda x: x['efficiency'], reverse=True)

    # Get color palette
    colors = get_color_palette(theme)

    scenario_names = [d['scenario'] for d in efficiency_data]
    efficiencies = [d['efficiency'] for d in efficiency_data]
    descriptions = [d['description'] for d in efficiency_data]

    bar_colors = [
        colors['primary'] if n == 'Current (8x5)' else
        colors['warning'] if '3×14' in n else
        colors['info'] if '3×12' in n else
        colors['success'] if '4×10' in n and 'Weekend' not in n else
        '#8B5CF6' if 'Weekend' in n else
        colors['primary']
        for n in scenario_names
    ]

    # Create vertical bar chart for efficiency with improved labels
    fig = go.Figure()

    # Create shorter, more readable labels for x-axis
    short_labels = []
    for name in scenario_names:
        if name == 'Current (8x5)':
            short_labels.append('Current\n(8x5)')
        elif name == '3×14 Staggered':
            short_labels.append('3×14\nStaggered')
        elif name == '3×12 Staggered':
            short_labels.append('3×12\nStaggered')
        elif name == '4×10 Staggered A':
            short_labels.append('4×10\nStaggered A')
        elif name == '4×10 Staggered B':
            short_labels.append('4×10\nStaggered B')
        elif name == '4×10 Staggered C':
            short_labels.append('4×10\nStaggered C')
        elif name == '4×10 Staggered D':
            short_labels.append('4×10\nStaggered D')
        elif name == '4×10 Weekend Overlap':
            short_labels.append('4×10\nWeekend')
        else:
            # Fallback to first few words
            short_labels.append(name.split()[0] + '\n' + ' '.join(name.split()[1:3]))

    # Create hover texts with proper crew information
    hover_texts = []
    for i, name in enumerate(scenario_names):
        scenario = scenarios[name]
        short_name = scenario.get('short_name', name)
        hover_text = f"<b>{short_name}</b><br>"
        
        # Extract crew information from description
        description = scenario['description']
        
        # Parse crew schedules based on scenario type
        if 'Crew 1:' in description and 'Crew 2:' in description:
            # Split the description to extract crew schedules
            parts = description.split(', Crew 2:')
            if len(parts) >= 2:
                crew1_info = parts[0].replace('Crew 1: ', '').replace('Dual Crew Staggered: Crew 1 (', '').replace('Staggered: Crew 1 (', '').rstrip(')')
                crew2_info = parts[1].rstrip(')')
                hover_text += f"Crew 1: {crew1_info}<br>"
                hover_text += f"Crew 2: {crew2_info}<br>"
            else:
                # Fallback if split fails
                hover_text += f"Schedule: {description}<br>"
                hover_text += f"<br>"
        elif 'Both crews:' in description:
            # Single schedule for both crews
            schedule = description.replace('Both crews: ', '')
            hover_text += f"Crew 1: {schedule}<br>"
            hover_text += f"Crew 2: {schedule}<br>"
        elif name == 'Current (8x5)':
            # Single crew scenario
            hover_text += f"Single Crew: {description}<br>"
            hover_text += f"<br>"  # Empty line for Crew 2
        elif '3×14 Extended' in name or '3×12' in name:
            # Handle extended scenarios with staggered crews
            if 'Staggered:' in description:
                parts = description.replace('Staggered: ', '').split(', Crew 2 (')
                if len(parts) >= 2:
                    crew1_info = parts[0].replace('Crew 1 (', '').rstrip(')')
                    crew2_info = parts[1].rstrip(')')
                    hover_text += f"Crew 1: {crew1_info}<br>"
                    hover_text += f"Crew 2: {crew2_info}<br>"
                else:
                    # Fallback if split fails
                    hover_text += f"Schedule: {description}<br>"
                    hover_text += f"<br>"
            else:
                # Single extended crew
                hover_text += f"Extended Crew: {description}<br>"
                hover_text += f"<br>"  # Empty line for Crew 2
        else:
            # Fallback for other scenarios
            hover_text += f"Schedule: {description}<br>"
            hover_text += f"<br>"  # Empty line for consistency
        
        hover_text += f"Efficiency: {efficiencies[i]:.3f} coverage % per hour"
        hover_texts.append(hover_text)

    fig.add_trace(
        go.Bar(
            x=short_labels,  # Use shorter labels for x-axis
            y=efficiencies,
            marker_color=bar_colors,
            text=[f'{e:.2f}' for e in efficiencies],
            textposition='outside',
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_texts
        )
    )

    # Set x-axis labels to be vertical for mobile readability
    fig.update_xaxes(
        tickangle=90,  # Vertical labels for better mobile readability
        tickmode='array',
        tickvals=list(range(len(short_labels))),
        ticktext=short_labels,
        title=None,  # Remove x-axis title
        tickfont=dict(size=14, family="Roboto"),  # Consistent font size and family
        ticklabelstandoff=10,  # Set specific standoff distance
        showgrid=False,  # Remove vertical grid lines
        showline=True,  # Show the x-axis line
        linewidth=1,  # Set line width
        linecolor='lightgray',  # Set line color
        zeroline=True  # No zero line needed for this chart
    )

    # Set y-axis label standoff and font
    fig.update_yaxes(
        title=None,  # Remove y-axis title
        tickfont=dict(size=14, family="Roboto"),  # Consistent font size and family
        ticklabelstandoff=10,  # Set specific standoff distance
        showgrid=True,  # Keep horizontal grid lines
        range=[0, 0.8],  # Set y-axis range from 0 to 0.8
        dtick=0.1,  # Show tick marks every 0.1
    )

    # Update layout using style_plotly_layout    
    fig = style_plotly_layout(
        fig,
        theme=theme,
        height=640,
        scroll_zoom=False,
        x_title=None,
        y_title=None,
        margin=dict(t=45, l=45, r=40, b=45),
        hovermode_unified=False,
    )

    # Explicitly remove vertical grid lines after styling
    fig.update_layout(
        xaxis=dict(showgrid=False),  # Remove vertical grid lines
        yaxis=dict(showgrid=True)    # Keep horizontal grid lines
    )

    chart_config = fig._config.copy()
    chart_config.update({
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "hover",  # Show modebar only on hover
        "staticPlot": False,  # Ensure interactivity
    })

    return plot(fig, output_type="div", config=chart_config)