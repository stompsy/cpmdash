import pandas as pd
import plotly.express as px
from plotly.offline import plot

from ...utils.plotly import style_plotly_layout
from dashboard.models import ODReferrals


# Dataframe
odreferrals = ODReferrals.objects.all()
df = pd.DataFrame.from_records(
    odreferrals.values(
        "od_date",
    )
)

# Ensure 'od_date' is parsed as datetime
df["od_date"] = pd.to_datetime(df["od_date"], errors="coerce")

# Extract hour
df["hour"] = df["od_date"].dt.hour

# Define working hours (08:00 to 16:59)
df["during_work_hours"] = df["hour"].between(8, 15)

# Count overdoses during and outside working hours
work_hours_count = (
    df["during_work_hours"]
    .value_counts()
    .rename(index={True: "During Work Hours", False: "Outside Work Hours"})
)

# Determine which hour has the most overdoses
hourly_counts = df["hour"].value_counts().sort_values(ascending=False)
work_hours_count, hourly_counts.head(1)

# Prepare data for work vs. non-work hours comparison
work_hour_df = work_hours_count.reset_index()
work_hour_df.columns = ["Time Category", "Overdose Count"]

hourly_df = df["hour"].value_counts().sort_index().reset_index()
hourly_df.columns = ["Hour of Day", "Overdose Count"]

# Bar chart: Working hours vs Outside working hours
def build_chart_od_work_hours(theme):
    fig = px.bar(
        work_hour_df,
        x="Time Category",
        y="Overdose Count",
        color="Time Category",
        text="Overdose Count",
        color_discrete_map={
            "During Work Hours": "#1f77b4",
            "Outside Work Hours": "#ff7f0e",
        },
    )
    fig = style_plotly_layout(
        fig,
        theme=theme,
        export_filename="pafd_cpm_chart_work_hours",
        scroll_zoom=False,
        y_title="Overdose Count",
        margin=dict(t=0, l=75, r=20, b=65),
    )
    return plot(fig, output_type="div", config=fig._config)

