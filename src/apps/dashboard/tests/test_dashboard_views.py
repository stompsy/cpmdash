from __future__ import annotations

import warnings

import pandas as pd

from apps.dashboard import views as dashboard_views


def test_patients_story_impact_section_mixed_sud_values_handles_mask_without_future_warning():
    df_patients = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "age": [50, 44, 39, 67, 61, 73, 28],
            "sud": [None, 0, 1, True, False, "1", "0"],
        }
    )
    df_enc = pd.DataFrame(
        {
            "patient_ID": [3, 3, 6],
            "encounter_date": pd.to_datetime(["2026-01-01", "2026-01-10", "2026-01-04"]),
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        section = dashboard_views._patients_story_impact_section(
            df_patients, df_enc, total_patients=7
        )

    future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert not future_warnings

    metric_by_label = {metric["label"]: metric["value"] for metric in section["metrics"]}
    assert metric_by_label["SUD cohort engaged"] == "66.7%"
