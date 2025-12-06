"""Normalization helpers shared by dashboard chart builders."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def add_share_columns(
    df: pd.DataFrame,
    count_col: str,
    *,
    share_col: str = "share_pct",
    label_col: str | None = "share_label",
    precision: int = 1,
) -> pd.DataFrame:
    """Add percentage-of-total columns to ``df``.

    ``df`` is mutated and returned for convenience. When the denominator is zero the share
    defaults to ``0.0``.
    """

    if count_col not in df.columns:
        df[share_col] = 0.0
        if label_col:
            df[label_col] = ["0.0%"] * len(df)
        return df

    # Ensure count column is numeric to prevent errors
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)

    total = float(df[count_col].sum())
    if total <= 0:
        df[share_col] = 0.0
    else:
        df[share_col] = (df[count_col] / total) * 100.0

    # Ensure share column is clean
    df[share_col] = df[share_col].fillna(0.0)

    if label_col:
        df[label_col] = df[share_col].apply(lambda value: f"{value:.{precision}f}%")
    return df


def count_share_text(count: float, share_pct: float, *, precision: int = 1) -> str:
    """Return a compact label like ``"34 • 12.5%"`` for chart annotations."""

    try:
        count_value: int | float = int(count)
    except Exception:
        count_value = count
    return f"{count_value} • {share_pct:.{precision}f}%"


def rolling_average(values: Iterable[int | float], *, window: int = 3) -> list[float]:
    """Compute a simple rolling average with ``min_periods=1`` semantics."""

    series = pd.Series(list(values), dtype="float")
    if series.empty:
        return []
    return series.rolling(window=window, min_periods=1).mean().tolist()
