"""
Professional, engaging color palettes for charts.
Designed to contrast well with the dark slate website theme.
"""

from .tailwind_colors import TAILWIND_COLORS

# ==============================================================================
# PRIMARY CHART COLOR SEQUENCES
# ==============================================================================

# Vibrant professional palette - great for categorical data
CHART_COLORS_VIBRANT = [
    TAILWIND_COLORS["violet-500"],  # #8b5cf6 - Rich violet
    TAILWIND_COLORS["cyan-500"],  # #06b6d4 - Bright cyan
    TAILWIND_COLORS["rose-500"],  # #f43f5e - Vibrant rose
    TAILWIND_COLORS["emerald-500"],  # #10b981 - Fresh emerald
    TAILWIND_COLORS["amber-500"],  # #f59e0b - Warm amber
    TAILWIND_COLORS["blue-500"],  # #3b82f6 - Classic blue
    TAILWIND_COLORS["pink-500"],  # #ec4899 - Energetic pink
    TAILWIND_COLORS["teal-500"],  # #14b8a6 - Professional teal
    TAILWIND_COLORS["orange-500"],  # #f97316 - Bold orange
    TAILWIND_COLORS["indigo-500"],  # #6366f1 - Deep indigo
    TAILWIND_COLORS["lime-500"],  # #84cc16 - Fresh lime
    TAILWIND_COLORS["fuchsia-500"],  # #d946ef - Striking fuchsia
]

# Cool professional palette - calm and trustworthy
CHART_COLORS_COOL = [
    TAILWIND_COLORS["blue-600"],  # #2563eb - Trust blue
    TAILWIND_COLORS["teal-600"],  # #0d9488 - Medical teal
    TAILWIND_COLORS["cyan-600"],  # #0891b2 - Clear cyan
    TAILWIND_COLORS["indigo-600"],  # #4f46e5 - Professional indigo
    TAILWIND_COLORS["violet-600"],  # #7c3aed - Calm violet
    TAILWIND_COLORS["sky-600"],  # #0284c7 - Bright sky
    TAILWIND_COLORS["blue-500"],  # #3b82f6 - Standard blue
    TAILWIND_COLORS["purple-600"],  # #9333ea - Deep purple
]

# Warm professional palette - energetic and approachable
CHART_COLORS_WARM = [
    TAILWIND_COLORS["orange-600"],  # #ea580c - Energetic orange
    TAILWIND_COLORS["amber-600"],  # #d97706 - Rich amber
    TAILWIND_COLORS["rose-600"],  # #e11d48 - Strong rose
    TAILWIND_COLORS["red-600"],  # #dc2626 - Bold red
    TAILWIND_COLORS["pink-600"],  # #db2777 - Vibrant pink
    TAILWIND_COLORS["yellow-600"],  # #ca8a04 - Warm yellow
    TAILWIND_COLORS["orange-500"],  # #f97316 - Bright orange
    TAILWIND_COLORS["amber-500"],  # #f59e0b - Sunny amber
]

# ==============================================================================
# GENDER/DEMOGRAPHIC COLORS
# ==============================================================================

GENDER_COLORS = {
    "male": TAILWIND_COLORS[
        "sky-500"
    ],  # #0ea5e9 - Bright sky blue (more modern than traditional blue)
    "female": TAILWIND_COLORS["rose-500"],  # #f43f5e - Vibrant rose (more sophisticated than pink)
    "other": TAILWIND_COLORS["violet-500"],  # #8b5cf6 - Inclusive violet
    "unknown": TAILWIND_COLORS["slate-400"],  # #94a3b8 - Neutral gray
}

# ==============================================================================
# SEMANTIC COLORS (Status, Priority, Risk)
# ==============================================================================

# Risk/Vulnerability levels
RISK_COLORS = {
    "critical": TAILWIND_COLORS["red-600"],  # #dc2626 - High alert
    "high": TAILWIND_COLORS["orange-600"],  # #ea580c - Elevated concern
    "moderate": TAILWIND_COLORS["amber-500"],  # #f59e0b - Moderate attention
    "low": TAILWIND_COLORS["emerald-500"],  # #10b981 - Safe/stable
    "minimal": TAILWIND_COLORS["sky-400"],  # #38bdf8 - Very low risk
}

# Success/Warning/Error states
STATUS_COLORS = {
    "success": TAILWIND_COLORS["emerald-600"],  # #059669 - Completed/good
    "warning": TAILWIND_COLORS["amber-600"],  # #d97706 - Needs attention
    "error": TAILWIND_COLORS["red-600"],  # #dc2626 - Failed/critical
    "info": TAILWIND_COLORS["cyan-600"],  # #0891b2 - Informational
    "neutral": TAILWIND_COLORS["slate-500"],  # #64748b - Inactive/unknown
}

# ==============================================================================
# AGE-BASED VULNERABILITY COLORS
# ==============================================================================

# Refined age vulnerability palette with better contrast
AGE_VULNERABILITY_COLORS = {
    "0–17": TAILWIND_COLORS["rose-600"],  # #e11d48 - Children - high vulnerability
    "18–24": TAILWIND_COLORS["sky-500"],  # #0ea5e9 - Young adults - lower risk
    "25–34": TAILWIND_COLORS["cyan-500"],  # #06b6d4 - Adults - lower risk
    "35–44": TAILWIND_COLORS["teal-500"],  # #14b8a6 - Adults - stable
    "45–54": TAILWIND_COLORS["emerald-500"],  # #10b981 - Middle age - stable
    "55–64": TAILWIND_COLORS["amber-500"],  # #f59e0b - Pre-senior - moderate
    "65–74": TAILWIND_COLORS["orange-600"],  # #ea580c - Young senior - elevated
    "75–84": TAILWIND_COLORS["red-600"],  # #dc2626 - Senior - high
    "85+": TAILWIND_COLORS["red-700"],  # #b91c1c - Oldest - highest
    "Unknown": TAILWIND_COLORS["slate-400"],  # #94a3b8 - Unknown
}

# ==============================================================================
# GRADIENT COLORS FOR CONTINUOUS DATA
# ==============================================================================

# Sequential gradient (light to dark) - for quantitative data
GRADIENT_SEQUENTIAL_BLUE = [
    TAILWIND_COLORS["blue-200"],
    TAILWIND_COLORS["blue-400"],
    TAILWIND_COLORS["blue-600"],
    TAILWIND_COLORS["blue-800"],
]

# Diverging gradient (cool to warm) - for data with natural midpoint
GRADIENT_DIVERGING = [
    TAILWIND_COLORS["blue-600"],  # Cool end
    TAILWIND_COLORS["cyan-400"],
    TAILWIND_COLORS["slate-300"],  # Neutral middle
    TAILWIND_COLORS["amber-400"],
    TAILWIND_COLORS["orange-600"],  # Warm end
]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_chart_palette(style: str = "vibrant") -> list[str]:
    """
    Get a color palette for charts based on the desired style.

    Args:
        style: One of "vibrant", "cool", "warm"

    Returns:
        List of hex color codes
    """
    palettes = {
        "vibrant": CHART_COLORS_VIBRANT,
        "cool": CHART_COLORS_COOL,
        "warm": CHART_COLORS_WARM,
    }
    return palettes.get(style, CHART_COLORS_VIBRANT)


def get_gender_color(gender: str) -> str:
    """
    Get the color for a specific gender category.

    Args:
        gender: Gender identifier (case-insensitive)

    Returns:
        Hex color code
    """
    gender_lower = str(gender).lower().strip()

    if gender_lower in ["male", "m"]:
        return GENDER_COLORS["male"]
    elif gender_lower in ["female", "f"]:
        return GENDER_COLORS["female"]
    elif gender_lower in ["other", "non-binary", "nb"]:
        return GENDER_COLORS["other"]
    else:
        return GENDER_COLORS["unknown"]


def get_risk_color(level: str) -> str:
    """
    Get the color for a risk/vulnerability level.

    Args:
        level: Risk level identifier

    Returns:
        Hex color code
    """
    return RISK_COLORS.get(level.lower(), RISK_COLORS["moderate"])
