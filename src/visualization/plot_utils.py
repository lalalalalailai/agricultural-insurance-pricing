import os
from typing import List, Optional

try:
    import plotly.io as pio
except ImportError:
    pio = None

FINANCIAL_CSS = """
<style>
    .stApp { background: #FAFBF8; }
    h1, h2, h3 { color: #0D110E !important; }
    .stMetric { background: #ffffff; border-radius: 10px; padding: 10px;
                border: 1px solid #E0E5DD; box-shadow: 0 1px 3px rgba(26,31,22,0.06); }
</style>
"""

COLOR_SCHEMES = {
    'financial': {
        'primary': '#1B6B48',
        'accent': '#C4880C',
        'success': '#1E8A5A',
        'warning': '#D4790C',
        'danger': '#C7302D',
        'bg_dark': '#1A1F16',
        'bg_card': '#FFFFFF',
        'text_primary': '#1A1F16',
        'text_secondary': '#3D4A3A'
    },
    'gradient_green': ['#1B6B48', '#2D8B5A', '#4AAF7E', '#8FD4AD', '#C8EBDC'],
    'risk_colors': ['#1E8A5A', '#D4790C', '#E07800', '#C7302D'],
    'feature_colors': ['#1B6B48', '#2D8B5A', '#4AAF7E', '#6BC49E', '#96DDBB',
                        '#A8E3C9', '#BAE9D7', '#D0EEE2', '#E2F4EC', '#F0F8F3']
}


def set_financial_theme():
    try:
        import streamlit as st
        st.markdown(FINANCIAL_CSS, unsafe_allow_html=True)
    except Exception:
        pass


def save_figure(fig, filename: str, dpi: int = 300,
                 formats: List[str] = ['png'], directory: str = None):
    if pio is None or fig is None:
        return None
    save_dir = directory or os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    for fmt in formats:
        filepath = os.path.join(save_dir, f"{filename}.{fmt}")
        try:
            if fmt == 'html':
                pio.write_html(fig, filepath, include_plotlyjs=True)
            elif fmt == 'png':
                pio.write_image(fig, filepath, scale=dpi/96, engine='kaleido')
            saved_paths.append(filepath)
        except Exception as e:
            pass
    return saved_paths


def format_large_number(x, decimals: int = 1) -> str:
    if abs(x) >= 1e8:
        return f"{x/1e8:.{decimals}f}亿"
    elif abs(x) >= 1e4:
        return f"{x/1e4:.{decimals}f}万"
    else:
        return f"{x:.{decimals}f}"


def annotate_significant_points(fig, points: dict, text: str = ""):
    if fig is None:
        return fig
    try:
        for idx, (date, val) in points.items():
            fig.add_annotation(
                x=date, y=val,
                text=text or f"关键点: {val:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#C4880C",
                font=dict(color="#C4880C", size=10)
            )
    except Exception:
        pass
    return fig
