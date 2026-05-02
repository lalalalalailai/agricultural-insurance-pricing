import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None

from utils.helpers import logger_setup

logger = logger_setup('plot_pricing')

def plot_pricing_result(pricing_df: pd.DataFrame,
                          date_col: str = 'date',
                          actual_col: str = 'actual_price',
                          predicted_col: str = 'predicted_price',
                          final_col: str = 'final_price',
                          title: str = "定价结果详情") -> 'go.Figure':
    if go is None or pricing_df.empty:
        return _placeholder("需要安装plotly或无数据")
    dates = pricing_df[date_col].values if date_col in pricing_df.columns else range(len(pricing_df))
    actual = pricing_df[actual_col].values if actual_col in pricing_df.columns else None
    predicted = pricing_df[predicted_col].values if predicted_col in pricing_df.columns else None
    final = pricing_df[final_col].values if final_col in pricing_df.columns else predicted
    fig = go.Figure()
    if actual is not None:
        fig.add_trace(go.Scatter(x=dates, y=actual, name='实际价格',
                                 line=dict(color='#4caf50', width=2)))
    if predicted is not None:
        fig.add_trace(go.Scatter(x=dates, y=predicted, name='基准预测',
                                 line=dict(color='#d97706', width=2, dash='dash')))
    if final is not None:
        fig.add_trace(go.Scatter(x=dates, y=final, name='最终定价',
                                 line=dict(color='#ff9800', width=2)))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#475569')),
        template='plotly_white', paper_bgcolor='#f8fafc', plot_bgcolor='#f8fafc',
        legend=dict(font=dict(color='#475569'), orientation='h', yanchor='bottom', y=1.02),
        xaxis=dict(title='日期', gridcolor='#e2e8f0'), yaxis=dict(title='价格', gridcolor='#e2e8f0'),
        height=450, margin=dict(l=60, r=40, t=60, b=60), hovermode='x unified'
    )
    return fig


def plot_risk_premium_composition(premium_breakdown: Dict[str, float],
                                    title: str = "风险溢价构成") -> 'go.Figure':
    if go is None:
        return _placeholder("需要安装plotly")
    labels = list(premium_breakdown.keys())
    values = list(premium_breakdown.values())
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'][:len(labels)]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.4, marker_colors=colors,
        textinfo='label+percent', textfont=dict(color='#475569', size=12)
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#475569')),
        template='plotly_white', paper_bgcolor='#f8fafc',
        height=400, margin=dict(t=60, b=40, l=40, r=40)
    )
    return fig


def plot_sensitivity_analysis(params: Dict[str, Tuple[float, float]],
                                 base_value: float,
                                 title: str = "敏感性分析") -> 'go.Figure':
    if go is None:
        return _placeholder("需要安装plotly")
    param_names = list(params.keys())
    impacts_low = []
    impacts_high = []
    for p_name, (low_val, high_val) in params.items():
        impacts_low.append(base_value * (1 + low_val))
        impacts_high.append(base_value * (1 + high_val))
    fig = go.Figure()
    for i, pname in enumerate(param_names):
        fig.add_trace(go.Scatter(
            x=[impacts_low[i], base_value, impacts_high[i]],
            y=[pname, pname, pname],
            mode='lines+markers',
            line=dict(width=3, color='#1a237e'),
            marker=dict(size=8),
            name=pname
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#475569')),
        template='plotly_white', paper_bgcolor='#f8fafc', plot_bgcolor='#f8fafc',
        xaxis=dict(title='定价变化', gridcolor='#e2e8f0'),
        yaxis=dict(categoryorder='array', categoryarray=param_names),
        height=max(300, len(param_names) * 35), showlegend=False,
        margin=dict(l=100, r=40, t=60, b=60)
    )
    return fig


def _placeholder(msg):
    if go is None:
        class D:
            def show(self): print(msg)
        return D()
    fig = go.Figure()
    fig.update_layout(title=msg, template='plotly_white')
    return fig
