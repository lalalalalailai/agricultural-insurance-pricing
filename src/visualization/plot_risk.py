import pandas as pd
import numpy as np
from typing import Dict, List, Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None

from utils.helpers import logger_setup
from utils.constants import RISK_LEVELS

logger = logger_setup('plot_risk')


def plot_risk_dashboard(risk_score: float,
                           risk_level: str = None,
                           title: str = "综合风险指数") -> 'go.Figure':
    if go is None:
        return _placeholder("需要安装plotly")
    level_info = RISK_LEVELS.get(risk_level or 'medium', RISK_LEVELS['medium'])
    color = level_info.get('color', '#ff9800')
    label = level_info.get('label', risk_level or '未知')
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': 50, 'increasing': {'color': '#f44336'}, 'decreasing': {'color': '#4caf50'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#8b949e"},
            'bar': {'color': color},
            'bgcolor': "#ffffff",
            'borderwidth': 2,
            'bordercolor': "#30363d",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(76,175,80,0.2)'},
                {'range': [26, 50], 'color': 'rgba(255,152,0,0.2)'},
                {'range': [51, 75], 'color': 'rgba(255,87,34,0.2)'},
                {'range': [76, 100], 'color': 'rgba(244,67,54,0.2)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        },
        title={'text': f"{title}<br><span style='font-size:14px;color:{color}'>{label}</span>",
               'font': {'size': 16, 'color': 'white'}}
    ))
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='#f8fafc',
        height=300,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig


def plot_risk_radar(dimension_scores: Dict[str, float],
                       title: str = "四维风险评估雷达图") -> 'go.Figure':
    if go is None:
        return _placeholder("需要安装plotly")
    categories = ['天气风险', '市场风险', '政策风险', '宏观风险']
    dim_keys = ['weather', 'market', 'policy', 'macro']
    scores = [dimension_scores.get(k, 50) for k in dim_keys]
    scores_closed = scores + [scores[0]]
    categories_closed = categories + [categories[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_closed, theta=categories_closed,
        fill='toself', fillcolor='rgba(37,99,235,0.15)',
        line=dict(color='#2563eb', width=3),
        name='当前风险值'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[50]*5, theta=categories_closed,
        fill='toself', fillcolor='#e2e8f0',
        line=dict(color='#94a3b8', width=1, dash='dash'),
        name='历史均值'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                                   gridcolor='#e2e8f0', tickfont=dict(color='#94a3b8')),
                   angularaxis=dict(gridcolor='#e2e8f0', tickfont=dict(color='#94a3b8'))),
        title=dict(text=title, font=dict(size=16, color='#475569')),
        template='plotly_white', paper_bgcolor='#f8fafc',
        showlegend=True, legend=dict(font=dict(color='#475569')),
        height=450, margin=dict(l=60, r=60, t=60, b=60)
    )
    return fig


def plot_risk_trend(dates, risk_scores: List[float],
                      threshold: float = 50,
                      title: str = "风险趋势分析") -> 'go.Figure':
    if go is None:
        return _placeholder("需要安装plotly")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=risk_scores,
        name='风险指数', line=dict(color='#2563eb', width=2),
        fill='tozeroy', fillcolor='rgba(26,35,126,0.1)'
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#f44336",
                   annotation_text="预警阈值", annotation_position="top right",
                   annotation_font=dict(color="#f44336"))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#475569')),
        template='plotly_white', paper_bgcolor='#f8fafc', plot_bgcolor='#f8fafc',
        xaxis=dict(title='时间', gridcolor='#e2e8f0'),
        yaxis=dict(title='风险指数', range=[0, 100], gridcolor='#e2e8f0'),
        height=350, margin=dict(l=60, r=40, t=60, b=60)
    )
    return fig


def plot_risk_alert_table(alerts: List,
                            title: str = "风险预警列表") -> 'go.Figure':
    if go is None or not alerts:
        return _placeholder("需要安装plotly或无预警信息")
    symbols = [a.symbol for a in alerts]
    scores = [a.risk_score for a in alerts]
    levels = [a.risk_level for a in alerts]
    messages = [a.message for a in alerts]
    level_colors = {
        'low': '#4caf50', 'medium': '#ff9800',
        'high': '#ff5722', 'extreme': '#f44336'
    }
    cell_colors = [[level_colors.get(l, '#888')] for l in levels]
    fig = go.Figure(data=[go.Table(
        header=dict(values=['品种', '风险分数', '风险等级', '建议'],
                    fill_color='#2563eb', font=dict(color='#475569', size=12),
                    align='center', height=30),
        cells=dict(values=[symbols, [f'{s:.1f}' for s in scores], levels, messages],
                  fill_color=cell_colors, font=dict(color='#475569', size=11),
                  align='left', height=28)
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#475569')),
        template='plotly_white', paper_bgcolor='#f8fafc',
        height=min(400, 50 + len(alerts) * 32),
        margin=dict(l=20, r=20, t=60, b=20)
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
