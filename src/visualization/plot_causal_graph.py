import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    go = None
    px = None

from utils.helpers import logger_setup

logger = logger_setup('plot_causal')

def plot_causal_dag(nodes: List[str], edges: List[Tuple[str, str]],
                     edge_strengths: Dict[Tuple[str, str], float] = None,
                     node_sizes: Dict[str, int] = None,
                     title: str = "因果发现图谱 (PC Algorithm)") -> 'go.Figure':
    if go is None:
        return _placeholder_figure("请安装plotly: pip install plotly")
    edge_strengths = edge_strengths or {}
    node_positions = {}
    for i, node in enumerate(nodes):
        row = i // 4
        col = i % 4
        node_positions[node] = {
            'x': col * 2 + (row % 2) * 0.5,
            'y': -row * 2
        }
    edge_traces = []
    for src, tgt in edges:
        if src not in node_positions or tgt not in node_positions:
            continue
        src_pos = node_positions[src]
        tgt_pos = node_positions[tgt]
        strength = edge_strengths.get((src, tgt), 0.5)
        line_width = max(1, min(4, strength * 3))
        edge_traces.append(go.Scatter(
            x=[src_pos['x'], tgt_pos['x']],
            y=[src_pos['y'], tgt_pos['y']],
            mode='lines',
            line=dict(width=line_width, color='#888'),
            hoverinfo='text',
            hovertext=f"{src} → {tgt}: {strength:.1%}" if strength > 0 else f"{src} → {tgt}",
            showlegend=False
        ))
    if node_sizes:
        sizes = [node_sizes.get(n, 35) for n in nodes]
    else:
        sizes = [35] * len(nodes)
    node_x = [node_positions[n]['x'] for n in nodes]
    node_y = [node_positions[n]['y'] for n in nodes]
    node_text = []
    for n in nodes:
        influence = edge_strengths.get((n, 'close'), None)
        if influence is not None:
            node_text.append(f"{n}<br>影响度: {influence:.1f}%")
        else:
            node_text.append(n)
    node_colors = []
    for n in nodes:
        if n == 'close':
            node_colors.append('#d97706')
        elif n in ['weather_risk', 'yield', 'cpi', 'supply_demand']:
            node_colors.append('#4caf50')
        elif n in ['futures_high', 'futures_low', 'settle', 'volume']:
            node_colors.append('#2196f3')
        elif n in ['macro_pmi', 'policy_risk', 'risk_premium']:
            node_colors.append('#ff9800')
        else:
            node_colors.append('#1a237e')
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(
            size=sizes,
            color=node_colors,
            line=dict(width=2, color='#475569')
        ),
        hovertemplate='%{text}<extra></extra>',
        textfont=dict(size=10, color='#475569'),
        showlegend=False
    )
    annotations = []
    for src, tgt in edges:
        if src not in node_positions or tgt not in node_positions:
            continue
        strength = edge_strengths.get((src, tgt), 0)
        if strength > 0:
            src_pos = node_positions[src]
            tgt_pos = node_positions[tgt]
            mid_x = (src_pos['x'] + tgt_pos['x']) / 2
            mid_y = (src_pos['y'] + tgt_pos['y']) / 2
            annotations.append(dict(
                x=mid_x, y=mid_y,
                text=f"{strength:.0f}%",
                showarrow=False,
                font=dict(size=8, color='#d97706'),
                bgcolor='rgba(248,250,252,0.85)'
            ))
    all_traces = edge_traces + [node_trace]
    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#475569', family='Microsoft YaHei')),
        showlegend=False,
        template='plotly_white',
        paper_bgcolor='#f8fafc',
        plot_bgcolor='#f8fafc',
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, 10]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-8, 2]),
        margin=dict(l=50, r=50, t=80, b=50),
        height=550,
        hovermode='closest'
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame,
                               variables: List[str] = None,
                               title: str = "变量相关性热力图") -> 'go.Figure':
    if go is None:
        return _placeholder_figure("请安装plotly: pip install plotly")
    if variables is None:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()[:12]
    corr_matrix = df[variables].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values.round(3),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=9),
        colorbar=dict(title="相关系数", thickness=15)
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#475569', family='Microsoft YaHei')),
        template='plotly_white',
        paper_bgcolor='#f8fafc',
        plot_bgcolor='#f8fafc',
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        height=500,
        width=600,
        margin=dict(l=100, r=20, t=60, b=120)
    )
    return fig


def _placeholder_figure(message: str) -> 'go.Figure':
    if go is None:
        class DummyFigure:
            def show(self): print(message)
        return DummyFigure()
    fig = go.Figure()
    fig.update_layout(title=message, template='plotly_white', paper_bgcolor='#f8fafc')
    return fig
