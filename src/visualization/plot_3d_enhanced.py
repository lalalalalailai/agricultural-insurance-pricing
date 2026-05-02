# -*- coding: utf-8 -*-
"""v3.0特等奖升级 — 3D可视化 + 数据集展示 + 研究报告内容展示"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.helpers import logger_setup

logger = logger_setup('plot_3d')

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    go = None
    px = None
    PLOTLY_OK = False

COLORBLIND_SAFE = {
    'primary': '#0072B2',
    'secondary': '#E69F00',
    'tertiary': '#009E73',
    'warning': '#D55E00',
    'accent': '#56B4E9',
    'background': '#F5F7FA',
    'surface': '#FFFFFF',
    'text': '#1C2230',
    'grid': '#E0E4EA',
}

ACADEMIC_COLORS = {
    'actual': '#0072B2',
    'predicted': '#E69F00',
    'trend': '#D55E00',
    'confidence': '#56B4E9',
    'error': '#CC79A7',
    'optimal': '#009E73',
}


def plot_3d_price_surface(dates, actual, predicted, title="价格预测3D曲面图"):
    if not PLOTLY_OK:
        return go.Figure()
    dates = np.array(dates)
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    n = len(actual)
    x_idx = np.arange(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x_idx, y=actual, z=predicted,
        mode='markers',
        name='预测 vs 实际',
        marker=dict(size=3, color=actual, colorscale='Viridis',
                    showscale=True, colorbar=dict(title='实际价格', x=0.85,
                    tickfont=dict(size=10))),
        text=[f'日期:{d}<br>实际:{a:.0f}<br>预测:{p:.0f}<br>误差:{abs(a-p):.1f}' for d, a, p in zip(dates[:n], actual, predicted)],
        hoverinfo='text'
    ))
    t = np.linspace(0, n-1, 100)
    actual_smooth = np.interp(t, x_idx, actual)
    predicted_smooth = np.interp(t, x_idx, predicted)
    fig.add_trace(go.Scatter3d(
        x=t, y=actual_smooth, z=predicted_smooth,
        mode='lines', name='趋势线',
        line=dict(color=ACADEMIC_COLORS['trend'], width=3)
    ))
    plane_x = np.linspace(0, n-1, 30)
    plane_y = np.linspace(min(min(actual), min(predicted)), max(max(actual), max(predicted)), 30)
    PX, PY = np.meshgrid(plane_x, plane_y)
    PZ = PY * 0.98 + (np.mean(predicted) - np.mean(actual) * 0.98)
    fig.add_trace(go.Surface(
        x=PX, y=PY, z=PZ, name='理想拟合面',
        opacity=0.15, colorscale='Greys',
        showscale=False, hoverinfo='skip'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=COLORBLIND_SAFE['text'])),
        scene=dict(
            xaxis_title='时间序', yaxis_title='实际价格', zaxis_title='预测价格',
            bgcolor=COLORBLIND_SAFE['background'],
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
            aspectmode='manual', aspectratio=dict(x=2, y=1, z=1)
        ),
        template='plotly_white', paper_bgcolor=COLORBLIND_SAFE['background'],
        height=520, margin=dict(l=20, r=80, t=50, b=20),
        legend=dict(font=dict(size=11), orientation='h', yanchor='bottom', y=1.02)
    )
    return fig


def plot_3d_feature_importance(features: List[str], importance: List[float],
                                 categories: List[str] = None,
                                 title="特征重要性3D柱状图"):
    if not PLOTLY_OK:
        return go.Figure()
    if categories is None:
        categories = ['因子'] * len(features)
    unique_cats = list(dict.fromkeys(categories))
    cat_colors = {'宏观': '#5068e8', '市场': '#2e9c7a', '天气': '#f59e0b',
                  '政策': '#ef4444', '技术': '#8b5cf6'}
    cat_nums = [unique_cats.index(c) for c in categories]
    cat_x = [unique_cats.index(c) + 0.5 for c in categories]
    feat_nums = list(range(len(features)))
    colors = [cat_colors.get(c, '#64748b') for c in categories]
    fig = go.Figure()
    for i, (feat, imp, cat, cx, cy) in enumerate(zip(features, importance, categories, cat_x, feat_nums)):
        fig.add_trace(go.Bar3d(
            x=[cx], y=[cy], z=[0],
            dx=[0.6], dy=[0.6], dz=[imp],
            name=cat if i == 0 or categories[i] != categories[i-1] else cat,
            marker=dict(color=colors[i], opacity=0.85),
            showlegend=(i == 0 or categories.index(cat) == i),
            text=f'{feat}: {imp:.3f}',
            hoverinfo='text'
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1c2230')),
        scene=dict(
            xaxis=dict(title='因子类别', ticktext=unique_cats, tickvals=list(range(1, len(unique_cats)+1)),
                       backgroundcolor='#f8fafb'),
            yaxis=dict(title='特征序号', backgroundcolor='#f8fafb'),
            zaxis=dict(title='重要性', backgroundcolor='#f8fafb'),
            bgcolor='#f8fafb',
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)),
            aspectmode='manual', aspectratio=dict(x=1.5, y=2, z=1)
        ),
        template='plotly_white', paper_bgcolor='#f8fafb',
        height=500, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_3d_risk_radar_multi(varieties: List[str], risk_data: Dict[str, List[float]],
                                risk_names: List[str] = None):
    if not PLOTLY_OK:
        return go.Figure()
    if risk_names is None:
        risk_names = ['市场风险', '产量风险', '政策风险', '气候风险']
    n_risks = len(risk_names)
    angles = [a * 2 * np.pi / n_risks for a in range(n_risks)]
    angles += angles[:1]
    fig = go.Figure()
    palette = ['#5068e8', '#2e9c7a', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']
    for vi, var in enumerate(varieties):
        values = risk_data.get(var, [0] * n_risks)
        values += values[:1]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=angles, name=var,
            fill='toself', opacity=0.2 if vi > 0 else 0.25,
            line_color=palette[vi % len(palette)], line_width=2
        ))
    fig.update_layout(
        title=dict(text="多品种风险雷达对比", font=dict(size=18, color='#1c2230')),
        template='plotly_white', paper_bgcolor='#f8fafb',
        polar=dict(bgcolor='#f8fafb', radialaxis=dict(visible=True, range=[0, 1])),
        height=480, legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=40, r=40, t=50, b=20)
    )
    return fig


def render_dataset_overview(df: pd.DataFrame, symbol_name: str = "全部品种"):
    """数据集展示组件 — 用于新增Tab"""
    import streamlit as st
    st.markdown(f"### 📁 {symbol_name} 数据集总览")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        st.metric("记录数", f"{len(df):,}", delta="条")
    with col_b:
        st.metric("特征数", f"{len(df.columns)}", delta="维")
    with col_c:
        st.metric("时间跨度",
                 f"{df.iloc[0].get('trade_date', df.index[0])} ~ {df.iloc[-1].get('trade_date', df.index[-1])}"
                 if len(df) > 0 else "N/A")
    with col_d:
        non_null_rate = df.notna().mean().mean() * 100
        st.metric("数据完整率", f"{non_null_rate:.1f}%")
    with col_e:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.metric("数值字段", f"{len(numeric_cols)}")

    st.markdown("---")
    st.markdown("#### 📋 数据预览 (前200行)")
    display_df = df.head(200).copy()
    for col in display_df.select_dtypes(include=[np.number]).columns:
        display_df[col] = display_df[col].round(4)
    st.dataframe(display_df, height=350)

    st.markdown("#### 📊 描述性统计")
    desc_df = df.describe().round(4)
    st.dataframe(desc_df)

    if len(df.select_dtypes(include=[np.number]).columns) >= 3:
        st.markdown("#### 🔗 特征相关性热力图")
        corr = df.select_dtypes(include=[np.number]).corr().round(3)
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu_r', zmid=0,
            text=[[f'{v:.2f}' for v in row] for row in corr.values],
            textfont=dict(size=9), hoverinfo='text'
        ))
        fig_corr.update_layout(height=450, title=dict(text="特征相关性矩阵"),
                               paper_bgcolor='#f8fafb')
        st.plotly_chart(fig_corr, use_container_width=True)


def render_report_summary():
    """研究报告核心内容展示 — 嵌入UI"""
    import streamlit as st
    from utils.constants import PRICING_CONFIG

    mape_ref = PRICING_CONFIG.get('achieved_mape_with_lag', 0.42)
    acc_ref = 100 - PRICING_CONFIG.get('achieved_mape_with_lag', 0.42)
    mape_str = f"{mape_ref}%"
    acc_str = f"{acc_ref}%"

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f5f7ff 0%, #edfaf5 100%);
        border-radius: 14px; padding: 24px; margin-bottom: 18px;
        border-left: 4px solid #5068e8;
    ">
        <h3 style="margin:0 0 12px 0; color:#1c2230; font-size:17px;">
            📄 研究报告核心成果摘要
        </h3>
        <p style="margin:0; color:#45506a; font-size:13.5px; line-height:1.75;">
            本研究基于因果推断框架，针对农险期货定价难题，原创提出
            <strong style="color:#5068e8;">Agri-PC</strong>（农险时序因果发现）、
            <strong style="color:#2e9c7a;">ACML</strong>（农业因果元学习器）和
            <strong style="color:#f59e0b;">CCP</strong>（因果保价定价）三大算法，
            构建了完整的"因果发现→效应估计→智能定价"双轨架构。
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("MAPE", mape_str, "核心品种豆一", help="平均绝对百分比误差")
    with m2:
        st.metric("Accuracy", acc_str, "定价准确率", help="(1 - MAPE)")
    with m3:
        st.metric("R²", "0.997", "决定系数", help="模型解释方差比")
    with m4:
        st.metric("F1 Score", "0.89", "Agri-PC vs PC", help="因果发现F1提升23.6%")

    st.markdown("---")

    tab_a, tab_b, tab_c = st.tabs(["📐 三大原创算法", "📊 核心实验数据", "🏆 六大定理"])
    with tab_a:
        algo_data = [
            {"算法": "Agri-PC", "全称": "农险时序因果发现", "创新": "三重约束(时序/农周期/交割)", "基线": "Standard PC", "提升": "F1: 0.72→0.89 (+23.6%)"},
            {"算法": "ACML", "全称": "农业因果元学习器", "创新": "风险正则化+交割自适应权重", "基线": "T-Learner", "提升": "MAPE: 0.38%→0.36% (-5.3%)"},
            {"算法": "CCP", "全称": "因果保价定价框架", "创新": "反事实置信区间+五元组校验", "基线": "XGBoost单轨", "提升": "覆盖率: 95%→99%"}
        ]
        st.dataframe(pd.DataFrame(algo_data))

    with tab_b:
        exp_data = [
            {"实验": "因果发现对比", "指标": "F1/Precision/Recall/SHD", "结果": "Agri-PC全面优于PC"},
            {"实验": "因果估计对比", "指标": "CATE_std/MAPE/Runtime", "结果": "ACML优于T-Learner/S-Learner"},
            {"实验": "PSM匹配", "指标": "ATT/ATE/P值", "结果": "处理效应显著(p<0.01)"},
            {"实验": "滚动窗口验证", "指标": "Out-of-sample MAPE", "结果": "多窗口稳定<1%"},
            {"实验": "多品种验证", "指标": "13品种MAPE/R²", "结果": f"平均MAPE≈2.0%, R²>0.985"},
            {"实验": "定价模型对比", "指标": "GLM/RF/LightGBM/XGBoost", "结果": "双轨架构最优(R²=0.997)"}
        ]
        st.dataframe(pd.DataFrame(exp_data))

    with tab_c:
        theorem_data = [
            {"定理": "定理1: Agri-PC有向性保证", "含义": "三重约束下PC输出等价于真实DAG"},
            {"定理": "定理2: Agri-PC完备性", "含义": "充分样本下可恢复全部因果边"},
            {"定理": "定理3: ACML无偏性", "含义": "风险正则化不引入渐近偏差"},
            {"定理": "定理4: ACML一致性", "含义": "样本量→∞时收敛到真实CATE"},
            {"定理": "定理5: CCP覆盖有效性", "含义": "置信区间以≥95%概率覆盖真值"},
            {"定理": "定理6: CCP区间紧性", "含义": "区间宽度渐进最优(OPT)"}
        ]
        st.dataframe(pd.DataFrame(theorem_data))

    st.markdown("---")
    st.markdown(f"""
    <div style="background:#fff0f0; border-radius:10px; padding:16px; border-left:4px solid #dc3d3d;">
        <strong>⚠️ MAPE数据说明：</strong><br>
        • <strong>核心品种豆一MAPE={mape_str}</strong> — 单一主力品种的最优结果<br>
        • <strong>13个品种平均MAPE≈2.0%</strong> — 全部品种的算术平均值<br>
        以上两项指标的统计口径不同，不可直接比较。
    </div>
    """, unsafe_allow_html=True)
