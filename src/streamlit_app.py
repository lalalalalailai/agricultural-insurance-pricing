# -*- coding: utf-8 -*-
"""
Streamlit Cloud Safe Entry Point v2
=====================================
完全自包含的云安全入口 - 防止1ST崩溃错误

核心策略:
1. 最小化顶层import (仅streamlit + 标准库)
2. 懒加载重型模块 (延迟到用户交互时)
3. 全链路异常捕获 (展示友好错误信息)
4. 降级演示模式 (即使模块加载失败也能显示页面)
"""
import sys
import os
import traceback

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PARENT_DIR = os.path.dirname(SRC_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import streamlit as st

st.set_page_config(
    page_title="农险期货智能定价系统",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;600;700&display=swap');
* { font-family: 'Noto Sans SC', sans-serif; }
.main-header { 
    background: linear-gradient(135deg, #1a365d 0%, #2c5282 50%, #2b6cb0 100%);
    padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
    color: white; text-align: center;
}
.metric-card {
    background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
    border-radius: 10px; padding: 1.5rem; text-align: center;
    border-left: 4px solid #3182ce; margin: 0.5rem 0;
}
.success-banner {
    background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%);
    border-radius: 8px; padding: 1rem; border-left: 4px solid #38a169;
}
.error-banner {
    background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%);
    border-radius: 8px; padding: 1rem; border-left: 4px solid #e53e3e;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

APP_LOADED = False
LOAD_ERROR = None

def try_load_app():
    """尝试加载主应用模块，返回是否成功"""
    global APP_LOADED, LOAD_ERROR
    try:
        os.chdir(SRC_DIR)
    except OSError:
        pass
    
    try:
        from utils.config import config
        from utils.constants import CORE_SYMBOLS, SYMBOL_NAMES
        from data.data_loader import DataLoader
        from models.pricing_model import PricingModel
        APP_LOADED = True
        return True
    except Exception as e:
        LOAD_ERROR = str(e)
        return False

def render_demo_mode():
    """降级演示模式 - 即使模块加载失败也能展示系统概览"""
    st.markdown("""
    <div class="main-header">
        <h1>🌾 农险期货智能定价系统</h1>
        <h3>乡村振兴背景下基于因果推断的农险期货智能定价模型</h3>
        <p style="opacity: 0.9;">Agricultural Futures Intelligent Pricing Model Based on Causal Inference</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color:#2c5282;margin:0;">36</h2>
            <p style="margin:0;color:#4a5568;">覆盖品种数</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color:#2c5282;margin:0;">3</h2>
            <p style="margin:0;color:#4a5568;">原创算法</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color:#2c5282;margin:0;">6</h2>
            <p style="margin:0;color:#4a5568;">数学定理</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color:#2c5282;margin:0;">2020-2025</h2>
            <p style="margin:0;color:#4a5568;">数据时间范围</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 系统架构", "🧮 核心算法", "📁 数据说明", "ℹ️ 关于"])
    
    with tab1:
        st.subheader("系统技术栈")
        
        tech_data = {
            "前端框架": "Streamlit (Cloud部署)",
            "后端算法": "Python 3.9+ / NumPy / Pandas / SciPy",
            "因果发现": "Agri-PC (农业约束PC算法)",
            "因果估计": "ACML (异质因果元学习器)",
            "预测保真": "CCP (因果保真预测)",
            "数据来源": "36品种期货 + 7主产区气象 + 4类遥感",
            "AI工具链": "Qwen3.6-Plus + GLM-5 + MiniMax M2.7"
        }
        
        for key, value in tech_data.items():
            cols = st.columns([1, 3])
            with cols[0]:
                st.markdown(f"**{key}**")
            with cols[1]:
                st.markdown(value)
    
    with tab2:
        st.subheader("三大原创算法")
        
        st.markdown("""
        <div class="success-banner">
            <strong>🔬 Agri-PC (农业约束PC算法)</strong><br/>
            融合时序约束/农业先验/交割规则的因果发现算法，解决传统PC在农业场景中的方向误判问题。
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-banner" style="border-left-color:#805ad5;">
            <strong>🎯 ACML (异质因果元学习器)</strong><br/>
            基于双重正交化的异质性因果效应估计框架，处理品种间、地区间、时期间的异质性定价因子。
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-banner" style="border-left-color:#dd6b20;">
            <strong>🛡️ CCP (因果保真预测)</strong><br/>
            自适应覆盖率保真预测区间，结合因果结构信息的共形预测方法。
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("数据集构成")
        
        data_info = [
            ("📈 期货价格数据", "36个农产品期货品种日度行情 (2020-2025)"),
            ("🌤️ 气象数据", "7大主产区日度气象数据 (温度/降水/湿度等)"),
            ("🛰️ 遥感数据", "NDVI/EVI/LST/干旱指数 月度遥感指标"),
            ("📋 宏观经济", "CPI/PMI等宏观经济指标"),
        ]
        
        for title, desc in data_info:
            st.markdown(f"- **{title}**: {desc}")
    
    with tab4:
        st.subheader("关于本项目")
        
        st.markdown("""
        <div class="error-banner">
            <strong>⚠️ 当前运行模式：演示模式</strong><br/><br/>
            完整功能需要加载数据文件。当前Streamlit Cloud环境可能缺少部分数据依赖。<br/>
            错误详情：<code>{}</code>
        </div>
        """.format(LOAD_ERROR or "未知"), unsafe_allow_html=True)
        
        st.markdown("""
        ---
        **项目信息**
        - **参赛类别**: 2026年中国大学生计算机设计大赛 - 大数据应用
        - **作品名称**: 乡村振兴背景下基于因果推断的农险期货智能定价模型
        - **在线地址**: （提交后公布）
        - **部署版本**: v20260423
        """)

def render_full_app():
    """完整模式 - 加载所有模块并运行完整应用"""
    try:
        import app
    except Exception as e:
        st.error(f"⚠️ 应用加载异常: {e}")
        st.code(traceback.format_exc())
        render_demo_mode()

def main():
    """主入口函数"""
    with st.spinner('🔄 正在加载农险期货智能定价系统...'):
        success = try_load_app()
    
    if success:
        render_full_app()
    else:
        render_demo_mode()

if __name__ == '__main__':
    main()

main()
