# -*- coding: utf-8 -*-
"""
Streamlit Cloud Deployment Entry Point
========================================
Agricultural Futures Intelligent Pricing System
Deployed on Streamlit Community Cloud (Linux/Ubuntu)
"""
import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

if os.path.exists(os.path.join(_SRC, 'app.py')):
    from app import *
else:
    import streamlit as st
    st.error("Source code not found. Please ensure src/app.py exists.")
    st.stop()
