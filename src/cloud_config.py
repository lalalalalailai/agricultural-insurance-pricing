# -*- coding: utf-8 -*-
"""
Cloud Deployment Config Override
==================================
Overrides base config for Streamlit Cloud (Linux) environment.
Auto-detects data paths and handles missing data gracefully.
"""
import os
import sys
from pathlib import Path

_CLOUD_ROOT = Path(__file__).resolve().parent.parent


def get_cloud_base_dir() -> Path:
    return _CLOUD_ROOT


def get_cloud_data_root() -> Path:
    for candidate in ['enhanced_data', 'data', 'enhanced_data/all_factors']:
        p = _CLOUD_ROOT / candidate
        if p.exists():
            return p
    return _CLOUD_ROOT / 'data'


def is_cloud_env() -> bool:
    return os.environ.get('STREAMLIT_CLOUD', '') == '1' or \
           'STREAMLIT_SHARING_MODE' in os.environ or \
           not sys.platform.startswith('win')


DEMO_SYMBOLS = ['A0']
CLOUD_AVAILABLE_SYMBOLS = []

DEPLOY_VERSION = '2026-04-23T12:30:00'
DEPLOY_NOTE = 'Streamlit Cloud auto-deploy trigger - pricing_model.py sync'
