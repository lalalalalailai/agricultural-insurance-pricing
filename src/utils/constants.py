CORE_SYMBOLS = [
    'A0', 'C0', 'Y0', 'P0', 'OI0', 'M0', 'RM0',
    'CF0', 'SR0', 'AP0', 'CJ0', 'LH0', 'JD0'
]

SYMBOL_NAMES = {
    'A0': '豆一', 'AP0': '苹果', 'C0': '玉米', 'CF0': '棉花',
    'CJ0': '红枣', 'CS0': '玉米淀粉', 'EB0': '苯乙烯',
    'EG0': '乙二醇', 'FG0': '玻璃', 'HC0': '热卷',
    'I0': '铁矿石', 'J0': '焦炭', 'JD0': '鸡蛋',
    'JM0': '焦煤', 'L0': '塑料', 'LH0': '生猪',
    'M0': '豆粕', 'MA0': '甲醇', 'OI0': '菜油',
    'P0': '棕榈油', 'PG0': 'LPG', 'PK0': '花生',
    'PP0': 'PP', 'RM0': '菜粕', 'RU0': '橡胶',
    'SA0': '纯碱', 'SF0': '硅铁', 'SM0': '锰硅',
    'SP0': '纸浆', 'SR0': '白糖', 'SS0': '不锈钢',
    'TA0': 'PTA', 'UR0': '尿素', 'V0': 'PVC',
    'Y0': '豆油', 'ZC0': '动力煤'
}

ALL_SYMBOLS = list(SYMBOL_NAMES.keys())

PROVINCES = {
    'northeast_heilongjiang': '黑龙江',
    'northeast_jilin': '吉林',
    'north_china_shandong': '山东',
    'north_china_henan': '河南',
    'central_china_hubei': '湖北',
    'south_china_guangxi': '广西',
    'northwest_xinjiang': '新疆'
}

PROVINCE_FILES = {
    '黑龙江': '东北_黑龙江.csv',
    '吉林': '东北_吉林.csv',
    '山东': '华北_山东.csv',
    '河南': '华北_河南.csv',
    '湖北': '华中_湖北.csv',
    '广西': '华南_广西.csv',
    '新疆': '西北_新疆.csv'
}

SYMBOL_PROVINCE_MAP = {
    'A0': ['黑龙江', '吉林'],       # 豆一 → 东北主产区
    'M0': ['黑龙江', '内蒙古'],     # 豆粕 → 黑龙江（用黑龙江代替）
    'C0': ['吉林', '黑龙江'],       # 玉米 → 东北主产区
    'Y0': ['黑龙江', '吉林'],       # 豆油 → 东北主产区
    'P0': ['海南', '广东'],         # 棕榈油 → 进口（无数据，用广西代替）
    'OI0': ['长江流域', '四川'],    # 菜油 → 长江流域（用湖北代替）
    'RM0': ['长江流域', '广西'],    # 菜粕 → （用湖北、广西代替）
    'CF0': ['新疆'],                 # 棉花 → 新疆主产区
    'SR0': ['广西'],                 # 白糖 → 广西主产区
    'AP0': ['陕西', '山东'],        # 苹果 → 陕西、山东主产区
    'CJ0': ['新疆'],                 # 红枣 → 新疆主产区
    'LH0': ['河南', '四川'],        # 生猪 → 河南、四川主产区
    'JD0': ['河北', '山东'],        # 鸡蛋 → 河北、山东主产区
}
DEFAULT_PROVINCES = ['黑龙江', '吉林', '山东', '河南', '湖北', '广西', '新疆']

RISK_LEVELS = {
    'low': {'range': (0, 25), 'color': '#4caf50', 'label': '低风险', 'icon': '🟢'},
    'medium': {'range': (26, 50), 'color': '#ff9800', 'label': '中风险', 'icon': '🟡'},
    'high': {'range': (51, 75), 'color': '#ff5722', 'label': '高风险', 'icon': '🟠'},
    'extreme': {'range': (76, 100), 'color': '#f44336', 'label': '极高风险', 'icon': '🔴'}
}

MODEL_PARAMS = {
    'xgboost': {
        'learning_rate': 0.05,
        'max_depth': 6,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    },
    'pc_algorithm': {
        'alpha': 0.05,
        'max_cond_set_size': 3,
        'method': 'pearsonr'
    },
    'psm': {
        'match_method': 'nearest',
        'ratio': 3,
        'caliper': None
    }
}

COLOR_PALETTE = {
    'primary': '#2563eb',
    'accent': '#d97706',
    'success': '#059669',
    'warning': '#d97706',
    'danger': '#dc2626',
    'bg_dark': '#1e293b',
    'bg_card': '#ffffff',
    'text_primary': '#0f172a',
    'text_secondary': '#475569'
}

DATA_CONFIG = {
    'futures_dir': 'enhanced_data/all_factors/futures',
    'macro_dir': 'enhanced_data/all_factors/macro',
    'weather_dir': 'enhanced_data/weather',
    'extended_dir': 'enhanced_data/extended_factors',
    'supplementary_dir': 'enhanced_data/supplementary_data',
    'remote_sensing_dir': 'enhanced_data/remote_sensing',
    'date_col': 'date',
    'price_cols': ['open', 'high', 'low', 'close', 'settle', 'volume'],
    'target_col': 'close'
}

FEATURE_CONFIG = {
    'lag_periods': [1, 5, 10, 20],
    'ma_windows': [5, 10, 20],
    'total_features': 28
}

DAG_CONFIG = {
    'num_nodes': 15,
    'num_edges': None,
    'num_edges_note': '实际边数由Agri-PC算法动态生成(约束前约52条,三重约束后约22-30条),此处不硬编码',
    'nodes': ['futures_high', 'futures_low', 'settle', 'close', 'volume',
              'weather_risk', 'yield', 'cpi', 'supply_demand',
              'macro_pmi', 'policy_risk', 'risk_premium',
              'ndvi', 'lst', 'evi'],
    'core_chains': [
        'weather_risk -> yield -> close -> risk_premium',
        'cpi -> supply_demand -> close',
        'ndvi -> yield -> supply_demand -> close -> risk_premium',
        'lst -> weather_risk -> yield -> close',
        'evi -> ndvi -> yield -> close'
    ],
    'node_categories': {
        'meteorological': ['weather_risk'],
        'remote_sensing': ['ndvi', 'evi', 'lst'],
        'agricultural': ['yield', 'supply_demand'],
        'macroeconomic': ['cpi', 'macro_pmi', 'policy_risk'],
        'market': ['futures_high', 'futures_low', 'settle', 'close', 'volume'],
        'derived': ['risk_premium']
    },
    'temporal_order': {
        'T0_exogenous': ['weather_risk', 'ndvi', 'evi', 'lst'],
        'T1_agricultural': ['yield'],
        'T2_macro': ['cpi', 'macro_pmi'],
        'T3_supply_policy': ['supply_demand', 'policy_risk'],
        'T4_market_volume': ['volume', 'futures_high', 'futures_low'],
        'T5_price': ['settle', 'close'],
        'T6_risk': ['risk_premium']
    },
    'note': '15节点30边为Agri-PC三重约束后的最终因果图结构，含5条核心因果链和7层时序偏序'
}

PRICING_CONFIG = {
    'target_mape': 3.0,
    'achieved_mape': 0.42,
    'achieved_accuracy': 99.58,
    'achieved_mape_with_lag': 0.42,
    'achieved_mape_pure_prediction': 3.8,
    'accuracy_note': '99.58% = (1 - MAPE) * 100，含lag特征；纯预测MAPE 3.8%反映模型独立预测能力(无lag特征)',
    'train_period': '2020-2024',
    'test_period': '2025',
    'cv_folds': 5,
    'market_size_billion': 1521,
    'market_size_source': '财政部2024年数据',
    'market_size_year': 2024,
    'lag_vs_pure_note': '含lag特征MAPE反映模型+随机游走综合精度，纯预测MAPE反映模型独立预测能力，二者均需报告'
}

DATA_VALIDATION_CONFIG = {
    'date_range_start': '2020-01-01',
    'date_range_end': '2025-12-31',
    'strict_mode': True,
    'check_items': [
        'date_range_compliance',
        'missing_value_ratio',
        'duplicate_records',
        'value_continuity',
        'cross_source_consistency'
    ],
    'thresholds': {
        'max_missing_ratio': 0.05,
        'max_duplicate_ratio': 0.01,
        'max_date_gap_days': 7
    }
}
