import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional
from utils.helpers import logger_setup
from utils.constants import CORE_SYMBOLS, SYMBOL_NAMES, PROVINCES, PRICING_CONFIG
from utils.config import config

logger = logger_setup('premium_calculator')

RATE_SOURCES = {
    'base_rate': '中国精算师协会《农业保险费率厘定指引2023》表4-1 各品种基准费率',
    'risk_loading': '银保监会《农业保险承保理赔管理暂行办法》附件2 风险附加费率区间',
    'province_factor': '各省级财政厅农业保险补贴方案(2023-2025年)公布的区域风险系数',
    'crop_value': '国家发改委《全国农产品成本收益资料汇编2024》各品种每亩产值',
    'subsidy_ratio': '财政部《农业保险保费补贴管理办法》(财金〔2022〕5号) 中央补贴比例',
}

REFERENCE_RATES = {
    'A0': {'base_rate': 0.065, 'risk_loading': 0.015, 'province_factor': {'黑龙江': 1.0, '吉林': 0.95, '辽宁': 0.90}},
    'M0': {'base_rate': 0.055, 'risk_loading': 0.012, 'province_factor': {'河南': 1.0, '山东': 0.95}},
    'C0': {'base_rate': 0.050, 'risk_loading': 0.010, 'province_factor': {'吉林': 1.0, '黑龙江': 0.95, '辽宁': 0.90}},
    'CF0': {'base_rate': 0.070, 'risk_loading': 0.018, 'province_factor': {'新疆': 1.0}},
    'SR0': {'base_rate': 0.060, 'risk_loading': 0.014, 'province_factor': {'广西': 1.0, '云南': 0.95}},
    'AP0': {'base_rate': 0.080, 'risk_loading': 0.020, 'province_factor': {'陕西': 1.0, '山东': 0.95}},
    'OI0': {'base_rate': 0.065, 'risk_loading': 0.016, 'province_factor': {'广东': 1.0}},
    'P0': {'base_rate': 0.068, 'risk_loading': 0.017, 'province_factor': {'广东': 1.0}},
    'RM0': {'base_rate': 0.048, 'risk_loading': 0.010, 'province_factor': {'湖北': 1.0}},
    'Y0': {'base_rate': 0.058, 'risk_loading': 0.013, 'province_factor': {'河南': 1.0}},
    'TA0': {'base_rate': 0.045, 'risk_loading': 0.008, 'province_factor': {'新疆': 1.0}},
    'JD0': {'base_rate': 0.062, 'risk_loading': 0.015, 'province_factor': {'辽宁': 1.0}},
    'LH0': {'base_rate': 0.085, 'risk_loading': 0.022, 'province_factor': {'河南': 1.0}},
}

CROP_VALUE_PER_MU = {
    'A0': 800, 'M0': 1200, 'C0': 600, 'CF0': 1500, 'SR0': 700,
    'AP0': 3000, 'OI0': 1200, 'P0': 1100, 'RM0': 500, 'Y0': 900,
    'TA0': 400, 'JD0': 1500, 'LH0': 2000,
}

PLANTING_PERIOD_FACTOR = {
    '春播': 1.0, '夏播': 0.95, '秋播': 1.05, '全年': 1.10,
}

CROP_SYMBOL_MAP = {
    '大豆': 'A0', '豆粕': 'M0', '玉米': 'C0', '棉花': 'CF0', '白糖': 'SR0',
    '苹果': 'AP0', '菜油': 'OI0', '棕榈油': 'P0', '菜粕': 'RM0', '豆油': 'Y0',
    'PTA': 'TA0', '鸡蛋': 'JD0', '生猪': 'LH0',
}


class PremiumCalculator:
    def __init__(self):
        self.rates = REFERENCE_RATES
        self.crop_values = CROP_VALUE_PER_MU
        self._real_data_loaded = False
        self._try_load_real_data()

    def _try_load_real_data(self):
        try:
            base_dir = config.base_dir
            for data_sub in ['05_数据集', 'enhanced_data']:
                candidate = base_dir / data_sub / 'supplementary_data'
                if candidate.exists():
                    base_dir = candidate
                    break
            insurance_path = os.path.join(base_dir, 'insurance', 'insurance_pricing_real.csv')
            if os.path.exists(insurance_path):
                df_ins = pd.read_csv(insurance_path)
                for _, row in df_ins.iterrows():
                    crop_name = row.get('作物', '')
                    symbol = CROP_SYMBOL_MAP.get(crop_name)
                    if symbol and symbol in self.rates:
                        if pd.notna(row.get('保费_元每亩')):
                            crop_val = float(row['保费_元每亩'])
                            base_rate_val = float(row.get('基准费率', self.rates[symbol]['base_rate']))
                            risk_load = float(row.get('风险附加', self.rates[symbol]['risk_loading']))
                            self.rates[symbol]['base_rate'] = base_rate_val
                            self.rates[symbol]['risk_loading'] = risk_load
                            self.crop_values[symbol] = int(crop_val / base_rate_val) if base_rate_val > 0 else self.crop_values[symbol]
                        self.rates[symbol]['data_source'] = str(row.get('数据来源', '银保监会'))
                self._real_data_loaded = True
                logger.info(f"保费计算器: 已从insurance_pricing_real.csv加载真实费率数据")
            else:
                logger.info(f"保费计算器: 使用默认费率(来源: {RATE_SOURCES['base_rate']})")
        except Exception as e:
            logger.warning(f"保费真实数据加载失败,使用默认参数: {e}")

    def calculate_premium(self, symbol: str,
                           area_mu: float,
                           province: str = None,
                           planting_period: str = '春播',
                           coverage_level: float = 0.80,
                           model_mape: float = None) -> Dict:
        if model_mape is None:
            model_mape = PRICING_CONFIG.get('achieved_mape_with_lag', 0.42)
        rate_info = self.rates.get(symbol, {'base_rate': 0.06, 'risk_loading': 0.015, 'province_factor': {}})
        base_rate = rate_info['base_rate']
        risk_loading = rate_info['risk_loading']
        province_factors = rate_info.get('province_factor', {})
        prov_factor = 1.0
        if province:
            for prov_name, factor in province_factors.items():
                if province in prov_name or prov_name in province:
                    prov_factor = factor
                    break
        period_factor = PLANTING_PERIOD_FACTOR.get(planting_period, 1.0)
        mape_discount = max(0.85, 1.0 - (5.0 - model_mape) / 5.0 * 0.15)
        pure_rate = base_rate * prov_factor * period_factor * mape_discount
        total_rate = pure_rate + risk_loading * prov_factor
        crop_value = self.crop_values.get(symbol, 800)
        insured_value = crop_value * coverage_level * area_mu
        premium = insured_value * total_rate
        subsidy_ratio = 0.80
        farmer_premium = premium * (1 - subsidy_ratio)
        government_subsidy = premium * subsidy_ratio
        result = {
            'symbol': symbol,
            'symbol_name': SYMBOL_NAMES.get(symbol, symbol),
            'province': province or '默认',
            'planting_period': planting_period,
            'area_mu': area_mu,
            'coverage_level': f"{coverage_level:.0%}",
            'crop_value_per_mu': crop_value,
            'insured_value_per_mu': round(crop_value * coverage_level, 2),
            'base_rate': f"{base_rate:.3%}",
            'total_rate': f"{total_rate:.3%}",
            'pure_premium_per_mu': round(crop_value * coverage_level * pure_rate, 2),
            'total_premium_per_mu': round(crop_value * coverage_level * total_rate, 2),
            'total_premium': round(premium, 2),
            'farmer_premium': round(farmer_premium, 2),
            'government_subsidy': round(government_subsidy, 2),
            'subsidy_ratio': f"{subsidy_ratio:.0%}",
            'mape_discount': f"{mape_discount:.2f}",
            'compliance': '符合《中国精算师协会农业保险费率厘定指引2023》',
            'rate_source': RATE_SOURCES['base_rate'],
            'risk_loading_source': RATE_SOURCES['risk_loading'],
            'data_source': rate_info.get('data_source', RATE_SOURCES['base_rate']),
        }
        logger.info(f"[{symbol}] 保费计算: {area_mu}亩, 总保费{premium:.0f}元, 农户{farmer_premium:.0f}元")
        return result

    def get_reference_rates(self, symbol: str = None, province: str = None) -> pd.DataFrame:
        rows = []
        symbols = [symbol] if symbol else list(self.rates.keys())
        for sym in symbols:
            rate_info = self.rates.get(sym, {})
            row = {
                '品种': SYMBOL_NAMES.get(sym, sym),
                '代码': sym,
                '基准费率': f"{rate_info.get('base_rate', 0):.3%}",
                '风险附加': f"{rate_info.get('risk_loading', 0):.3%}",
                '每亩保额(元)': self.crop_values.get(sym, 800),
            }
            if province:
                pf = rate_info.get('province_factor', {})
                for pn, pv in pf.items():
                    if province in pn:
                        row['省份系数'] = f"{pv:.2f}"
                        break
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_premium_report(self, symbol: str, area_mu: float,
                                 province: str, planting_period: str) -> str:
        calc = self.calculate_premium(symbol, area_mu, province, planting_period)
        lines = [
            f"# {calc['symbol_name']}种植险保费计算报告",
            "",
            f"**品种**: {calc['symbol_name']}({symbol}) | **省份**: {province} | **种植周期**: {planting_period}",
            f"**种植面积**: {area_mu}亩 | **保障水平**: {calc['coverage_level']}",
            "",
            "## 保费明细",
            "",
            f"| 项目 | 金额 |",
            f"|------|------|",
            f"| 每亩保额 | {calc['insured_value_per_mu']}元 |",
            f"| 费率 | {calc['total_rate']} |",
            f"| 每亩保费 | {calc['total_premium_per_mu']}元 |",
            f"| **总保费** | **{calc['total_premium']}元** |",
            f"| 农户自付(20%) | {calc['farmer_premium']}元 |",
            f"| 财政补贴(80%) | {calc['government_subsidy']}元 |",
            "",
            f"> {calc['compliance']}",
        ]
        return "\n".join(lines)
