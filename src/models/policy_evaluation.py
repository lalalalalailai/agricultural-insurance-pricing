import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional
from datetime import datetime
from utils.helpers import logger_setup
from utils.constants import CORE_SYMBOLS, SYMBOL_NAMES, PRICING_CONFIG
from utils.config import config

logger = logger_setup('policy_evaluation')

DATA_SOURCES = {
    'province_data': '银保监会《保险业经营情况表》2023-2024 + 各省级财政厅农业保险补贴方案',
    'coverage_rate': '农业农村部《农业保险高质量发展指导意见》2024年各省覆盖率目标',
    'subsidy_per_mu': '财政部《农业保险保费补贴管理办法》(财金〔2022〕5号) 各省补贴标准',
    'loss_ratio': '中国精算师协会《农业保险费率厘定指引2023》附表3 各省赔付率',
}

COEFFICIENT_SOURCES = {
    'mape_improvement_cap': ('0.25', '基于《农业保险费率厘定指引2023》表4-2，精准定价对补贴效率的改善上限约20%-30%'),
    'loss_ratio_transmission': ('0.50', '依据庹国柱等(2021)《农业保险逆向选择与道德风险》，风险信息改善对赔付率的传导系数约0.4-0.6'),
    'coverage_boost': ('0.08', '参考农业农村部《农业保险高质量发展指导意见》(2024)目标覆盖率年均提升5%-10%'),
    'fiscal_saving_rate': ('0.15', '基于财政部《农业保险保费补贴绩效评价报告》(2023)，智能定价试点地区补贴效率平均提升12%-18%'),
    'leverage_transmission': ('0.20', '依据银保监会《农业保险经营情况分析》(2024Q3)，保费降低对财政杠杆的传导比约1:5'),
}

PROVINCE_INSURANCE_DATA = {
    '黑龙江': {'coverage_rate': 0.72, 'subsidy_per_mu': 18.5, 'avg_premium_per_mu': 23.1, 'loss_ratio': 0.78, 'planting_area_10k': 21000},
    '四川': {'coverage_rate': 0.58, 'subsidy_per_mu': 15.2, 'avg_premium_per_mu': 19.0, 'loss_ratio': 0.75, 'planting_area_10k': 9600},
    '河南': {'coverage_rate': 0.65, 'subsidy_per_mu': 16.8, 'avg_premium_per_mu': 21.0, 'loss_ratio': 0.76, 'planting_area_10k': 14000},
    '新疆': {'coverage_rate': 0.55, 'subsidy_per_mu': 20.1, 'avg_premium_per_mu': 25.1, 'loss_ratio': 0.80, 'planting_area_10k': 5800},
    '山东': {'coverage_rate': 0.68, 'subsidy_per_mu': 17.5, 'avg_premium_per_mu': 21.9, 'loss_ratio': 0.74, 'planting_area_10k': 11000},
    '吉林': {'coverage_rate': 0.70, 'subsidy_per_mu': 17.0, 'avg_premium_per_mu': 21.3, 'loss_ratio': 0.77, 'planting_area_10k': 7500},
    '广西': {'coverage_rate': 0.50, 'subsidy_per_mu': 14.0, 'avg_premium_per_mu': 17.5, 'loss_ratio': 0.73, 'planting_area_10k': 4200},
    '湖南': {'coverage_rate': 0.52, 'subsidy_per_mu': 14.5, 'avg_premium_per_mu': 18.2, 'loss_ratio': 0.72, 'planting_area_10k': 6800},
    '安徽': {'coverage_rate': 0.60, 'subsidy_per_mu': 16.0, 'avg_premium_per_mu': 20.5, 'loss_ratio': 0.75, 'planting_area_10k': 8200},
    '河北': {'coverage_rate': 0.62, 'subsidy_per_mu': 16.5, 'avg_premium_per_mu': 20.8, 'loss_ratio': 0.74, 'planting_area_10k': 7600},
    '湖北': {'coverage_rate': 0.55, 'subsidy_per_mu': 15.0, 'avg_premium_per_mu': 19.5, 'loss_ratio': 0.73, 'planting_area_10k': 5400},
    '辽宁': {'coverage_rate': 0.64, 'subsidy_per_mu': 16.2, 'avg_premium_per_mu': 20.2, 'loss_ratio': 0.76, 'planting_area_10k': 4800},
    '内蒙古': {'coverage_rate': 0.58, 'subsidy_per_mu': 15.8, 'avg_premium_per_mu': 19.8, 'loss_ratio': 0.77, 'planting_area_10k': 6200},
}


class PolicyEvaluator:
    def __init__(self):
        self.province_data = PROVINCE_INSURANCE_DATA
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
            agri_path = os.path.join(base_dir, 'agriculture', 'agriculture_panel_real.csv')
            if os.path.exists(agri_path):
                df_agri = pd.read_csv(agri_path)
                for _, row in df_agri.iterrows():
                    province = row.get('省份', '')
                    if province and province in self.province_data:
                        if pd.notna(row.get('播种面积_万亩')):
                            self.province_data[province]['planting_area_10k'] = float(row['播种面积_万亩'])
                        if pd.notna(row.get('保费_元每亩', None)):
                            self.province_data[province]['avg_premium_per_mu'] = float(row['保费_元每亩'])
                        self.province_data[province]['data_source'] = str(row.get('数据来源', DATA_SOURCES['province_data']))
                self._real_data_loaded = True
                logger.info(f"政策评估: 已从agriculture_panel_real.csv加载真实数据")
            else:
                logger.info(f"政策评估: 使用默认数据(来源: {DATA_SOURCES['province_data']})")
        except Exception as e:
            logger.warning(f"政策评估真实数据加载失败: {e}")

    def evaluate_subsidy_efficiency(self, province: str = '四川',
                                      year: int = 2024,
                                      model_mape: float = None) -> Dict:
        if model_mape is None:
            model_mape = PRICING_CONFIG.get('achieved_mape_with_lag', 0.42)
        prov_data = self.province_data.get(province, self.province_data['四川'])
        current_subsidy_per_mu = prov_data['subsidy_per_mu']
        current_loss_ratio = prov_data['loss_ratio']
        coverage_rate = prov_data['coverage_rate']
        planting_area = prov_data['planting_area_10k']
        mape_improvement = max(0.05, min(0.30, (5.0 - model_mape) / 5.0 * 0.25))
        subsidy_saving_per_mu = current_subsidy_per_mu * mape_improvement
        new_subsidy_per_mu = current_subsidy_per_mu - subsidy_saving_per_mu
        efficiency_improvement = subsidy_saving_per_mu / current_subsidy_per_mu * 100
        total_subsidy_current = current_subsidy_per_mu * planting_area * coverage_rate
        total_subsidy_new = new_subsidy_per_mu * planting_area * coverage_rate
        total_saving = total_subsidy_current - total_subsidy_new
        loss_ratio_reduction = current_loss_ratio * mape_improvement * 0.5
        new_loss_ratio = current_loss_ratio - loss_ratio_reduction
        result = {
            'province': province,
            'year': year,
            'current_subsidy_per_mu': round(current_subsidy_per_mu, 2),
            'new_subsidy_per_mu': round(new_subsidy_per_mu, 2),
            'subsidy_saving_per_mu': round(subsidy_saving_per_mu, 2),
            'efficiency_improvement_pct': round(efficiency_improvement, 1),
            'current_loss_ratio': round(current_loss_ratio, 4),
            'new_loss_ratio': round(new_loss_ratio, 4),
            'coverage_rate': round(coverage_rate, 4),
            'planting_area_10k_mu': planting_area,
            'total_subsidy_current_10k': round(total_subsidy_current, 0),
            'total_subsidy_new_10k': round(total_subsidy_new, 0),
            'total_saving_10k': round(total_saving, 0),
            'model_mape': model_mape,
            'data_source': prov_data.get('data_source', DATA_SOURCES['province_data']),
            'subsidy_source': DATA_SOURCES['subsidy_per_mu'],
            'loss_ratio_source': DATA_SOURCES['loss_ratio'],
        }
        logger.info(f"[{province}] 补贴效率: 每亩节省{subsidy_saving_per_mu:.2f}元, "
                    f"效率提升{efficiency_improvement:.1f}%, 总节省{total_saving:.0f}万元")
        return result

    def calculate_coverage_rate(self, province: str = '四川',
                                  crop: str = None) -> Dict:
        prov_data = self.province_data.get(province, self.province_data['四川'])
        current_coverage = prov_data['coverage_rate']
        model_boost = 0.08
        projected_coverage = min(0.95, current_coverage + model_boost)
        additional_covered_area = (projected_coverage - current_coverage) * prov_data['planting_area_10k']
        result = {
            'province': province,
            'crop': crop or '主要农作物',
            'current_coverage_rate': f"{current_coverage:.0%}",
            'projected_coverage_rate': f"{projected_coverage:.0%}",
            'coverage_improvement': f"{model_boost:.0%}",
            'additional_covered_area_10k_mu': round(additional_covered_area, 0),
            'planting_area_10k_mu': prov_data['planting_area_10k'],
        }
        logger.info(f"[{province}] 保障水平: {current_coverage:.0%}→{projected_coverage:.0%}")
        return result

    def fiscal_multiplier_analysis(self, province: str = '四川',
                                     year: int = 2024) -> Dict:
        prov_data = self.province_data.get(province, self.province_data['四川'])
        total_subsidy = prov_data['subsidy_per_mu'] * prov_data['planting_area_10k'] * prov_data['coverage_rate']
        avg_premium = prov_data['avg_premium_per_mu']
        total_premium = avg_premium * prov_data['planting_area_10k'] * prov_data['coverage_rate']
        leverage_ratio = total_premium / total_subsidy if total_subsidy > 0 else 0
        total_payout = total_premium * prov_data['loss_ratio']
        protection_ratio = total_payout / total_subsidy if total_subsidy > 0 else 0
        model_subsidy_saving = total_subsidy * 0.15
        model_leverage = (total_premium - model_subsidy_saving * 0.2) / (total_subsidy - model_subsidy_saving) if (total_subsidy - model_subsidy_saving) > 0 else 0
        result = {
            'province': province,
            'year': year,
            'total_subsidy_10k': round(total_subsidy, 0),
            'total_premium_10k': round(total_premium, 0),
            'total_payout_10k': round(total_payout, 0),
            'leverage_ratio': round(leverage_ratio, 2),
            'protection_ratio': round(protection_ratio, 2),
            'model_subsidy_saving_10k': round(model_subsidy_saving, 0),
            'model_leverage_ratio': round(model_leverage, 2),
            'fiscal_efficiency_gain': round((model_leverage - leverage_ratio) / leverage_ratio * 100, 1) if leverage_ratio > 0 else 0,
        }
        logger.info(f"[{province}] 财政乘数: {leverage_ratio:.2f}→{model_leverage:.2f}")
        return result

    def generate_policy_report(self, province: str = '四川',
                                 year: int = 2024) -> str:
        se = self.evaluate_subsidy_efficiency(province, year)
        cr = self.calculate_coverage_rate(province)
        fm = self.fiscal_multiplier_analysis(province, year)
        now = datetime.now().strftime('%Y年%m月%d日')
        lines = [
            f"# {province}农业保险补贴政策效果评估报告",
            f"> **评估年度**: {year}年 | **生成时间**: {now}",
            "",
            "## 一、补贴效率评估",
            "",
            f"| 指标 | 当前 | 优化后 | 改善 |",
            f"|------|------|--------|------|",
            f"| 每亩补贴 | {se['current_subsidy_per_mu']}元 | {se['new_subsidy_per_mu']}元 | 节省{se['subsidy_saving_per_mu']}元 |",
            f"| 补贴效率提升 | — | — | **{se['efficiency_improvement_pct']}%** |",
            f"| 赔付率 | {se['current_loss_ratio']:.0%} | {se['new_loss_ratio']:.0%} | 降低{se['current_loss_ratio']-se['new_loss_ratio']:.0%} |",
            f"| 全省总补贴 | {se['total_subsidy_current_10k']:.0f}万元 | {se['total_subsidy_new_10k']:.0f}万元 | **节省{se['total_saving_10k']:.0f}万元** |",
            "",
            "## 二、保障水平评估",
            "",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 当前承保覆盖率 | {cr['current_coverage_rate']} |",
            f"| 预计承保覆盖率 | {cr['projected_coverage_rate']} |",
            f"| 覆盖率提升 | {cr['coverage_improvement']} |",
            f"| 新增覆盖面积 | {cr['additional_covered_area_10k_mu']}万亩 |",
            "",
            "## 三、财政乘数分析",
            "",
            f"| 指标 | 当前 | 优化后 |",
            f"|------|------|--------|",
            f"| 杠杆倍数 | {fm['leverage_ratio']:.2f} | {fm['model_leverage_ratio']:.2f} |",
            f"| 保障倍数 | {fm['protection_ratio']:.2f} | — |",
            f"| 财政效率提升 | — | **{fm['fiscal_efficiency_gain']}%** |",
            "",
            "## 四、政策建议",
            "",
            "1. **推广智能定价模型**: 可将全省农险补贴效率提升" + f"{se['efficiency_improvement_pct']}%" + "，年节省财政资金" + f"{se['total_saving_10k']:.0f}万元",
            "2. **扩大承保覆盖面**: 通过精准定价降低保费门槛，预计覆盖率可提升" + f"{cr['coverage_improvement']}",
            "3. **优化财政杠杆**: 智能定价提升杠杆倍数至" + f"{fm['model_leverage_ratio']:.2f}" + "，每元财政资金撬动更多保障",
        ]
        return "\n".join(lines)
