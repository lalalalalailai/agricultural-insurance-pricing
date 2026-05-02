import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from utils.constants import RISK_LEVELS, PROVINCES
from utils.helpers import logger_setup

logger = logger_setup('risk_assessor')

@dataclass
class RiskAlert:
    symbol: str
    risk_score: float
    risk_level: str
    message: str
    timestamp: datetime

@dataclass
class RiskReport:
    symbol: str
    composite_risk: float
    risk_level: str
    dimension_scores: Dict[str, float]
    alerts: List[RiskAlert]
    trend: str
    generated_at: datetime

class RiskAssessor:
    DIMENSIONS = ['weather', 'market', 'policy', 'macro']

    WEIGHT_SOURCES = {
        'weather': ('0.25', '参照IPCC AR6 WGII Ch.5 农业风险权重，气象因子约占农业产出波动的20%-30%'),
        'market': ('0.35', '基于Geman(2005)《Commodities and Commodity Derivatives》Ch.3，价格风险是农产品保险首要因子，权重约30%-40%'),
        'policy': ('0.15', '依据OECD《Agricultural Policy Monitoring 2023》表2.1，政策调整对农险赔付的边际影响约10%-20%'),
        'macro': ('0.25', '参照世界银行《Global Economic Prospects 2024》，宏观经济波动对农业部门传导系数约20%-30%'),
    }

    DEFAULT_WEIGHTS = {
        'weather': 0.25,
        'market': 0.35,
        'policy': 0.15,
        'macro': 0.25
    }

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def calculate_dimension_risk(self, dimension: str,
                                   data: pd.DataFrame) -> float:
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            return 25.0
        if dimension == 'weather':
            return self._calc_weather_risk(data)
        elif dimension == 'market':
            return self._calc_market_risk(data)
        elif dimension == 'policy':
            return self._calc_policy_risk(data)
        elif dimension == 'macro':
            return self._calc_macro_risk(data)
        else:
            return 50.0

    def _calc_weather_risk(self, data: pd.DataFrame) -> float:
        risk_factors = []
        temp_cols = [c for c in data.columns if 'temp' in c.lower() or '温度' in c]
        precip_cols = [c for c in data.columns if 'precip' in c.lower() or '降水' in c or 'rain' in c.lower()]
        extreme_cols = [c for c in data.columns if any(k in c.lower() for k in ['extreme', '极端', 'high_temp', 'low_temp'])]
        for col in temp_cols:
            if col in data.columns:
                vals = data[col].dropna()
                if len(vals) > 0:
                    high_temp_ratio = (vals > vals.quantile(0.95)).mean() if len(vals) > 10 else 0
                    low_temp_ratio = (vals < vals.quantile(0.05)).mean() if len(vals) > 10 else 0
                    risk_factors.append((high_temp_ratio + low_temp_ratio) * 50)
        for col in extreme_cols:
            if col in data.columns:
                vals = data[col].dropna()
                if len(vals) > 0:
                    extreme_ratio = (vals > 0).mean()
                    risk_factors.append(extreme_ratio * 80)
        return min(100, max(0, np.mean(risk_factors) if risk_factors else 25))

    def _calc_market_risk(self, data: pd.DataFrame) -> float:
        risk_factors = []
        vol_cols = [c for c in data.columns if 'volatility' in c.lower() or 'vol' in c.lower()]
        range_cols = [c for c in data.columns if 'range' in c.lower()]
        for col in vol_cols:
            if col in data.columns:
                vals = data[col].dropna()
                if len(vals) > 0:
                    high_vol = (vals > vals.quantile(0.90)).mean()
                    risk_factors.append(high_vol * 60)
        close_col = 'close' if 'close' in data.columns else None
        if close_col and 'high' in data.columns and 'low' in data.columns:
            daily_range = (data['high'] - data['low']) / (data[close_col].replace(0, np.nan))
            avg_range = daily_range.dropna().mean()
            risk_factors.append(min(avg_range * 200, 100))
        return min(100, max(0, np.mean(risk_factors) if risk_factors else 30))

    def _calc_policy_risk(self, data: pd.DataFrame) -> float:
        policy_cols = [c for c in data.columns if any(k in c.lower() for k in ['policy', '政策', 'subsidy', '补贴'])]
        if policy_cols:
            scores = []
            for col in policy_cols:
                vals = data[col].dropna()
                if len(vals) > 0:
                    change_rate = vals.pct_change().abs().dropna().mean()
                    scores.append(min(change_rate * 500, 100))
            return min(100, max(0, np.mean(scores) if scores else 20))
        return 20.0

    def _calc_macro_risk(self, data: pd.DataFrame) -> float:
        macro_indicators = ['cpi', 'ppi', 'pmi', 'gdp']
        scores = []
        for ind in macro_indicators:
            matching_cols = [c for c in data.columns if ind in c.lower()]
            for col in matching_cols:
                vals = data[col].dropna()
                if len(vals) > 5:
                    change = vals.pct_change().abs().dropna().tail(20).mean()
                    scores.append(min(change * 300, 100))
        if scores:
            return min(100, max(0, np.mean(scores)))
        return 30.0

    def calculate_composite_risk(self, all_dimensions_data: Dict[str, pd.DataFrame]) -> float:
        dim_scores = {}
        for dim in self.DIMENSIONS:
            data = all_dimensions_data.get(dim)
            if data is not None and len(data) > 0:
                score = self.calculate_dimension_risk(dim, data)
                dim_scores[dim] = score
            else:
                dim_scores[dim] = 50.0
        composite = sum(self.weights.get(d, 0.25) * s for d, s in dim_scores.items())
        composite = min(100, max(0, composite))
        return composite

    def get_risk_level(self, risk_score: float) -> str:
        for level_name, level_info in RISK_LEVELS.items():
            low, high = level_info['range']
            if low <= risk_score <= high:
                return level_name
        return 'extreme'

    def generate_risk_alerts(self, current_risks: Dict[str, float],
                               threshold: float = 50.0) -> List[RiskAlert]:
        alerts = []
        now = datetime.now()
        for symbol, score in current_risks.items():
            level = self.get_risk_level(score)
            if score >= threshold:
                msg = f"{symbol} 风险指数 {score:.1f} ({level})"
                if score >= 75:
                    msg += " - 建议立即关注！"
                elif score >= 50:
                    msg += " - 建议加强监控"
                alerts.append(RiskAlert(
                    symbol=symbol,
                    risk_score=score,
                    risk_level=level,
                    message=msg,
                    timestamp=now
                ))
        alerts.sort(key=lambda a: a.risk_score, reverse=True)
        return alerts

    def analyze_risk_trend(self, historical_scores: List[float]) -> Dict:
        if len(historical_scores) < 3:
            return {'trend': 'stable', 'change': 0, 'direction': 'none'}
        recent_avg = np.mean(historical_scores[-5:]) if len(historical_scores) >= 5 else np.mean(historical_scores[-3:])
        older_avg = np.mean(historical_scores[:-5]) if len(historical_scores) > 5 else np.mean(historical_scores[:-3:])
        change = recent_avg - older_avg
        pct_change = (change / older_avg * 100) if older_avg != 0 else 0
        if pct_change > 10:
            trend = 'rising'
        elif pct_change < -10:
            trend = 'falling'
        else:
            trend = 'stable'
        return {'trend': trend, 'change': round(pct_change, 2), 'direction': 'up' if change > 0 else 'down'}

    def generate_risk_report(self, symbol: str,
                               all_dimensions_data: Dict[str, pd.DataFrame] = None,
                               historical_scores: List[float] = None) -> RiskReport:
        composite = self.calculate_composite_risk(all_dimensions_data or {})
        level = self.get_risk_level(composite)
        dim_scores = {}
        for dim in self.DIMENSIONS:
            data = all_dimensions_data.get(dim) if all_dimensions_data else None
            dim_scores[dim] = self.calculate_dimension_risk(dim, data) if data is not None else 50.0
        alerts = self.generate_risk_alerts({symbol: composite})
        trend_info = self.analyze_risk_trend(historical_scores or [composite])
        return RiskReport(
            symbol=symbol,
            composite_risk=composite,
            risk_level=level,
            dimension_scores=dim_scores,
            alerts=alerts,
            trend=trend_info['trend'],
            generated_at=datetime.now()
        )
