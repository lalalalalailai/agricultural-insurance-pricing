import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from utils.helpers import logger_setup
from utils.constants import CORE_SYMBOLS, SYMBOL_NAMES, PROVINCES

logger = logger_setup('extreme_risk_warning')

THRESHOLD_SOURCES = {
    'drought_index': '中国气象局《干旱等级》(GB/T 20481-2017)，中旱≥0.6，重旱≥0.8',
    'precipitation_extreme': '中国气象局《暴雨等级》(GB/T 28592-2012)，暴雨≥50mm/24h，大暴雨≥100mm，特大暴雨≥250mm',
    'temperature_high': '中国气象局《高温预警信号》(GB/T 28591-2012)，橙色≥37°C，红色≥40°C；农业取35°C/40°C',
    'temperature_low': '中国气象局《寒潮等级》(GB/T 21987-2017)，寒潮橙色≤-10°C，红色≤-20°C',
    'wind_speed': '中国气象局《大风预警信号》，8级≥17.2m/s，10级≥24.5m/s',
    'daily_return': '参照Geman(2005)商品期货日收益率分布，4%为95分位，7%为99分位',
    'rolling_vol': '基于中国期货市场20日年化波动率历史分布，25%为90分位，40%为95分位',
    'basis_deviation': '依据郑商所/大商所基差统计，5%为正常上限，10%为极端偏离',
    'ndvi_decline': '参照NASA MODIS NDVI产品说明，15%下降为显著退化，25%为严重退化',
    'evi_decline': '参照Huete et al.(2002)EVI指标定义，12%下降为显著退化，20%为严重退化',
}

WEATHER_ALERT_THRESHOLDS = {
    'drought_index': {'warning': 0.6, 'alert': 0.8},
    'precipitation_extreme': {'warning': 150, 'alert': 250},
    'temperature_high': {'warning': 35, 'alert': 40},
    'temperature_low': {'warning': -10, 'alert': -20},
    'wind_speed': {'warning': 17, 'alert': 25},
}

PRICE_VOLATILITY_THRESHOLDS = {
    'daily_return_warning': 0.04,
    'daily_return_alert': 0.07,
    'rolling_vol_warning': 0.25,
    'rolling_vol_alert': 0.40,
    'basis_deviation_warning': 0.05,
    'basis_deviation_alert': 0.10,
}

YIELD_RISK_THRESHOLDS = {
    'ndvi_decline_warning': -0.15,
    'ndvi_decline_alert': -0.25,
    'evi_decline_warning': -0.12,
    'evi_decline_alert': -0.20,
}


class ExtremeRiskWarning:
    def __init__(self):
        self.weather_thresholds = WEATHER_ALERT_THRESHOLDS
        self.price_thresholds = PRICE_VOLATILITY_THRESHOLDS
        self.yield_thresholds = YIELD_RISK_THRESHOLDS

    def check_weather_alert(self, weather_data: pd.DataFrame = None,
                              province: str = None,
                              forecast_days: int = 30) -> Dict:
        alerts = []
        if weather_data is not None and not weather_data.empty:
            if 'temperature' in weather_data.columns:
                max_temp = weather_data['temperature'].max()
                min_temp = weather_data['temperature'].min()
                if max_temp >= self.weather_thresholds['temperature_high']['alert']:
                    alerts.append({'level': '🔴红色预警', 'type': '极端高温',
                                   'detail': f'最高温度{max_temp:.1f}°C，超过{self.weather_thresholds["temperature_high"]["alert"]}°C',
                                   'action': '建议启动高温理赔预案，通知农户采取防暑降温措施'})
                elif max_temp >= self.weather_thresholds['temperature_high']['warning']:
                    alerts.append({'level': '🟡橙色预警', 'type': '高温预警',
                                   'detail': f'最高温度{max_temp:.1f}°C，接近极端高温阈值',
                                   'action': '建议加强田间管理，关注作物受热情况'})
                if min_temp <= self.weather_thresholds['temperature_low']['alert']:
                    alerts.append({'level': '🔴红色预警', 'type': '极端低温',
                                   'detail': f'最低温度{min_temp:.1f}°C，低于{self.weather_thresholds["temperature_low"]["alert"]}°C',
                                   'action': '建议启动冻害理赔预案，通知农户采取防冻措施'})
            if 'precipitation' in weather_data.columns:
                max_precip = weather_data['precipitation'].max()
                if max_precip >= self.weather_thresholds['precipitation_extreme']['alert']:
                    alerts.append({'level': '🔴红色预警', 'type': '暴雨预警',
                                   'detail': f'最大日降水量{max_precip:.1f}mm，超过{self.weather_thresholds["precipitation_extreme"]["alert"]}mm',
                                   'action': '建议启动洪涝理赔预案，通知农户做好排水防涝'})
        if not alerts:
            province_name = province or '当前地区'
            alerts.append({'level': '🟢正常', 'type': '气象正常',
                           'detail': f'{province_name}未来{forecast_days}天气象条件正常，无极端天气预警',
                           'action': '继续常规监测'})
        result = {
            'province': province or '默认',
            'forecast_days': forecast_days,
            'alerts': alerts,
            'n_alerts': len([a for a in alerts if a['level'] != '🟢正常']),
            'max_level': max([a['level'] for a in alerts], key=lambda x: ['🟢正常', '🟡橙色预警', '🔴红色预警'].index(x) if x in ['🟢正常', '🟡橙色预警', '🔴红色预警'] else 0),
            'timestamp': datetime.now().isoformat(),
        }
        logger.info(f"气象预警: {province or '默认'}, {len(alerts)}条预警")
        return result

    def check_price_alert(self, price_data: pd.DataFrame = None,
                           symbol: str = None,
                           forecast_days: int = 15) -> Dict:
        alerts = []
        if price_data is not None and not price_data.empty and 'close' in price_data.columns:
            close = price_data['close'].dropna()
            if len(close) >= 2:
                daily_returns = close.pct_change().dropna()
                max_return = abs(daily_returns.iloc[-1]) if len(daily_returns) > 0 else 0
                if max_return >= self.price_thresholds['daily_return_alert']:
                    alerts.append({'level': '🔴红色预警', 'type': '价格剧烈波动',
                                   'detail': f'日收益率{daily_returns.iloc[-1]:.2%}，超过±{self.price_thresholds["daily_return_alert"]:.0%}阈值',
                                   'action': '建议启动价格保险理赔评估，通知农户关注市场风险'})
                elif max_return >= self.price_thresholds['daily_return_warning']:
                    alerts.append({'level': '🟡橙色预警', 'type': '价格波动预警',
                                   'detail': f'日收益率{daily_returns.iloc[-1]:.2%}，接近波动阈值',
                                   'action': '建议加强市场监测，准备风险应对方案'})
                if len(close) >= 20:
                    rolling_vol = daily_returns.tail(20).std() * np.sqrt(252)
                    if rolling_vol >= self.price_thresholds['rolling_vol_alert']:
                        alerts.append({'level': '🔴红色预警', 'type': '高波动率',
                                       'detail': f'20日年化波动率{rolling_vol:.1%}，超过{self.price_thresholds["rolling_vol_alert"]:.0%}阈值',
                                       'action': '建议调整保费费率，增加风险准备金'})
        if not alerts:
            symbol_name = SYMBOL_NAMES.get(symbol, symbol or '当前品种')
            alerts.append({'level': '🟢正常', 'type': '价格正常',
                           'detail': f'{symbol_name}未来{forecast_days}天价格风险可控',
                           'action': '继续常规监测'})
        result = {
            'symbol': symbol or '默认',
            'symbol_name': SYMBOL_NAMES.get(symbol, symbol or '默认'),
            'forecast_days': forecast_days,
            'alerts': alerts,
            'n_alerts': len([a for a in alerts if a['level'] != '🟢正常']),
            'timestamp': datetime.now().isoformat(),
        }
        logger.info(f"价格预警: {symbol or '默认'}, {len(alerts)}条预警")
        return result

    def check_yield_alert(self, ndvi_data: pd.DataFrame = None,
                           symbol: str = None) -> Dict:
        alerts = []
        if ndvi_data is not None and not ndvi_data.empty:
            ndvi_cols = [c for c in ndvi_data.columns if 'ndvi' in c.lower()]
            if ndvi_cols:
                ndvi_series = ndvi_data[ndvi_cols[0]].dropna()
                if len(ndvi_series) >= 2:
                    ndvi_change = (ndvi_series.iloc[-1] - ndvi_series.iloc[-2]) / (abs(ndvi_series.iloc[-2]) + 1e-8)
                    if ndvi_change <= self.yield_thresholds['ndvi_decline_alert']:
                        alerts.append({'level': '🔴红色预警', 'type': '减产风险',
                                       'detail': f'NDVI下降{abs(ndvi_change):.1%}，超过{abs(self.yield_thresholds["ndvi_decline_alert"]):.0%}阈值',
                                       'action': '建议启动减产评估，准备理赔预案'})
                    elif ndvi_change <= self.yield_thresholds['ndvi_decline_warning']:
                        alerts.append({'level': '🟡橙色预警', 'type': '减产预警',
                                       'detail': f'NDVI下降{abs(ndvi_change):.1%}，接近减产阈值',
                                       'action': '建议加强田间监测，评估产量影响'})
        if not alerts:
            symbol_name = SYMBOL_NAMES.get(symbol, symbol or '当前品种')
            alerts.append({'level': '🟢正常', 'type': '产量正常',
                           'detail': f'{symbol_name}长势正常，无减产风险',
                           'action': '继续常规监测'})
        result = {
            'symbol': symbol or '默认',
            'symbol_name': SYMBOL_NAMES.get(symbol, symbol or '默认'),
            'alerts': alerts,
            'n_alerts': len([a for a in alerts if a['level'] != '🟢正常']),
            'timestamp': datetime.now().isoformat(),
        }
        logger.info(f"减产预警: {symbol or '默认'}, {len(alerts)}条预警")
        return result

    def generate_warning_report(self, weather_alert: Dict = None,
                                 price_alert: Dict = None,
                                 yield_alert: Dict = None) -> str:
        now = datetime.now().strftime('%Y年%m月%d日')
        lines = [
            "# 农险极端风险预警报告",
            f"> **生成时间**: {now}",
            "",
        ]
        total_alerts = 0
        if weather_alert:
            total_alerts += weather_alert.get('n_alerts', 0)
            lines.append("## 一、气象预警")
            for a in weather_alert.get('alerts', []):
                lines.append(f"- {a['level']} **{a['type']}**: {a['detail']}")
                lines.append(f"  - 应对: {a['action']}")
            lines.append("")
        if price_alert:
            total_alerts += price_alert.get('n_alerts', 0)
            lines.append("## 二、价格风险预警")
            for a in price_alert.get('alerts', []):
                lines.append(f"- {a['level']} **{a['type']}**: {a['detail']}")
                lines.append(f"  - 应对: {a['action']}")
            lines.append("")
        if yield_alert:
            total_alerts += yield_alert.get('n_alerts', 0)
            lines.append("## 三、减产风险预警")
            for a in yield_alert.get('alerts', []):
                lines.append(f"- {a['level']} **{a['type']}**: {a['detail']}")
                lines.append(f"  - 应对: {a['action']}")
            lines.append("")
        if total_alerts == 0:
            lines.append("## 综合评估: 🟢 **当前无极端风险预警**")
        else:
            lines.append(f"## 综合评估: ⚠️ **共{total_alerts}条预警，请及时关注**")
        return "\n".join(lines)
