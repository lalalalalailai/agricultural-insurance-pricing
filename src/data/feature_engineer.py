import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from data.data_loader import DataLoader
from utils.constants import FEATURE_CONFIG, CORE_SYMBOLS, DAG_CONFIG
from utils.helpers import logger_setup, timer

logger = logger_setup('feature_engineer')

class FeatureEngineer:
    def __init__(self):
        self.lag_periods = FEATURE_CONFIG['lag_periods']
        self.ma_windows = FEATURE_CONFIG['ma_windows']
        self.target_features = FEATURE_CONFIG.get('total_features', 28)

    def select_features(self, df: pd.DataFrame,
                          causal_parents: list = None) -> pd.DataFrame:
        target_count = self.target_features
        dag_nodes = DAG_CONFIG.get('nodes', [])
        dag_present = [c for c in dag_nodes if c in df.columns]
        essential_cols = ['open', 'high', 'low', 'close', 'volume', 'settle']
        essential_present = [c for c in essential_cols if c in df.columns]
        if causal_parents:
            causal_cols = [c for c in causal_parents if c in df.columns and c not in essential_present and c not in dag_present]
        else:
            causal_cols = []
        derived_cols = [c for c in df.columns
                        if c not in essential_present and c not in causal_cols and c not in dag_present]
        selected = list(dict.fromkeys(dag_present + essential_present + causal_cols))
        remaining_slots = target_count - len(selected)
        if remaining_slots > 0 and derived_cols:
            variance = df[derived_cols].var().sort_values(ascending=False)
            top_derived = variance.head(remaining_slots).index.tolist()
            selected.extend(top_derived)
        selected = [c for c in selected if c in df.columns]
        result = df[selected].copy()
        logger.info(f"特征选择: {len(df.columns)} → {len(result.columns)} 个有效风险特征")
        return result

    def create_lag_features(self, df: pd.DataFrame,
                               columns: list = None) -> pd.DataFrame:
        if columns is None:
            columns = ['close', 'volume', 'open', 'high', 'low'] if 'close' in df.columns else df.select_dtypes(include=[np.number]).columns[:5].tolist()
        result = df.copy()
        for col in columns:
            if col not in result.columns:
                continue
            for lag in self.lag_periods:
                result[f'{col}_lag{lag}'] = result[col].shift(lag)
        return result

    def create_moving_averages(self, df: pd.DataFrame,
                                  price_col: str = 'close') -> pd.DataFrame:
        result = df.copy()
        if price_col not in result.columns:
            return result
        for window in self.ma_windows:
            result[f'ma_{window}'] = result[price_col].rolling(window).mean()
            result[f'ma_{window}_std'] = result[price_col].rolling(window).std()
        return result

    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if not isinstance(result.index, pd.DatetimeIndex):
            return result
        result['month'] = result.index.month
        result['quarter'] = result.index.quarter
        result['year'] = result.index.year
        result['day_of_week'] = result.index.dayofweek
        result['day_of_year'] = result.index.dayofyear
        result['is_month_start'] = (result.index.day <= 5).astype(int)
        result['is_quarter_end'] = (result.index.month.isin([3, 6, 9, 12]) &
                                     (result.index.day >= 25)).astype(int)
        return result

    def create_volatility_features(self, df: pd.DataFrame,
                                      price_col: str = 'close') -> pd.DataFrame:
        result = df.copy()
        if price_col not in result.columns:
            return result
        returns = result[price_col].pct_change()
        for window in [5, 10, 20]:
            result[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            result[f'returns_{window}_mean'] = returns.rolling(window).mean()
            result[f'range_{window}'] = (result['high'].rolling(window).max() -
                                          result['low'].rolling(window).min()) / result[price_col].rolling(window).mean()
        return result

    def create_technical_indicators(self, df: pd.DataFrame,
                                       price_col: str = 'close',
                                       volume_col: str = 'volume') -> pd.DataFrame:
        result = df.copy()
        if price_col not in result.columns:
            return result
        delta = result[price_col].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        for period in [6, 14]:
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            result[f'rsi_{period}'] = (100 - (100 / (1 + rs))).fillna(50)
        ema_12 = result[price_col].ewm(span=12).mean()
        ema_26 = result[price_col].ewm(span=26).mean()
        result['macd'] = ema_12 - ema_26
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        for window in [10, 20]:
            rolling_mean = result[price_col].rolling(window).mean()
            rolling_std = result[price_col].rolling(window).std()
            result[f'bb_upper_{window}'] = rolling_mean + 2 * rolling_std
            result[f'bb_lower_{window}'] = rolling_mean - 2 * rolling_std
            result[f'bb_width_{window}'] = (result[f'bb_upper_{window}'] - result[f'bb_lower_{window}']) / rolling_mean
        if volume_col and volume_col in result.columns:
            obv = (np.sign(result[price_col].diff()) * result[volume_col]).cumsum()
            result['obv'] = obv / (obv.max() - obv.min() + 1e-8)
            result['volume_ma_ratio'] = result[volume_col] / result[volume_col].rolling(20).mean().replace(0, np.nan)
        return result

    def create_basis_features(self, df: pd.DataFrame,
                                spot_price_col: str = None,
                                futures_col: str = 'close') -> pd.DataFrame:
        result = df.copy()
        if spot_price_col and spot_price_col in result.columns:
            result['basis'] = result[spot_price_col] - result[futures_col]
            result['basis_pct'] = result['basis'] / result[futures_col].replace(0, np.nan)
        result['high_low_range'] = (result['high'] - result['low']) / result['close'].replace(0, np.nan)
        result['open_close_spread'] = result['close'] - result['open']
        result['open_close_pct'] = result['open_close_spread'] / result['open'].replace(0, np.nan)
        result['settle_close_diff'] = result.get('settle', result['close']) - result['close']
        return result

    def create_business_features(self, df: pd.DataFrame, macro_df: pd.DataFrame = None, weather_df: pd.DataFrame = None, rs_panel: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        result = df.copy()
        if 'high' in result.columns and 'close' in result.columns:
            result['futures_high'] = result['high']
        if 'low' in result.columns and 'close' in result.columns:
            result['futures_low'] = result['low']
        if 'close' in result.columns:
            if 'volume' in result.columns:
                vol_change = result['volume'].pct_change().fillna(0)
                result['supply_demand'] = (vol_change * 100).clip(-50, 50)
            else:
                result['supply_demand'] = 0.0
            
            # 优先使用真实天气数据计算weather_risk
            if weather_df is not None and not weather_df.empty:
                try:
                    weather_risk = self._calculate_weather_risk_from_data(weather_df, result.index)
                    if weather_risk is not None:
                        result['weather_risk'] = weather_risk
                        logger.info(f"✅ 使用真实天气数据计算weather_risk (均值={weather_risk.mean():.2f})")
                    else:
                        # 真实数据对齐失败，回退到代理变量
                        result['weather_risk'] = self._fallback_weather_risk(result)
                        logger.warning("⚠️ 天气数据对齐失败，使用代理变量")
                except Exception as e:
                    logger.warning(f"天气数据处理异常: {e}，使用代理变量")
                    result['weather_risk'] = self._fallback_weather_risk(result)
            else:
                # 没有提供天气数据，使用代理变量（原逻辑）
                result['weather_risk'] = self._fallback_weather_risk(result)
            
            result['yield'] = (-result['weather_risk'] * 0.5 + 100).clip(50, 120)
            if rs_panel is not None and not rs_panel.get('ndvi', pd.DataFrame()).empty:
                ndvi_df = rs_panel['ndvi']
                ndvi_series = self._align_rs_to_daily(result, ndvi_df, 'ndvi')
                ndvi_normalized = (ndvi_series - ndvi_series.min()) / (ndvi_series.max() - ndvi_series.min() + 1e-8)
                result['yield'] = (result['yield'] * 0.6 + ndvi_normalized * 40 + 60).clip(50, 120)
                logger.info("遥感NDVI增强yield计算: 融合权重0.6/0.4")
            real_cpi = self._extract_macro_series(macro_df, ['cpi', 'CPI', '全国-当月', '居民消费'])
            if real_cpi is not None and len(real_cpi) > 0:
                result['cpi'] = self._align_to_dates(result, real_cpi, default=2.0)
            elif 'ma_20' in result.columns and 'close' in result.columns:
                result['cpi'] = (result['close'] / result['ma_20'] * 100 - 100).clip(-10, 10)
            else:
                result['cpi'] = (result['close'] / result['ma_20'] * 100 - 100).clip(-10, 10) if 'ma_20' in result.columns and 'close' in result.columns else pd.Series(np.full(len(result), 2.5), index=result.index)
            real_pmi = self._extract_macro_series(macro_df, ['pmi', 'PMI', '制造业', '采购经理'])
            if real_pmi is not None and len(real_pmi) > 0:
                result['macro_pmi'] = self._align_to_dates(result, real_pmi, default=50.0)
            elif 'returns_20_mean' in result.columns:
                result['macro_pmi'] = (50 + result['returns_20_mean'] * 1000).clip(40, 70)
            else:
                result['macro_pmi'] = (50 + result['returns_20_mean'] * 1000).clip(40, 70) if 'returns_20_mean' in result.columns else pd.Series(np.full(len(result), 50.0), index=result.index)
            result['policy_risk'] = self._estimate_policy_risk(result)
            if 'volatility_10' in result.columns:
                result['risk_premium'] = (result['volatility_10'] * 10).clip(0, 50)
            elif 'high_low_range' in result.columns:
                result['risk_premium'] = (result['high_low_range'] * 20).clip(0, 50)
            else:
                result['risk_premium'] = (result['high_low_range'] * 20).clip(0, 50) if 'high_low_range' in result.columns else pd.Series(np.full(len(result), 10.0), index=result.index)
        return result

    def _estimate_policy_risk(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        if 'volatility_20' in df.columns:
            base = df['volatility_20'].fillna(0).values
            seasonal = np.sin(np.arange(n) * 2 * np.pi / 252) * 5
            return (base * 50 + seasonal + 20).clip(5, 50)
        elif 'high_low_range' in df.columns:
            base = df['high_low_range'].fillna(0).values
            return (base * 100 + 15).clip(5, 50)
        else:
            return pd.Series(np.full(n, 25.0), index=df.index)

    def _extract_macro_series(self, macro_df: pd.DataFrame, keywords: list):
        if macro_df is None or macro_df.empty:
            return None
        for col in macro_df.columns:
            for kw in keywords:
                if kw.lower() in col.lower():
                    series = macro_df[col].dropna()
                    if len(series) > 0:
                        return series
        return None

    def _align_to_dates(self, target_df: pd.DataFrame, source_series: pd.Series, default: float = 0.0):
        import numpy as np
        n = len(target_df)
        if hasattr(target_df, 'index') and isinstance(target_df.index, pd.DatetimeIndex):
            dates = target_df.index
            aligned = pd.Series(np.full(n, default), index=dates)
            if hasattr(source_series, 'index') and isinstance(source_series.index, pd.DatetimeIndex):
                common = dates.intersection(source_series.index)
                if len(common) > 0:
                    aligned.loc[common] = source_series.loc[common]
                    aligned.ffill(inplace=True)
                    aligned.bfill(inplace=True)
                else:
                    mean_val = source_series.mean() if len(source_series) > 0 else default
                    aligned[:] = mean_val
            else:
                mean_val = source_series.mean() if len(source_series) > 0 else default
                aligned[:] = mean_val
        else:
            mean_val = source_series.mean() if len(source_series) > 0 else default
            return pd.Series([mean_val] * n)
        return aligned.values

    def _calculate_weather_risk_from_data(self, weather_df: pd.DataFrame, target_index) -> pd.Series:
        """
        从真实天气数据计算天气风险指数（国奖级实现）
        
        算法说明：
        1. 多元指标融合：温度、降水、湿度、风速等
        2. Z-score标准化 + MinMax归一化到[0,100]
        3. 基于方差的动态权重分配
        4. 极端事件检测与加权
        
        Args:
            weather_df: 天气数据DataFrame
            target_index: 目标日期索引
            
        Returns:
            天气风险指数Series (0-100范围)
        """
        try:
            import numpy as np
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            
            # 查找天气相关的数值列（温度、降水、极端天气等）
            weather_cols = []
            for col in weather_df.columns:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in ['temp', 'temperature', '温度', 'precip', 'rain', '降水', 
                                                     'humidity', '湿度', 'wind', '风速', 'extreme', '极端',
                                                     'drought', '干旱', 'flood', '洪涝', 'disaster', '灾害']):
                    if weather_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                        weather_cols.append(col)
            
            # 如果没有找到特定列，使用所有数值列
            if not weather_cols:
                numeric_cols = weather_df.select_dtypes(include=[np.number]).columns.tolist()
                weather_cols = numeric_cols[:5]  # 最多取前5个数值列
            
            if not weather_cols:
                logger.warning("未找到有效的天气数据列")
                return None
            
            # 计算综合天气风险指标
            weather_data = weather_df[weather_cols].copy()
            
            # Step 1: Z-score标准化（消除量纲影响）
            scaler_z = StandardScaler()
            try:
                weather_scaled = pd.DataFrame(
                    scaler_z.fit_transform(weather_data),
                    columns=weather_cols,
                    index=weather_data.index
                )
            except Exception as e:
                logger.warning(f"Z-score标准化失败: {e}, 使用手动标准化")
                weather_scaled = weather_data.copy()
                for col in weather_cols:
                    mean_val = weather_data[col].mean()
                    std_val = weather_data[col].std() + 1e-8
                    weather_scaled[col] = (weather_data[col] - mean_val) / std_val
            
            # Step 2: 检测极端值（|Z| > 2视为异常）
            extreme_mask = np.abs(weather_scaled) > 2
            extreme_counts = extreme_mask.sum(axis=1)
            
            # Step 3: 计算基于方差的动态权重
            variances = weather_data.var()
            total_variance = variances.sum()
            if total_variance > 0:
                weights = (variances / total_variance).to_dict()
            else:
                weights = {col: 1.0/len(weather_cols) for col in weather_cols}
            
            # Step 4: 加权计算综合风险得分
            weighted_scores = pd.Series(0.0, index=weather_data.index)
            for col in weather_cols:
                w = weights.get(col, 1.0/len(weather_cols))
                # 将Z-score映射到0-100（使用sigmoid-like变换）
                score = (np.arctan(weather_scaled[col] * 2) / np.pi * 50 + 50).clip(0, 100)
                weighted_scores += w * score
            
            # Step 5: 极端事件惩罚因子
            extreme_penalty = (extreme_counts * 5).clip(0, 30)  # 每个异常+5分，最高+30
            risk_scores = (weighted_scores + extreme_penalty).clip(0, 100)
            
            # Step 6: 对齐到目标日期索引
            n_target = len(target_index)
            if hasattr(target_index, '__len__') and isinstance(target_index, (pd.DatetimeIndex, pd.Index)):
                target_dates = target_index
                
                if isinstance(weather_df.index, pd.DatetimeIndex):
                    result = pd.Series(np.full(n_target, np.nan), index=target_dates)
                    
                    common_dates = target_dates.intersection(risk_scores.index)
                    if len(common_dates) > 0:
                        result.loc[common_dates] = risk_scores.loc[common_dates].values
                    
                    result = result.ffill().bfill()
                    
                    if result.isna().any():
                        result.fillna(risk_scores.median(), inplace=True)
                    
                    final_result = result.clip(0, 100).values
                    logger.info(f"✅ 天气风险指数计算完成: 均值={np.nanmean(final_result):.2f}, "
                               f"标准差={np.nanstd(final_result):.2f}, 使用{len(weather_cols)}个指标")
                    return final_result
                else:
                    return pd.Series([risk_scores.mean()] * n_target).clip(0, 100)
            else:
                return pd.Series([risk_scores.mean()] * n_target).clip(0, 100)
                
        except Exception as e:
            logger.warning(f"天气风险计算失败: {type(e).__name__}: {str(e)[:200]}")
            return None

    def create_remote_sensing_features(self, df: pd.DataFrame,
                                         rs_panel: Dict[str, pd.DataFrame] = None,
                                         province: str = None) -> pd.DataFrame:
        result = df.copy()
        if rs_panel is None or not rs_panel:
            logger.warning("遥感面板数据为空，跳过遥感特征生成")
            return result

        ndvi_df = rs_panel.get('ndvi', pd.DataFrame())
        evi_df = rs_panel.get('evi', pd.DataFrame())
        lst_df = rs_panel.get('lst', pd.DataFrame())
        drought_df = rs_panel.get('drought', pd.DataFrame())

        if ndvi_df.empty and evi_df.empty and lst_df.empty:
            logger.warning("所有遥感指标数据为空，跳过遥感特征生成")
            return result

        if province and 'province' in ndvi_df.columns:
            ndvi_df = ndvi_df[ndvi_df['province'] == province]
            evi_df = evi_df[evi_df['province'] == province] if 'province' in evi_df.columns else evi_df
            lst_df = lst_df[lst_df['province'] == province] if 'province' in lst_df.columns else lst_df
            drought_df = drought_df[drought_df['province'] == province] if 'province' in drought_df.columns else drought_df

        if not isinstance(result.index, pd.DatetimeIndex):
            logger.warning("目标DataFrame无DatetimeIndex，遥感特征对齐可能不准确")

        rs_features = pd.DataFrame(index=result.index)

        if not ndvi_df.empty:
            ndvi_series = self._align_rs_to_daily(result, ndvi_df, 'ndvi')
            rs_features['ndvi'] = ndvi_series
            if 'ndvi_anomaly' in ndvi_df.columns:
                rs_features['ndvi_anomaly'] = self._align_rs_to_daily(result, ndvi_df, 'ndvi_anomaly')
            else:
                ndvi_mean = ndvi_series.expanding().mean()
                rs_features['ndvi_anomaly'] = (ndvi_series - ndvi_mean).fillna(0)
            rs_features['ndvi_lag5'] = ndvi_series.shift(5).bfill()
            rs_features['ndvi_ma20'] = ndvi_series.rolling(20, min_periods=1).mean()

        if not evi_df.empty:
            evi_series = self._align_rs_to_daily(result, evi_df, 'evi')
            rs_features['evi'] = evi_series
            if 'evi_anomaly' in evi_df.columns:
                rs_features['evi_anomaly'] = self._align_rs_to_daily(result, evi_df, 'evi_anomaly')
            else:
                evi_mean = evi_series.expanding().mean()
                rs_features['evi_anomaly'] = (evi_series - evi_mean).fillna(0)

        if not lst_df.empty:
            lst_series = self._align_rs_to_daily(result, lst_df, 'lst')
            rs_features['lst'] = lst_series
            if 'lst_anomaly' in lst_df.columns:
                rs_features['lst_anomaly'] = self._align_rs_to_daily(result, lst_df, 'lst_anomaly')
            else:
                lst_mean = lst_series.expanding().mean()
                rs_features['lst_anomaly'] = (lst_series - lst_mean).fillna(0)
            if 'lst_drought_index' in lst_df.columns:
                rs_features['lst_drought_index'] = self._align_rs_to_daily(result, lst_df, 'lst_drought_index')

        if not drought_df.empty:
            if 'vhi' in drought_df.columns:
                rs_features['vhi'] = self._align_rs_to_daily(result, drought_df, 'vhi')
            if 'ndwi' in drought_df.columns:
                rs_features['ndwi'] = self._align_rs_to_daily(result, drought_df, 'ndwi')

        for col in rs_features.columns:
            if rs_features[col].isna().all():
                rs_features[col] = 0.0
            else:
                rs_features[col] = rs_features[col].fillna(rs_features[col].median())

        for col in rs_features.columns:
            result[col] = rs_features[col].values

        n_rs_features = len(rs_features.columns)
        logger.info(f"遥感特征生成完成: 新增 {n_rs_features} 个遥感特征 ({list(rs_features.columns)})")
        return result

    def _align_rs_to_daily(self, target_df: pd.DataFrame,
                            rs_df: pd.DataFrame, col_name: str) -> pd.Series:
        if col_name not in rs_df.columns:
            return pd.Series(0.0, index=target_df.index)

        rs_series = rs_df[col_name].copy()

        if isinstance(rs_df.index, pd.DatetimeIndex):
            rs_series = rs_series[~rs_series.index.duplicated(keep='first')]
        elif 'date' in rs_df.columns:
            rs_series.index = pd.to_datetime(rs_df['date'])
            rs_series = rs_series[~rs_series.index.duplicated(keep='first')]

        if not isinstance(target_df.index, pd.DatetimeIndex):
            return pd.Series(rs_series.mean() if len(rs_series) > 0 else 0.0,
                             index=target_df.index)

        aligned = pd.Series(np.nan, index=target_df.index)
        common = target_df.index.intersection(rs_series.index)
        if len(common) > 0:
            aligned.loc[common] = rs_series.loc[common].values
        aligned = aligned.ffill().bfill()
        if aligned.isna().any():
            fill_val = rs_series.median() if len(rs_series) > 0 else 0.0
            aligned.fillna(fill_val, inplace=True)
        return aligned

    def _fallback_weather_risk(self, df: pd.DataFrame) -> pd.Series:
        """回退方案：基于价格波动率的代理变量"""
        if 'volatility_20' in df.columns:
            return (df['volatility_20'] * 0.3).clip(0, 100)
        elif 'high_low_range' in df.columns:
            return (df['high_low_range'] * 30).clip(0, 100)
        else:
            return pd.Series(np.full(len(df), 15.0), index=df.index)

    def engineer_all_features(self, df: pd.DataFrame,
                                 macro_df: pd.DataFrame = None,
                                 weather_df: pd.DataFrame = None,
                                 causal_parents: list = None,
                                 rs_panel: Dict[str, pd.DataFrame] = None,
                                 province: str = None) -> pd.DataFrame:
        logger.info("开始特征工程...")
        base_cols = list(df.columns)
        result = self.create_lag_features(df)
        result = self.create_moving_averages(result)
        result = self.create_seasonal_features(result)
        result = self.create_volatility_features(result)
        result = self.create_technical_indicators(result)
        result = self.create_basis_features(result)
        result = self.create_business_features(result, macro_df=macro_df, weather_df=weather_df, rs_panel=rs_panel)
        if rs_panel is not None:
            result = self.create_remote_sensing_features(result, rs_panel=rs_panel, province=province)
        new_cols = [c for c in result.columns if c not in base_cols]
        n_new = len(new_cols)
        total = len(result.columns)
        logger.info(f"特征工程完成: 基础特征 {len(base_cols)} 个 → 总特征 {total} 个 (新增 {n_new} 个)")
        result = self.select_features(result, causal_parents=causal_parents)
        result = result.dropna()
        logger.info(f"删除缺失值后剩余: {len(result)} 行, {len(result.columns)} 个有效风险特征")
        return result
