import os
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from utils.config import config
from utils.constants import ALL_SYMBOLS, SYMBOL_NAMES, PROVINCE_FILES, CORE_SYMBOLS
from utils.helpers import logger_setup, timer
from utils.cache_utils import save_cache, load_cache, get_cache_key

logger = logger_setup('data_loader')

MACRO_NAME_MAP = {
    'fx_spot': '外汇现货汇率',
    'fx_spot_quote': '外汇现货报价',
    'macro_china_consumer_goods_retail': '社会消费品零售总额',
    'macro_china_cpi': '居民消费价格指数(CPI)',
    'macro_china_cpi_monthly': 'CPI月度数据',
    'macro_china_cpi_yearly': 'CPI年度数据',
    'macro_china_czsr': '财政收入',
    'macro_china_gdp': '国内生产总值(GDP)',
    'macro_china_gdzctz': '固定资产投资',
    'macro_china_m2_yearly': 'M2货币供应量',
    'macro_china_pmi': '制造业采购经理指数(PMI)',
    'macro_china_ppi': '工业生产者出厂价格指数(PPI)',
    'macro_china_ppi_yearly': 'PPI年度数据',
    'macro_china_shrzgm': '社会融资规模增量',
}

class DataLoader:
    def __init__(self, data_root: str = None):
        self.data_root = data_root or config.data_root_str
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_futures_data(self, symbol: str,
                           parse_dates: bool = True) -> pd.DataFrame:
        # 尝试从磁盘缓存加载
        cache_key = get_cache_key('futures', symbol, parse_dates=parse_dates)
        cached_data = load_cache(cache_key)
        if cached_data is not None:
            logger.info(f"从缓存加载期货数据: {SYMBOL_NAMES.get(symbol, symbol)}")
            return cached_data
        
        # 内存缓存检查
        mem_cache_key = f'futures_{symbol}_{parse_dates}'
        if mem_cache_key in self._cache:
            return self._cache[mem_cache_key].copy()
        
        # 从文件加载
        path = config.get_futures_path(symbol)
        if not os.path.exists(path):
            logger.warning(f"数据文件不存在: {path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            logger.warning(f"utf-8-sig解码失败，尝试GBK编码: {path}")
            df = pd.read_csv(path, encoding='gbk')
        if parse_dates and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
        
        # 保存到内存缓存
        self._cache[mem_cache_key] = df.copy()
        # 保存到磁盘缓存
        save_cache(cache_key, df)
        
        logger.info(f"加载期货数据: {SYMBOL_NAMES.get(symbol, symbol)} ({len(df)} 行)")
        return df

    def load_macro_data(self) -> pd.DataFrame:
        # 尝试从磁盘缓存加载
        cache_key = get_cache_key('macro', 'all')
        cached_data = load_cache(cache_key)
        if cached_data is not None:
            logger.info("从缓存加载宏观数据")
            return cached_data
        
        # 内存缓存检查
        mem_cache_key = 'macro_all'
        if mem_cache_key in self._cache:
            return self._cache[mem_cache_key].copy()
        
        # 从文件加载
        macro_dir = config.macro_dir
        if not os.path.exists(macro_dir):
            logger.warning(f"宏观数据目录不存在: {macro_dir}")
            return pd.DataFrame()
        rows = []
        for fname in sorted(os.listdir(macro_dir)):
            if fname.endswith('.csv'):
                fpath = os.path.join(macro_dir, fname)
                try:
                    df = pd.read_csv(fpath, encoding='utf-8-sig')
                    name_en = fname.replace('.csv', '')
                    name_cn = MACRO_NAME_MAP.get(name_en, name_en)
                    date_col = None
                    for c in df.columns:
                        if '日期' in c or '月份' in c or 'date' in c.lower():
                            date_col = c
                            break
                    latest_date = ''
                    value_summary = ''
                    if len(df) > 0:
                        if date_col and date_col in df.columns:
                            latest_date = str(df[date_col].iloc[0])
                        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if num_cols:
                            val_col = num_cols[0] if len(num_cols) > 0 else None
                            if val_col and val_col in df.columns:
                                latest_val = df[val_col].dropna().iloc[-1] if df[val_col].dropna().shape[0] > 0 else ''
                                value_summary = f"{df.columns.tolist()[1] if len(df.columns) > 1 else val_col}: {latest_val}"
                    rows.append({
                        '📁 数据名称': name_cn,
                        '📊 英文标识': name_en,
                        '📅 最新日期': latest_date,
                        '📈 数据概要': value_summary,
                        '🔢 记录数': f"{len(df)} 行 × {len(df.columns)} 列",
                        '📋 主要字段': ', '.join(df.columns[:5]) + ('...' if len(df.columns) > 5 else '')
                    })
                except Exception as e:
                    logger.warning(f"读取宏观文件失败 {fname}: {e}")
        result = pd.DataFrame(rows)
        
        # 保存到内存缓存
        self._cache[mem_cache_key] = result
        # 保存到磁盘缓存
        save_cache(cache_key, result)
        
        logger.info(f"加载宏观数据: {len(rows)} 个指标")
        return result

    def load_macro_raw(self, indicator: str) -> pd.DataFrame:
        # 尝试从磁盘缓存加载
        cache_key = get_cache_key('macro_raw', indicator)
        cached_data = load_cache(cache_key)
        if cached_data is not None:
            logger.info(f"从缓存加载宏观原始数据: {indicator}")
            return cached_data
        
        # 从文件加载
        macro_dir = config.macro_dir
        fname = f"{indicator}.csv"
        fpath = os.path.join(macro_dir, fname)
        if not os.path.exists(fpath):
            return pd.DataFrame()
        df = pd.read_csv(fpath, encoding='utf-8-sig')
        
        # 保存到磁盘缓存
        save_cache(cache_key, df)
        
        return df

    def load_weather_data(self, province_name: str) -> pd.DataFrame:
        # 尝试从磁盘缓存加载
        cache_key = get_cache_key('weather', province_name)
        cached_data = load_cache(cache_key)
        if cached_data is not None:
            logger.info(f"从缓存加载天气数据: {province_name}")
            return cached_data
        
        # 内存缓存检查
        mem_cache_key = f'weather_{province_name}'
        if mem_cache_key in self._cache:
            return self._cache[mem_cache_key].copy()
        
        # 从文件加载
        weather_dir = config.weather_dir
        filename = PROVINCE_FILES.get(province_name)
        if not filename:
            logger.warning(f"未知省份: {province_name}")
            return pd.DataFrame()
        path = os.path.join(weather_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"天气数据不存在: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path, encoding='utf-8-sig')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # 保存到内存缓存
        self._cache[mem_cache_key] = df.copy()
        # 保存到磁盘缓存
        save_cache(cache_key, df)
        
        logger.info(f"加载天气数据: {province_name} ({len(df)} 行)")
        return df

    def load_extended_factors(self) -> Dict[str, pd.DataFrame]:
        # 尝试从磁盘缓存加载
        cache_key = get_cache_key('extended', 'factors')
        cached_data = load_cache(cache_key)
        if cached_data is not None:
            logger.info("从缓存加载扩展因子数据")
            return cached_data
        
        # 从文件加载
        extended_dir = config.extended_dir
        result = {}
        if not os.path.exists(extended_dir):
            return result
        for fname in ['fao_food_price_index.csv', 'international_grain_prices.csv',
                       'ndvi_annual_summary.csv', 'ndvi_panel.csv']:
            fpath = os.path.join(extended_dir, fname)
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath, encoding='utf-8-sig')
                    result[fname.replace('.csv', '')] = df
                except Exception:
                    pass
        
        # 保存到磁盘缓存
        save_cache(cache_key, result)
        
        return result

    def get_available_symbols(self) -> List[str]:
        # 尝试从磁盘缓存加载
        cache_key = get_cache_key('symbols', 'available')
        cached_data = load_cache(cache_key)
        if cached_data is not None:
            logger.info("从缓存加载可用品种列表")
            return cached_data
        
        # 从文件系统加载
        futures_dir = config.futures_dir
        if not os.path.exists(futures_dir):
            return []
        symbols = []
        for fname in os.listdir(futures_dir):
            if fname.endswith('.csv'):
                code = fname.split('_')[0]
                symbols.append(code)
        symbols = sorted(symbols)
        
        # 保存到磁盘缓存
        save_cache(cache_key, symbols)
        
        return symbols

    def data_summary(self) -> pd.DataFrame:
        # 尝试从磁盘缓存加载
        cache_key = get_cache_key('summary', 'data')
        cached_data = load_cache(cache_key)
        if cached_data is not None:
            logger.info("从缓存加载数据摘要")
            return cached_data
        
        # 生成摘要
        symbols = self.get_available_symbols()
        rows = []
        for sym in symbols[:5]:
            df = self.load_futures_data(sym)
            if len(df) > 0:
                rows.append({
                    '品种': SYMBOL_NAMES.get(sym, sym),
                    '代码': sym,
                    '行数': len(df),
                    '列数': len(df.columns),
                    '日期范围': f"{df.index.min()} ~ {df.index.max()}" if hasattr(df, 'index') and hasattr(df.index, 'min') else 'N/A',
                    '缺失率': f"{df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%"
                })
        result = pd.DataFrame(rows)
        
        # 保存到磁盘缓存
        save_cache(cache_key, result)
        
        return result

    def load_remote_sensing_data(self, index_type: str = 'ndvi',
                                  province: str = None) -> pd.DataFrame:
        cache_key = get_cache_key('remote_sensing', index_type, province=province)
        cached_data = load_cache(cache_key)
        if cached_data is not None:
            logger.info(f"从缓存加载遥感数据: {index_type}")
            return cached_data

        rs_dir = config.remote_sensing_dir
        file_map = {
            'ndvi': 'ndvi_monthly.csv',
            'evi': 'evi_monthly.csv',
            'lst': 'lst_monthly.csv',
            'drought': 'drought_index_monthly.csv',
        }
        fname = file_map.get(index_type, 'ndvi_monthly.csv')
        fpath = os.path.join(rs_dir, fname)
        if not os.path.exists(fpath):
            logger.warning(f"遥感数据不存在: {fpath}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(fpath, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(fpath, encoding='gbk')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if province and 'province' in df.columns:
            df = df[df['province'] == province].copy()
        if 'date' in df.columns and df['date'].dtype == 'datetime64[ns]':
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

        save_cache(cache_key, df)
        logger.info(f"加载遥感数据: {index_type} ({len(df)} 行)")
        return df

    def load_remote_sensing_panel(self) -> Dict[str, pd.DataFrame]:
        cache_key = get_cache_key('remote_sensing', 'panel')
        cached_data = load_cache(cache_key)
        if cached_data is not None:
            logger.info("从缓存加载遥感面板数据")
            return cached_data

        result = {}
        for idx_type in ['ndvi', 'evi', 'lst', 'drought']:
            df = self.load_remote_sensing_data(idx_type)
            if not df.empty:
                result[idx_type] = df

        save_cache(cache_key, result)
        logger.info(f"加载遥感面板数据: {len(result)} 个指标")
        return result

    def clear_cache(self):
        self._cache.clear()
