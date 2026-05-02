import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import logging

from data.data_loader import DataLoader
from utils.constants import DATA_CONFIG
from utils.helpers import logger_setup

logger = logger_setup('data_preprocessor')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.05, random_state=42, n_jobs=-1
        )
        self.preprocessing_log = []

    def align_time_frequency(self, df: pd.DataFrame,
                               target_freq: str = 'D',
                               date_col: str = None) -> pd.DataFrame:
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        current_freq = pd.infer_freq(df.index[:5])
        if current_freq == target_freq or current_freq is None:
            return df
        if target_freq == 'D':
            df = df.asfreq('D')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            df[numeric_cols] = df[numeric_cols].ffill()
        elif target_freq == 'M':
            df = df.resample('M').mean()
        elif target_freq == 'Y':
            df = df.resample('Y').mean()
        else:
            df = df.asfreq(target_freq)
        self.preprocessing_log.append({
            'step': 'time_alignment',
            'from': current_freq,
            'to': target_freq,
            'rows_after': len(df)
        })
        return df

    def handle_missing_values(self, df: pd.DataFrame,
                                threshold: float = 0.05) -> pd.DataFrame:
        missing_ratio = df.isnull().sum() / len(df)
        total_missing = missing_ratio.sum()
        logger.info(f"缺失值总比例: {total_missing:.4f}")
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        for col in numeric_df.columns:
            col_missing = missing_ratio.get(col, 0)
            if col_missing == 0:
                continue
            if col_missing < threshold:
                numeric_df[col] = numeric_df[col].interpolate(method='linear')
                numeric_df[col] = numeric_df[col].bfill().ffill()
            else:
                knn_imputer = KNNImputer(n_neighbors=5)
                numeric_df[[col]] = knn_imputer.fit_transform(numeric_df[[col]])
        non_numeric = df.select_dtypes(exclude=[np.number])
        if not non_numeric.empty:
            for col in non_numeric.columns:
                non_numeric[col] = non_numeric[col].ffill().bfill()
        result = pd.concat([numeric_df, non_numeric], axis=1)
        self.preprocessing_log.append({
            'step': 'missing_values',
            'initial_missing_pct': float(total_missing),
            'final_missing_pct': float(result.isnull().sum().sum() / (result.shape[0] * result.shape[1])),
            'method': 'interpolation+KNN'
        })
        return result

    def detect_outliers(self, df: pd.DataFrame,
                          method: str = 'iqr') -> pd.Series:
        numeric_df = df.select_dtypes(include=[np.number])
        outlier_flags = pd.Series(False, index=df.index)
        if method == 'iqr':
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = ((numeric_df < lower_bound) | (numeric_df > upper_bound)).any(axis=1)
            outlier_flags = outlier_flags | iqr_outliers
        iso_labels = self.isolation_forest.fit_predict(numeric_df.fillna(numeric_df.median()))
        iso_outliers = pd.Series(iso_labels == -1, index=df.index)
        outlier_flags = outlier_flags | iso_outliers
        n_outliers = outlier_flags.sum()
        pct = n_outliers / len(df) * 100
        logger.info(f"异常值检测结果: {n_outliers} 个 ({pct:.2f}%)")
        self.preprocessing_log.append({
            'step': 'outlier_detection',
            'method': method,
            'outlier_count': int(n_outliers),
            'outlier_pct': float(pct)
        })
        return outlier_flags

    def handle_outliers(self, df: pd.DataFrame,
                          outlier_flags: pd.Series = None) -> pd.DataFrame:
        if outlier_flags is None:
            outlier_flags = self.detect_outliers(df)
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median_val = result[col].median()
            result.loc[outlier_flags, col] = median_val
        n_fixed = outlier_flags.sum()
        logger.info(f"已用中位数替换 {n_fixed} 个异常值")
        return result

    def normalize_data(self, df: pd.DataFrame,
                         columns: list = None,
                         method: str = 'zscore') -> pd.DataFrame:
        result = df.copy()
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        existing_cols = [c for c in columns if c in result.columns]
        if not existing_cols:
            return result
        if method == 'zscore':
            result[existing_cols] = self.scaler.fit_transform(result[existing_cols])
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            result[existing_cols] = scaler.fit_transform(result[existing_cols])
        self.preprocessing_log.append({
            'step': 'normalization',
            'method': method,
            'columns_normalized': len(existing_cols)
        })
        return result

    def preprocess_pipeline(self, df: pd.DataFrame,
                              target_freq: str = 'D',
                              normalize: bool = True) -> pd.DataFrame:
        logger.info("开始预处理流水线...")
        df = self.align_time_frequency(df, target_freq)
        df = self.handle_missing_values(df)
        outlier_flags = self.detect_outliers(df)
        df = self.handle_outliers(df, outlier_flags)
        if normalize:
            df = self.normalize_data(df)
        logger.info("预处理流水线完成")
        return df

    def get_preprocessing_report(self) -> dict:
        return {
            'steps_executed': len(self.preprocessing_log),
            'details': self.preprocessing_log,
            'status': 'completed'
        }
