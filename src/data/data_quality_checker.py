import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from utils.helpers import logger_setup

logger = logger_setup('data_quality')

VALID_DATE_START = '2020-01-01'
VALID_DATE_END = '2025-12-31'


class DataQualityChecker:
    def __init__(self):
        self.reports: Dict[str, Dict] = {}

    def check_date_range_compliance(self, df: pd.DataFrame,
                                      date_col: str = 'date',
                                      name: str = 'dataset') -> Dict[str, Any]:
        valid_start = pd.Timestamp(VALID_DATE_START)
        valid_end = pd.Timestamp(VALID_DATE_END)

        if date_col not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                dates = df.index
            else:
                return {'name': name, 'dimension': 'date_range', 'compliant': True,
                        'note': '无日期列，跳过日期范围检查'}
        else:
            dates = pd.to_datetime(df[date_col], errors='coerce')

        if len(dates) == 0:
            return {'name': name, 'dimension': 'date_range', 'compliant': True,
                    'note': '空数据集'}

        out_of_range = ((dates < valid_start) | (dates > valid_end)).sum()
        total = len(dates)
        min_date = dates.min()
        max_date = dates.max()

        report = {
            'name': name,
            'dimension': 'date_range_compliance',
            'valid_range': f'{VALID_DATE_START} ~ {VALID_DATE_END}',
            'actual_range': f'{min_date} ~ {max_date}',
            'total_records': int(total),
            'out_of_range_records': int(out_of_range),
            'compliant': bool(out_of_range == 0),
            'compliance_rate': float((total - out_of_range) / total * 100) if total > 0 else 100,
            'score': float((total - out_of_range) / total * 100) if total > 0 else 100
        }
        self.reports[f'{name}_date_range'] = report
        if out_of_range > 0:
            logger.warning(f"数据日期范围违规: {name} 有 {out_of_range}/{total} 条记录超出"
                          f" [{VALID_DATE_START}, {VALID_DATE_END}]")
        return report

    def check_value_continuity(self, df: pd.DataFrame,
                                 date_col: str = 'date',
                                 name: str = 'dataset',
                                 max_gap_days: int = 7) -> Dict[str, Any]:
        if date_col not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            return {'name': name, 'dimension': 'continuity', 'note': '无日期列，跳过'}

        dates = pd.to_datetime(df[date_col], errors='coerce') if date_col in df.columns else df.index
        if isinstance(dates, pd.Series):
            dates = pd.DatetimeIndex(dates.values)
        dates_sorted = dates.sort_values()
        gaps = dates_sorted[1:] - dates_sorted[:-1]
        gap_days = pd.Series(gaps.days.values, dtype=int)

        large_gaps = (gap_days > max_gap_days).sum()
        max_gap = gap_days.max() if len(gap_days) > 0 else 0
        median_gap = gap_days.median() if len(gap_days) > 0 else 0

        report = {
            'name': name,
            'dimension': 'value_continuity',
            'max_gap_days': int(max_gap),
            'median_gap_days': float(median_gap),
            'large_gaps_count': int(large_gaps),
            'max_gap_threshold': max_gap_days,
            'continuous': bool(large_gaps == 0),
            'score': max(0, 100 - large_gaps * 5)
        }
        self.reports[f'{name}_continuity'] = report
        return report

    def check_completeness(self, df: pd.DataFrame,
                             name: str = 'dataset') -> Dict[str, Any]:
        total_cells = df.shape[0] * df.shape[1]
        missing_total = df.isnull().sum().sum()
        missing_pct = missing_total / total_cells * 100 if total_cells > 0 else 0
        col_missing = (df.isnull().sum() / len(df) * 100).to_dict()
        row_missing = (df.isnull().sum(axis=1) / df.shape[1] * 100).describe().to_dict()
        report = {
            'name': name,
            'dimension': 'completeness',
            'total_cells': int(total_cells),
            'missing_cells': int(missing_total),
            'missing_pct': float(missing_pct),
            'columns_with_missing': sum(1 for v in col_missing.values() if v > 0),
            'max_col_missing_pct': max(col_missing.values()) if col_missing else 0,
            'avg_row_missing_pct': float(row_missing.get('mean', 0)),
            'score': max(0, 100 - missing_pct)
        }
        self.reports[f'{name}_completeness'] = report
        return report

    def check_consistency(self, df: pd.DataFrame,
                            name: str = 'dataset') -> Dict[str, Any]:
        issues = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() == 1:
                issues.append(f'{col}: 常数列')
            if df[col].dtype == 'float64':
                if (df[col] % 1 == 0).all():
                    pass
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        for col in date_cols:
            try:
                pd.to_datetime(df[col])
            except (ValueError, TypeError):
                issues.append(f'{col}: 非法日期格式')
        duplicates = df.duplicated().sum()
        report = {
            'name': name,
            'dimension': 'consistency',
            'issues_found': len(issues),
            'issue_details': issues[:10],
            'duplicate_rows': int(duplicates),
            'duplicate_pct': float(duplicates / len(df) * 100) if len(df) > 0 else 0,
            'score': max(0, 100 - len(issues) * 5 - duplicates / len(df) * 10)
        }
        self.reports[f'{name}_consistency'] = report
        return report

    def check_accuracy(self, df: pd.DataFrame,
                         name: str = 'dataset') -> Dict[str, Any]:
        numeric_df = df.select_dtypes(include=[np.number])
        n_outliers_total = 0
        outlier_details = {}
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
            outliers = ((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum()
            n_outliers_total += outliers
            if outliers > 0:
                outlier_details[col] = int(outliers)
        report = {
            'name': name,
            'dimension': 'accuracy',
            'total_outliers': int(n_outliers_total),
            'outlier_pct': float(n_outliers_total / (len(df) * len(numeric_df.columns)) * 100) if len(df) > 0 and len(numeric_df.columns) > 0 else 0,
            'outlier_columns': outlier_details,
            'score': max(0, 100 - n_outliers_total / len(df) * 2)
        }
        self.reports[f'{name}_accuracy'] = report
        return report

    def generate_overall_report(self) -> Dict[str, Any]:
        scores = []
        for key, report in self.reports.items():
            if isinstance(report, dict) and 'score' in report:
                scores.append(report['score'])
        overall_score = np.mean(scores) if scores else 0
        return {
            'total_checks': len(self.reports),
            'overall_score': float(overall_score),
            'grade': 'A' if overall_score >= 90 else 'B' if overall_score >= 70 else 'C' if overall_score >= 50 else 'D',
            'details': self.reports
        }

    def reset(self):
        self.reports.clear()
