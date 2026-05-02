import numpy as np
import pandas as pd
from typing import Dict, Optional
from utils.helpers import logger_setup

logger = logger_setup('arima_baseline')


class ARIMABaseline:
    def __init__(self, order: tuple = (1, 1, 1), seasonal_order: tuple = None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted = None
        self.residuals = None

    def fit(self, y_train: np.ndarray, exog_train: np.ndarray = None):
        y_train = np.asarray(y_train).flatten()
        try:
            from statsmodels.tsa.arima.model import ARIMA
            if self.seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                self.model = SARIMAX(y_train, exog=exog_train,
                                      order=self.order,
                                      seasonal_order=self.seasonal_order,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
            else:
                self.model = ARIMA(y_train, exog=exog_train, order=self.order)
            self.fitted = self.model.fit(disp=False)
            self.residuals = self.fitted.resid
            logger.info(f"ARIMA{self.order}拟合完成: AIC={self.fitted.aic:.2f}")
        except Exception as e:
            logger.warning(f"ARIMA拟合失败({e}),使用朴素基线")
            self.fitted = None
            self.residuals = y_train - np.mean(y_train)

    def predict(self, n_steps: int, exog: np.ndarray = None) -> np.ndarray:
        if self.fitted is not None:
            try:
                forecast = self.fitted.forecast(steps=n_steps, exog=exog)
                return np.asarray(forecast).flatten()
            except Exception as e:
                logger.warning(f"ARIMA预测失败({e}),使用均值基线")
        return np.full(n_steps, np.mean(self.residuals + self.fitted.fittedvalues if self.fitted else 0))

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        errors = y_true - y_pred
        mape = float(np.mean(np.abs(errors / (y_true + 1e-8))) * 100)
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {
            'mape': round(mape, 4),
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'r2': round(r2, 6),
            'method': f'ARIMA{self.order}',
        }


def arima_baseline_comparison(y_train: np.ndarray,
                                y_test: np.ndarray,
                                exog_train: np.ndarray = None,
                                exog_test: np.ndarray = None,
                                orders: list = None) -> Dict:
    if orders is None:
        orders = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (0, 1, 1)]
    results = {}
    best_mape = float('inf')
    best_order = None
    best_pred = None
    for order in orders:
        try:
            arima = ARIMABaseline(order=order)
            arima.fit(y_train, exog_train)
            pred = arima.predict(len(y_test), exog_test)
            metrics = arima.evaluate(y_test, pred)
            results[str(order)] = metrics
            if metrics['mape'] < best_mape:
                best_mape = metrics['mape']
                best_order = order
                best_pred = pred
        except Exception as e:
            results[str(order)] = {'error': str(e)}
    naive_pred = np.full(len(y_test), np.mean(y_train))
    naive_errors = y_test - naive_pred
    naive_mape = float(np.mean(np.abs(naive_errors / (y_test + 1e-8))) * 100)
    results['naive'] = {'mape': round(naive_mape, 4), 'method': 'Naive Mean'}
    return {
        'all_results': results,
        'best_order': best_order,
        'best_mape': best_mape,
        'best_pred': best_pred,
        'naive_mape': naive_mape,
    }
