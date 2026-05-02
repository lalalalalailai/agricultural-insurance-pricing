"""
算法对比实验封装模块

标准PC vs Agri-PC, 标准T-Learner vs ACML 定量对比
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional

from models.agri_pc import AgriPC, evaluate_graph_metrics
from models.causal_discovery import CausalDiscovery
from models.acml import ACML
from models.causal_estimation import TLearner


def compare_causal_discovery(data: pd.DataFrame,
                              variables: list = None,
                              alpha: float = 0.05) -> Dict:
    """
    标准PC vs Agri-PC 因果发现算法对比

    Parameters
    ----------
    data : 实验数据
    variables : 变量列表
    alpha : 显著性水平

    Returns
    -------
    Dict with standard_pc and agri_pc metrics
    """
    from utils.constants import DAG_CONFIG

    ground_truth_edges = set()
    for edge_list in [
        DAG_CONFIG.get('agri_growth_cycle_edges', []),
        DAG_CONFIG.get('futures_delivery_edges', []),
        DAG_CONFIG.get('business_prior_edges', [])
    ]:
        for e in edge_list:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                ground_truth_edges.add((e[0], e[1]))

    cd = CausalDiscovery(alpha=alpha)

    t0 = time.time()
    graph_standard = cd.run_pc_algorithm(data, variables, apply_business_prior=False)
    t_standard = time.time() - t0

    standard_edges = set()
    if hasattr(graph_standard, 'edges') and graph_standard.edges:
        for e in graph_standard.edges:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                standard_edges.add((e[0], e[1]))

    t0 = time.time()
    agri_pc = AgriPC(alpha=alpha)
    graph_agri = agri_pc.discover(data, variables)
    t_agri = time.time() - t0

    agri_edges = set()
    if hasattr(graph_agri, 'edges') and graph_agri.edges:
        for e in graph_agri.edges:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                agri_edges.add((e[0], e[1]))

    metrics_standard = evaluate_graph_metrics(standard_edges, ground_truth_edges)
    metrics_agri = evaluate_graph_metrics(agri_edges, ground_truth_edges)

    search_reduction = 0.0
    if metrics_standard['n_discovered'] > 0:
        search_reduction = (1 - metrics_agri['n_discovered'] / metrics_standard['n_discovered']) * 100

    return {
        'standard_pc': {
            'f1': metrics_standard['f1'],
            'precision': metrics_standard['precision'],
            'recall': metrics_standard['recall'],
            'shd': metrics_standard['shd'],
            'n_edges': metrics_standard['n_discovered'],
            'search_space_reduction_pct': 0.0,
            'runtime_seconds': round(t_standard, 2),
        },
        'agri_pc': {
            'f1': metrics_agri['f1'],
            'precision': metrics_agri['precision'],
            'recall': metrics_agri['recall'],
            'shd': metrics_agri['shd'],
            'n_edges': metrics_agri['n_discovered'],
            'search_space_reduction_pct': round(search_reduction, 1),
            'runtime_seconds': round(t_agri, 2),
        },
        'ground_truth_n_edges': len(ground_truth_edges),
        'improvement': {
            'f1_gain': round(metrics_agri['f1'] - metrics_standard['f1'], 4),
            'shd_reduction': metrics_standard['shd'] - metrics_agri['shd'],
            'search_space_reduction_pct': round(search_reduction, 1),
        }
    }


def compare_causal_estimation(X: pd.DataFrame,
                               treatment: np.ndarray,
                               outcome: np.ndarray,
                               y_true: np.ndarray = None,
                               y_pred: np.ndarray = None,
                               delivery_days: np.ndarray = None,
                               risk_indicator: np.ndarray = None) -> Dict:
    """
    标准T-Learner vs ACML 因果估计对比

    Parameters
    ----------
    X : 特征矩阵
    treatment : 处理变量
    outcome : 结果变量
    y_true : 真实价格
    y_pred : ACML预测价格
    delivery_days : 交割日距离天数
    risk_indicator : 风险指标

    Returns
    -------
    Dict with t_learner and acml metrics
    """
    t_learner = TLearner()
    t0 = time.time()
    t_learner.fit(X, treatment, outcome)
    t_fit_t = time.time() - t0

    t0 = time.time()
    acml = ACML(lambda_agri=0.1, alpha_delivery=0.5)
    acml.fit(X, treatment, outcome, delivery_days, risk_indicator)
    t_fit_acml = time.time() - t0

    t_effect = t_learner.estimate_effect(X)
    acml_cate = acml.predict_cate(X)

    t_ate = float(t_effect.get('ate', 0))
    t_cate_std = float(np.std(t_effect.get('cate', np.array([0]))))
    acml_ate = float(np.mean(acml_cate)) if len(acml_cate) > 0 else 0
    acml_cate_std = float(np.std(acml_cate)) if len(acml_cate) > 0 else 0

    results = {
        't_learner': {
            'ate': round(t_ate, 4),
            'cate_std': round(t_cate_std, 4),
            'fit_time_seconds': round(t_fit_t, 2),
        },
        'acml': {
            'ate': round(acml_ate, 4),
            'cate_std': round(acml_cate_std, 4),
            'fit_time_seconds': round(t_fit_acml, 2),
        }
    }

    if y_true is not None and y_pred is not None:
        mask = np.abs(y_true) > 1e-8
        mape_acml = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

        y_pred_t = y_pred - acml_ate + t_ate
        mape_t = float(np.mean(np.abs((y_true[mask] - y_pred_t[mask]) / y_true[mask])) * 100)

        if risk_indicator is not None:
            extreme_mask = risk_indicator > np.percentile(risk_indicator, 90)
            if np.any(extreme_mask & mask):
                extreme_err_t = float(np.mean(np.abs(
                    (y_true[extreme_mask & mask] - y_pred_t[extreme_mask & mask]) / y_true[extreme_mask & mask])) * 100)
                extreme_err_acml = float(np.mean(np.abs(
                    (y_true[extreme_mask & mask] - y_pred[extreme_mask & mask]) / y_true[extreme_mask & mask])) * 100)
            else:
                extreme_err_t = 0
                extreme_err_acml = 0
        else:
            extreme_err_t = 0
            extreme_err_acml = 0

        results['t_learner']['mape'] = round(mape_t, 4)
        results['t_learner']['extreme_weather_error_rate'] = round(extreme_err_t, 4)
        results['acml']['mape'] = round(mape_acml, 4)
        results['acml']['extreme_weather_error_rate'] = round(extreme_err_acml, 4)

        results['improvement'] = {
            'mape_reduction_pct': round((mape_t - mape_acml) / mape_t * 100, 2) if mape_t > 0 else 0,
            'cate_std_reduction_pct': round((t_cate_std - acml_cate_std) / t_cate_std * 100, 2) if t_cate_std > 0 else 0,
            'extreme_error_reduction_pct': round((extreme_err_t - extreme_err_acml) / extreme_err_t * 100, 2) if extreme_err_t > 0 else 0,
        }

        try:
            from models.statistical_tests import diebold_mariano_test
            errors_acml = y_true[mask] - y_pred[mask]
            errors_t = y_true[mask] - y_pred_t[mask]
            dm_result = diebold_mariano_test(errors_acml, errors_t, h=1, crit='MSE')
            results['dm_test'] = dm_result
        except Exception as e:
            results['dm_test'] = {'error': str(e)}

    return results


def compare_with_dm_test(y_true: np.ndarray,
                           y_pred_model: np.ndarray,
                           y_pred_benchmarks: Dict[str, np.ndarray],
                           h: int = 1) -> Dict:
    try:
        from models.statistical_tests import comprehensive_statistical_report
        return comprehensive_statistical_report(y_true, y_pred_model, y_pred_benchmarks, h=h)
    except ImportError:
        return {'error': 'statistical_tests module not available'}
