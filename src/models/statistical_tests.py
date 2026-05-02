import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from utils.helpers import logger_setup

logger = logger_setup('statistical_tests')


def diebold_mariano_test(errors_model: np.ndarray,
                          errors_benchmark: np.ndarray,
                          h: int = 1,
                          crit: str = 'MSE') -> Dict:
    d_t = _compute_loss_differential(errors_model, errors_benchmark, crit)
    d_bar = np.mean(d_t)
    T = len(d_t)
    if T < 2:
        return {'dm_statistic': 0.0, 'p_value': 1.0, 'significant': False,
                'ci_lower': 0.0, 'ci_upper': 0.0, 'method': 'DM', 'crit': crit}
    gamma_0 = np.var(d_t, ddof=0)
    max_lag = min(h - 1, T // 3) if h > 1 else 0
    gamma_sum = gamma_0
    for lag in range(1, max_lag + 1):
        gamma_lag = np.cov(d_t[lag:], d_t[:-lag], ddof=0)[0, 1] if lag < T else 0
        gamma_sum += 2 * gamma_lag
    var_d_bar = gamma_sum / T
    if var_d_bar < 1e-15:
        var_d_bar = 1e-15
    dm_stat = d_bar / np.sqrt(var_d_bar)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    z_alpha = stats.norm.ppf(0.975)
    ci_lower = d_bar - z_alpha * np.sqrt(var_d_bar)
    ci_upper = d_bar + z_alpha * np.sqrt(var_d_bar)
    result = {
        'dm_statistic': round(float(dm_stat), 4),
        'p_value': round(float(p_value), 6),
        'significant': p_value < 0.05,
        'ci_lower': round(float(ci_lower), 6),
        'ci_upper': round(float(ci_upper), 6),
        'd_bar': round(float(d_bar), 6),
        'var_d_bar': round(float(var_d_bar), 8),
        'n_samples': T,
        'method': 'Diebold-Mariano',
        'crit': crit,
        'h': h,
        'interpretation': _interpret_dm(dm_stat, p_value, d_bar)
    }
    logger.info(f"DM检验: stat={dm_stat:.4f}, p={p_value:.6f}, "
                f"{'显著' if p_value < 0.05 else '不显著'}")
    return result


def _compute_loss_differential(errors_model: np.ndarray,
                                errors_benchmark: np.ndarray,
                                crit: str = 'MSE') -> np.ndarray:
    errors_model = np.asarray(errors_model).flatten()
    errors_benchmark = np.asarray(errors_benchmark).flatten()
    min_len = min(len(errors_model), len(errors_benchmark))
    errors_model = errors_model[:min_len]
    errors_benchmark = errors_benchmark[:min_len]
    if crit == 'MSE':
        loss_model = errors_model ** 2
        loss_benchmark = errors_benchmark ** 2
    elif crit == 'MAE':
        loss_model = np.abs(errors_model)
        loss_benchmark = np.abs(errors_benchmark)
    elif crit == 'MAPE':
        loss_model = np.abs(errors_model)
        loss_benchmark = np.abs(errors_benchmark)
    else:
        loss_model = errors_model ** 2
        loss_benchmark = errors_benchmark ** 2
    return loss_model - loss_benchmark


def _interpret_dm(dm_stat: float, p_value: float, d_bar: float) -> str:
    if p_value < 0.01:
        sig_level = "极显著(p<0.01)"
    elif p_value < 0.05:
        sig_level = "显著(p<0.05)"
    elif p_value < 0.1:
        sig_level = "边际显著(p<0.1)"
    else:
        sig_level = "不显著(p≥0.1)"
    if d_bar < 0:
        better = "模型精度显著优于基准"
    else:
        better = "基准精度优于模型"
    return f"DM检验{sig_level}，{better}，d̄={d_bar:.6f}"


def white_reality_check(model_errors: np.ndarray,
                         benchmark_errors_list: List[np.ndarray],
                         B: int = 1000,
                         alpha: float = 0.05,
                         crit: str = 'MSE') -> Dict:
    model_errors = np.asarray(model_errors).flatten()
    T = len(model_errors)
    n_benchmarks = len(benchmark_errors_list)
    if T < 10 or n_benchmarks == 0:
        return {'w_statistic': 0.0, 'p_value': 1.0, 'passed': False,
                'n_bootstrap': B, 'method': 'White Reality Check'}
    if crit == 'MSE':
        loss_model = model_errors ** 2
    else:
        loss_model = np.abs(model_errors)
    perf_matrix = np.zeros((T, n_benchmarks))
    for j, bench_err in enumerate(benchmark_errors_list):
        bench_err = np.asarray(bench_err).flatten()
        min_len = min(T, len(bench_err))
        if crit == 'MSE':
            loss_bench = bench_err[:min_len] ** 2
        else:
            loss_bench = np.abs(bench_err[:min_len])
        perf_matrix[:min_len, j] = loss_bench[:min_len] - loss_model[:min_len]
    v_bar = np.mean(perf_matrix, axis=0)
    v_max = np.max(v_bar)
    perf_centered = perf_matrix - v_bar
    rng = np.random.RandomState(42)
    boot_max_stats = np.zeros(B)
    for b in range(B):
        indices = rng.randint(0, T, size=T)
        boot_perf = perf_centered[indices]
        boot_v_bar = np.mean(boot_perf, axis=0)
        boot_max_stats[b] = np.max(boot_v_bar)
    p_value = np.mean(boot_max_stats >= v_max)
    result = {
        'w_statistic': round(float(v_max), 6),
        'p_value': round(float(p_value), 6),
        'passed': p_value < alpha,
        'n_bootstrap': B,
        'n_benchmarks': n_benchmarks,
        'n_samples': T,
        'alpha': alpha,
        'method': 'White Reality Check',
        'interpretation': _interpret_white(v_max, p_value, alpha)
    }
    logger.info(f"White Reality Check: W={v_max:.6f}, p={p_value:.6f}, "
                f"{'通过' if p_value < alpha else '未通过'}")
    return result


def _interpret_white(w_stat: float, p_value: float, alpha: float) -> str:
    if p_value < alpha:
        return (f"White Reality Check通过(p={p_value:.4f}<{alpha})，"
                f"模型预测能力真实，非数据挖掘结果")
    else:
        return (f"White Reality Check未通过(p={p_value:.4f}>={alpha})，"
                f"模型预测能力可能为数据挖掘偏差，需进一步验证")


def comprehensive_statistical_report(y_true: np.ndarray,
                                      y_pred_model: np.ndarray,
                                      y_pred_benchmarks: Dict[str, np.ndarray],
                                      h: int = 1) -> Dict:
    errors_model = np.asarray(y_true).flatten() - np.asarray(y_pred_model).flatten()
    min_len = min(len(y_true), len(y_pred_model))
    y_true_arr = np.asarray(y_true).flatten()[:min_len]
    errors_model = errors_model[:min_len]
    dm_results = {}
    for name, y_pred_bench in y_pred_benchmarks.items():
        y_pred_bench = np.asarray(y_pred_bench).flatten()
        min_len_bench = min(len(y_true_arr), len(y_pred_bench))
        errors_bench = (y_true_arr[:min_len_bench] - y_pred_bench[:min_len_bench])
        errors_model_aligned = errors_model[:min_len_bench]
        dm_results[name] = diebold_mariano_test(errors_model_aligned, errors_bench, h=h)
    benchmark_errors_list = []
    benchmark_names = []
    for name, y_pred_bench in y_pred_benchmarks.items():
        y_pred_bench = np.asarray(y_pred_bench).flatten()
        min_len_bench = min(len(y_true_arr), len(y_pred_bench))
        bench_errors = y_true_arr[:min_len_bench] - y_pred_bench[:min_len_bench]
        benchmark_errors_list.append(bench_errors)
        benchmark_names.append(name)
    min_len_all = min(len(errors_model), *[len(be) for be in benchmark_errors_list])
    white_result = white_reality_check(
        errors_model[:min_len_all],
        [be[:min_len_all] for be in benchmark_errors_list],
        B=1000,
        crit='MSE'
    )
    report = {
        'dm_tests': dm_results,
        'white_reality_check': white_result,
        'model_mape': round(float(np.mean(np.abs(errors_model / (y_true_arr + 1e-8))) * 100), 4),
        'n_samples': min_len,
        'summary': _generate_report_summary(dm_results, white_result)
    }
    logger.info(f"综合统计检验报告: DM检验{len(dm_results)}组, White检验{'通过' if white_result['passed'] else '未通过'}")
    return report


def _generate_report_summary(dm_results: Dict, white_result: Dict) -> str:
    lines = ["=== 统计检验综合报告 ===", ""]
    lines.append("1. Diebold-Mariano检验:")
    for name, dm in dm_results.items():
        sig = "✅显著" if dm['significant'] else "❌不显著"
        lines.append(f"   vs {name}: DM={dm['dm_statistic']:.4f}, p={dm['p_value']:.4f} {sig}")
    lines.append("")
    lines.append("2. White Reality Check:")
    passed = "✅通过" if white_result['passed'] else "❌未通过"
    lines.append(f"   W={white_result['w_statistic']:.6f}, p={white_result['p_value']:.4f} {passed}")
    lines.append(f"   {white_result['interpretation']}")
    return "\n".join(lines)


def clark_west_test(y_true: np.ndarray,
                    y_pred_model1: np.ndarray,
                    y_pred_model2: np.ndarray,
                    h: int = 1) -> Dict:
    y_true = np.asarray(y_true).flatten()
    e1 = y_true - np.asarray(y_pred_model1).flatten()
    e2 = y_true - np.asarray(y_pred_model2).flatten()
    min_len = min(len(e1), len(e2))
    e1 = e1[:min_len]
    e2 = e2[:min_len]
    T = len(e1)
    if T < 10:
        return {'cw_statistic': 0.0, 'p_value': 1.0, 'significant': False,
                'method': 'Clark-West', 'n_samples': T}
    mse1 = np.mean(e1 ** 2)
    mse2 = np.mean(e2 ** 2)
    f_t = e2 ** 2 - e1 ** 2 + (e1 * e2) - (e1 ** 2)
    f_bar = np.mean(f_t)
    gamma_0 = np.var(f_t, ddof=0)
    max_lag = min(h, T // 3)
    gamma_sum = gamma_0
    for lag in range(1, max_lag + 1):
        if lag < T:
            gamma_lag = np.cov(f_t[lag:], f_t[:-lag], ddof=0)[0, 1]
            gamma_sum += 2 * gamma_lag
    var_f_bar = gamma_sum / T
    if var_f_bar < 1e-15:
        var_f_bar = 1e-15
    cw_stat = f_bar / np.sqrt(var_f_bar)
    p_value = 1 - stats.norm.cdf(cw_stat)
    sig = p_value < 0.05
    interpretation = f"Clark-West检验{'极显著(p<0.01)' if p_value < 0.01 else '显著(p<0.05)' if p_value < 0.05 else '不显著(p≥0.05)'}，"
    interpretation += f"模型1{'显著优于' if sig else '不显著优于'}模型2，CW={cw_stat:.4f}"
    result = {
        'cw_statistic': round(float(cw_stat), 4),
        'p_value': round(float(p_value), 6),
        'significant': sig,
        'mse_model1': round(float(mse1), 6),
        'mse_model2': round(float(mse2), 6),
        'mse_diff': round(float(mse2 - mse1), 6),
        'f_bar': round(float(f_bar), 6),
        'var_f_bar': round(float(var_f_bar), 8),
        'n_samples': T,
        'h': h,
        'method': 'Clark-West',
        'interpretation': interpretation
    }
    logger.info(f"Clark-West检验: CW={cw_stat:.4f}, p={p_value:.6f}, "
                f"{'显著' if sig else '不显著'}")
    return result
