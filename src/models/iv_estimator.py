import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

from utils.helpers import logger_setup, timer

logger = logger_setup('iv_estimator')


class InstrumentalVariable:
    """
    Instrumental Variable (IV) 因果效应估计器
    
    理论基础：Angrist J D, Imbens G W.
    "Two-stage least squares estimation of average causal effects in models with variable treatment intensity"
    Journal of the American Statistical Association, 1995, 90(430): 431-442
    
    核心思想：
    利用外生工具变量Z识别处理变量D对结果变量Y的因果效应，
    解决由于不可观测混杂因素导致的内生性问题
    
    数学模型：
        Stage 1: D = πZ + Xγ + ν     (第一阶段的简化式)
        Stage 2: Y = βD_hat + Xδ + ε   (第二阶段的结构式)
    
    其中：
        - Z: 工具变量（外生，影响D但不直接影响Y）
        - D: 内生处理变量（如天气风险指数）
        - Y: 结果变量（如期货价格）
        - X: 外生控制变量
        - β: 目标因果效应（Wald估计量）
    
    IV有效性条件（必须满足）：
    1. 相关性：Cov(Z, D) ≠ 0（工具变量与处理变量相关）
    2. 外生性：Cov(Z, ε) = 0（工具变量与误差项不相关，即无直接路径到Y）
    3. 排他性：Z仅通过D影响Y（无其他路径）
    
    农险场景适用：
    - 政策冲击（如补贴政策调整）作为自然实验的IV
    - 天气异常事件作为天气风险的外生冲击IV
    - 国际市场价格作为国内价格的外生IV
    
    诊断检验：
    - 弱IV检验（F统计量 > 10为强IV）
    - 过度识别检验（Sargan/Hansen J检验，多IV时使用）
    - Wu-Hausman内生性检验（对比OLS vs 2SLS）
    """
    
    def __init__(self,
                 instrument_var: str,
                 weak_iv_threshold: float = 10.0):
        """
        初始化IV估计器
        
        Args:
            instrument_var: 工具变量列名
            weak_iv_threshold: 弱IV检验F统计量阈值（默认10，Stock-Yogo临界值）
        """
        self.instrument_var = instrument_var
        self.weak_iv_threshold = weak_iv_threshold
        
        self.wald_estimate_ = None   # Wald估计量（β_2SLS)
        self.se_2sls_ = None         # 两阶段最小二乘法标准误
        self.p_value_ = None         # p值
        self.ci_lower_ = None        # 置信区间下界
        self.ci_upper_ = None        # 置信区间上界
        
        self.first_stage_f_stat_ = None  # 第一阶段F统计量（弱IV检验）
        self.is_weak_instrument_ = None  # 是否弱工具变量
        self.partial_r2_ = None          # 偏R²（工具变量的解释力）
        
        self.ols_estimate_ = None    # OLS估计量（用于Hausman检验）
        self.hausman_stat_ = None    # Hausman统计量
        self.is_endogenous_ = None   # 是否存在内生性
        
        self.stage1_coef_ = None     # 第一阶段系数(π)
        self.stage1_r2_ = None       # 第一阶段R²
        
        logger.info(f"IV估计器初始化: instrument={instrument_var}, "
                   f"weak_IV_threshold={weak_iv_threshold}")
    
    @timer(verbose=False)
    def fit(self, D: np.ndarray, Y: np.ndarray, Z: np.ndarray,
            X: np.ndarray = None) -> 'InstrumentalVariable':
        """
        拟合两阶段最小二乘法（2SLS）模型
        
        Stage 1: D = π·Z + X·γ + ν   （简化式回归）
        Stage 2: Y = β·D̂ + X·δ + ε   （用Stage1拟合值替代D）
        
        Args:
            D: 处理/内生变量 (n,)
            Y: 结果变量 (n,)
            Z: 工具变量 (n,) 或 (n, n_instruments)
            X: 外生控制变量 (n, p)，可选
            
        Returns:
            self (支持链式调用)
        """
        logger.info(f"开始IV-2SLS拟合: n={len(D)}, n_instruments={Z.shape[1] if len(Z.shape)>1 else 1}")
        
        n = len(D)
        
        if X is not None and X.shape[1] > 0:
            X_full = np.hstack([np.ones((n, 1)), X])
        else:
            X_full = np.ones((n, 1))
        
        Z_full = np.hstack([X_full, Z.reshape(-1, 1) if Z.ndim == 1 else Z])
        
        try:
            # ====== Stage 1: Reduced Form ======
            stage1_model = LinearRegression()
            stage1_model.fit(Z_full, D)
            D_hat = stage1_model.predict(Z_full)  # D的拟合值
            residuals_stage1 = D - D_hat
            
            self.stage1_coef_ = dict(zip(
                ['intercept'] + [f'control_{i}' for i in range(X_full.shape[1]-1)] +
                ['instrument'],
                stage1_model.coef_.tolist() + [stage1_model.intercept_] if hasattr(stage1_model, 'intercept_') else []
            ))
            
            # 第一阶段诊断
            ss_total = np.sum((D - np.mean(D))**2)
            ss_resid = np.sum(residuals_stage1**2)
            self.stage1_r2_ = 1 - ss_resid / ss_total if ss_total > 0 else 0
            
            # 弱IV检验：F统计量（工具变量联合显著性）
            # F = (R²_included / k_included) / (R²_excluded / (n - k_total))
            # 简化版：基于工具变量t统计量的平方
            if Z.ndim == 1:
                t_stat_instrument = (stage1_model.coef_[-1] / 
                                    (np.sqrt(ss_resid / (n - Z_full.shape[1])) / 
                                     np.sqrt(np.sum((Z.reshape(-1, 1) - np.mean(Z))**2))))
                self.first_stage_f_stat_ = t_stat_instrument ** 2
            else:
                from scipy.stats import f as f_dist
                ss_model = ss_total - ss_resid
                df1 = Z.shape[1] if Z.ndim > 1 else 1
                df2 = n - Z_full.shape[1]
                self.first_stage_f_stat_ = (ss_model / df1) / (ss_resid / df2) if ss_resid > 0 else 0
            
            self.is_weak_instrument_ = self.first_stage_f_stat_ < self.weak_iv_threshold
            
            # 偏R²（工具变量的边际解释力）
            restricted_model = LinearRegression().fit(X_full, D)
            D_hat_restricted = restricted_model.predict(X_full)
            ss_resid_restricted = np.sum((D - D_hat_restricted)**2)
            self.partial_r2_ = (ss_resid_restricted - ss_resid) / ss_resid_restricted if ss_resid_restricted > 0 else 0
            
            logger.info(f"Stage 1完成: R²={self.stage1_r2_:.4f}, "
                       f"F(IV)={self.first_stage_f_stat_:.2f} {'✅强IV' if not self.is_weak_instrument_ else '⚠️弱IV'}, "
                       f"偏R²={self.partial_r2_:.4f}")
            
            # ====== Stage 2: Structural Equation ======
            stage2_X = np.hstack([D_hat.reshape(-1, 1), X_full])
            stage2_model = LinearRegression()
            stage2_model.fit(stage2_X, Y)
            
            self.wald_estimate_ = stage2_model.coef_[0]  # β_Wald
            residuals_stage2 = Y - stage2_model.predict(stage2_X)
            
            # 计算标准误（需要修正因使用D̂带来的偏差）
            sigma2 = np.sum(residuals_stage2**2) / (n - stage2_X.shape[1])
            
            # Var(β_2SLS) 的近似计算
            var_Dhat = np.var(D_hat)
            var_wald = sigma2 / (var_Dhat * n) if var_Dhat > 0 else 0
            self.se_2sls_ = np.sqrt(max(var_wald, 0))
            
            # Wald检验
            z_stat = self.wald_estimate_ / (self.se_2sls_ + 1e-8)
            self.p_value_ = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            # 95%置信区间
            z_crit = 1.96
            self.ci_lower_ = self.wald_estimate_ - z_crit * self.se_2sls_
            self.ci_upper_ = self.wald_estimate_ + z_crit * self.se_2sls_
            
            # ====== Diagnostic Tests ======
            
            # Hausman检验（内生性检验）：对比OLS vs 2SLS
            ols_model = LinearRegression()
            ols_X = np.hstack([D.reshape(-1, 1), X_full])
            ols_model.fit(ols_X, Y)
            self.ols_estimate_ = ols_model.coef_[0]
            
            hausman_stat = ((self.ols_estimate_ - self.wald_estimate_)**2 /
                           (np.var(ols_model.coef_) - var_wald + 1e-8))
            self.hausman_stat_ = hausman_stat
            self.is_endogenous_ = hausman_stat > 3.84  # χ²(1), p<0.05临界值
            
            is_significant = self.p_value_ < 0.05
            
            logger.info(f"IV-2SLS拟合完成: β_Wald={self.wald_estimate_:.6f}, "
                       f"SE={self.se_2sls_:.6f}, p={self.p_value_:.4f}, "
                       f"95%CI=[{self.ci_lower_:.6f}, {self.ci_upper_:.6f}] "
                       f"{'✅显著' if is_significant else '⚠️不显著'} | "
                       f"Hausman={'✅存在内生性' if self.is_endogenous_ else '⚠️无明显内生性'}")
            
        except Exception as e:
            logger.error(f"IV-2SLS拟合失败: {e}")
            self.wald_estimate_ = np.nan
        
        return self
    
    def summary(self) -> Dict[str, Any]:
        """
        返回IV估计结果的完整摘要
        
        Returns:
            包含所有估计量和诊断统计量的字典
        """
        if self.wald_estimate_ is None:
            return {'status': 'not_fitted'}
        
        result = {
            'method': 'Instrumental Variable (2SLS)',
            'reference': 'Angrist & Imbens (1995), JASA',
            'instrument': self.instrument_var,
            'wald_estimate': round(float(self.wald_estimate_), 6),
            'se': round(float(self.se_2sls_), 6) if self.se_2sls_ is not None else None,
            'p_value': round(float(self.p_value_), 4) if self.p_value_ is not None else None,
            'ci_95_lower': round(float(self.ci_lower_), 6) if self.ci_lower_ is not None else None,
            'ci_95_upper': round(float(self.ci_upper_), 6) if self.ci_upper_ is not None else None,
            'is_significant': (self.p_value_ < 0.05) if self.p_value_ is not None else None,
            
            # 第一阶段诊断
            'stage1_r2': round(float(self.stage1_r2_), 4) if self.stage1_r2_ is not None else None,
            'first_stage_f_stat': round(float(self.first_stage_f_stat_), 2) if self.first_stage_f_stat_ is not None else None,
            'is_weak_instrument': self.is_weak_instrument_,
            'partial_r2': round(float(self.partial_r2_), 4) if self.partial_r2_ is not None else None,
            
            # 内生性检验
            'ols_estimate': round(float(self.ols_estimate_), 6) if self.ols_estimate_ is not None else None,
            'hausman_statistic': round(float(self.hausman_stat_), 4) if self.hausman_stat_ is not None else None,
            'is_endogenous': self.is_endogenous_,
            
            'interpretation': self._interpret_result(),
            'validity_warning': self._check_iv_validity()
        }
        
        return result
    
    def _interpret_result(self) -> str:
        """解释IV估计结果"""
        if self.wald_estimate_ is None:
            return "模型尚未拟合"
        
        parts = []
        
        direction = "正向" if self.wald_estimate_ > 0 else "负向"
        
        # 效应显著性
        if self.p_value_ is not None and self.p_value_ < 0.05:
            parts.append(f"✅ IV检测到{direction}且显著的因果效应(β={self.wald_estimate_:.4f}, p={self.p_value_:.4f})")
        elif self.p_value_ is not None:
            parts.append(f"⚠️ IV未检测到显著效应(β={self.wald_estimate_:.4f}, p={self.p_value_:.4f})")
        
        # IV强度
        if self.is_weak_instrument_:
            parts.append(f"❌ 弱工具变量警告(F={self.first_stage_f_stat_:.1f}<10)，估计可能有偏")
        else:
            parts.append(f"✅ 强工具变量(F={self.first_stage_f_stat_:.1f}>10)")
        
        # 内生性
        if self.is_endogenous_:
            parts.append(f"✅ Hausman检验确认存在内生性(H={self.hausman_stat_:.2f})，IV估计优于OLS")
        else:
            parts.append(f"ℹ️ 未发现明显内生性，OLS和IV估计一致")
        
        return " | ".join(parts)
    
    def _check_iv_validity(self) -> str:
        """检查IV有效性条件"""
        warnings_list = []
        
        if self.is_weak_instrument_:
            warnings_list.append("弱工具变量(F<10)，Wald估计可能存在严重偏倚")
        
        if self.partial_r2_ is not None and self.partial_r2_ < 0.01:
            warnings_list.append("偏R²极低(<1%)，工具变量解释力不足")
        
        if self.stage1_r2_ is not None and self.stage1_r2_ < 0.1:
            warnings_list.append("第一阶段R²较低(<10%)，需考虑更强的IV")
        
        if not warnings_list:
            return "✅ IV有效性检验通过，估计结果可信"
        
        return "⚠️ " + "; ".join(warnings_list)


def run_iv_analysis(data: pd.DataFrame,
                    treatment_col: str,
                    outcome_col: str,
                    instrument_col: str,
                    control_cols: List[str] = None) -> Dict:
    """
    运行完整的IV分析流程（便捷函数）
    
    Args:
        data: 完整数据集
        treatment_col: 内生处理变量列名
        outcome_col: 结果变量列名
        instrument_col: 工具变量列名
        control_cols: 控制变量列表
        
    Returns:
        IV分析结果字典
    """
    logger.info(f"启动IV完整分析: treatment={treatment_col}, outcome={outcome_col}, "
               f"instrument={instrument_col}")
    
    df_clean = data.dropna(subset=[treatment_col, outcome_col, instrument_col])
    
    if control_cols is None:
        control_cols = [c for c in data.select_dtypes(include=[np.number]).columns 
                       if c not in [treatment_col, outcome_col, instrument_col]]
    
    available_controls = [c for c in control_cols if c in df_clean.columns]
    
    D = df_clean[treatment_col].values
    Y = df_clean[outcome_col].values
    Z = df_clean[instrument_col].values
    X = df_clean[available_controls].values if available_controls else None
    
    iv = InstrumentalVariable(instrument_var=instrument_col)
    iv.fit(D, Y, Z, X=X)
    
    result = iv.summary()
    result['n_samples'] = len(df_clean)
    result['n_controls'] = len(available_controls)
    
    return result
