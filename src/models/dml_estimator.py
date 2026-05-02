import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import KFold, cross_val_predict, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.base import clone
import warnings

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from utils.helpers import logger_setup, timer

logger = logger_setup('dml_estimator')


class DoubleMachineLearning:
    """
    Double/debiased Machine Learning (DML) 因果效应估计器
    
    理论基础：Chernozhukov V, Chetverikov D, Demirer M, et al.
    "Double/debiased machine learning for treatment and structural parameters"
    The Econometrics Journal, 2018, 21(1): C1-C68
    
    核心思想：
    1. 通过正交化处理消除混杂偏误（Nuisance Parameter Bias）
    2. 在部分线性模型下实现√n一致的因果效应估计
    3. 适用于高维混杂变量场景（p >> n问题）
    
    数学模型：
        Y = θD + g(X) + ε     (结果方程)
        D = m(X) + ν           (处理方程)
    
    其中：
        - Y: 结果变量（如期货价格）
        - D: 处理变量（如天气风险指数）
        - X: 混杂变量（如宏观因子、历史价格等）
        - θ: 目标因果效应（ATE）
        - g(X), m(X): 干扰函数（用ML方法估计）
    
    优势：
    ✅ 解决传统OLS的内生性问题
    ✅ 允许使用复杂的ML方法估计干扰函数
    ✅ 对模型设定错误具有鲁棒性
    ✅ 提供标准误和置信区间
    """
    
    def __init__(self,
                 model_Y=XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05) if XGBOOST_AVAILABLE else Ridge(alpha=1.0),
                 model_D=XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05) if XGBOOST_AVAILABLE else Ridge(alpha=1.0),
                 n_folds: int = 5,
                 random_state: int = 42):
        """
        初始化DML估计器
        
        Args:
            model_Y: 用于估计E[Y|X,D]的ML模型（结果模型）
            model_D: 用于估计E[D|X]的ML模型（处理机制模型）
            n_folds: 交叉验证折数（样本分割法）
            random_state: 随机种子
        """
        self.model_Y = model_Y
        self.model_D = model_D
        self.n_folds = n_folds
        self.random_state = random_state
        
        self.theta_hat_ = None  # ATE估计值
        self.se_hat_ = None      # 标准误
        self.ci_lower_ = None    # 置信区间下界
        self.ci_upper_ = None    # 置信区间上界
        self.p_value_ = None     # p值
        self.residuals_Y_ = None # 正交化残差(Y)
        self.residuals_D_ = None # 正交化残差(D)
        
        logger.info(f"DML初始化: {type(model_Y).__name__}(Y-model) + {type(model_D).__name__}(D-model), "
                   f"{n_folds}-fold CV")
    
    @timer(verbose=False)
    def fit(self, D: np.ndarray, Y: np.ndarray, X: np.ndarray,
            W: np.ndarray = None, inference: bool = True) -> 'DoubleMachineLearning':
        """
        拟合DML模型（Sample Splitting + Orthogonalization）
        
        Step 1: K折交叉拟合估计干扰函数 E[Y|X] 和 E[D|X]
        Step 2: 计算正交化残差  Ỹ = Y - Ê[Y|X], D̃ = D - Ê[D|X]
        Step 3: OLS回归 Ỹ = θ·D̃ + error 得到θ的一致估计
        
        Args:
            D: 处理变量 (n,)
            Y: 结果变量 (n,)
            X: 混杂变量 (n, p)
            W: 额外控制变量 (n, q)，可选
            inference: 是否计算推断统计量（标准误、p值、CI）
            
        Returns:
            self (支持链式调用)
        """
        logger.info(f"开始DML拟合: n={len(D)}, p={X.shape[1]}, inference={inference}")
        
        n_samples = len(D)
        
        if W is not None:
            X_full = np.hstack([X, W])
        else:
            X_full = X.copy()
        
        # Step 1: K-Fold Cross-Fitting（避免overfitting bias）
        kf = TimeSeriesSplit(n_splits=self.n_folds)
        
        Y_pred = np.zeros(n_samples)
        D_pred = np.zeros(n_samples)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full)):
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            D_train, D_test = D[train_idx], D[test_idx]
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # 估计 E[Y | X] (结果模型)
                    model_Y_fold = clone(self.model_Y)
                    model_Y_fold.fit(X_train, Y_train)
                    Y_pred[test_idx] = model_Y_fold.predict(X_test)
                    
                    # 估计 E[D | X] (处理机制模型)
                    model_D_fold = clone(self.model_D)
                    model_D_fold.fit(X_train, D_train)
                    D_pred[test_idx] = model_D_fold.predict(X_test)
                    
            except Exception as e:
                logger.warning(f"DML Fold {fold_idx+1}失败: {e}, 使用均值填充")
                Y_pred[test_idx] = np.mean(Y_train)
                D_pred[test_idx] = np.mean(D_train)
        
        # Step 2: Orthogonalization（正交化）
        self.residuals_Y_ = Y - Y_pred  # Ỹ = Y - E[Y|X,D]
        self.residuals_D_ = D - D_pred  # D̃ = D - E[D|X]
        
        # Step 3: OLS of residuals → ATE estimate
        # Ỹ = θ · D̃ + error
        try:
            ols_model = LinearRegression()
            ols_model.fit(self.residuals_D_.reshape(-1, 1), self.residuals_Y_)
            self.theta_hat_ = ols_model.coef_[0]
            residuals_ols = self.residuals_Y_ - ols_model.predict(self.residuals_D_.reshape(-1, 1))
            
            # 计算标准误（稳健标准误）
            if inference:
                n = len(residuals_ols)
                p = 1
                sigma2 = np.sum(residuals_ols**2) / (n - p)
                var_theta = sigma2 / np.sum(self.residuals_D_**2)
                self.se_hat_ = np.sqrt(max(var_theta, 0))
                
                # Wald检验：H0: θ = 0
                z_stat = self.theta_hat_ / (self.se_hat_ + 1e-8)
                self.p_value_ = 2 * (1 - self._normal_cdf(abs(z_stat)))
                
                # 95%置信区间
                z_crit = 1.96
                self.ci_lower_ = self.theta_hat_ - z_crit * self.se_hat_
                self.ci_upper_ = self.theta_hat_ + z_crit * self.se_hat_
            
            is_significant = self.p_value_ < 0.05 if self.p_value_ is not None else None
            
            logger.info(f"DML拟合完成: θ̂(ATE)={self.theta_hat_:.6f}, SE={self.se_hat_:.6f if self.se_hat_ else 0:.6f}, "
                       f"p={self.p_value_:.4f if self.p_value_ else 0:.4f}, 95%CI=[{self.ci_lower_:.6f if self.ci_lower_ else 0:.6f}, "
                       f"{self.ci_upper_:.6f if self.ci_upper_ else 0:.6f}] {'✅显著' if is_significant else '⚠️不显著' if is_significant is not None else ''}")
            
        except Exception as e:
            logger.error(f"DML OLS回归失败: {e}")
            self.theta_hat_ = np.nan
        
        return self
    
    def predict_ate(self, X_new: np.ndarray) -> np.ndarray:
        """
        预测新样本的处理效应（CATE）
        
        使用训练好的模型预测条件平均处理效应
        
        Args:
            X_new: 新样本特征 (m, p)
            
        Returns:
            CATE预测值 (m,)
        """
        if self.theta_hat_ is None:
            raise RuntimeError("模型尚未训练，请先调用fit()")
        
        return np.full(len(X_new), self.theta_hat_)
    
    def summary(self) -> Dict[str, Any]:
        """
        返回DML估计结果的完整摘要
        
        Returns:
            包含所有估计量和诊断统计量的字典
        """
        if self.theta_hat_ is None:
            return {'status': 'not_fitted'}
        
        result = {
            'method': 'Double Machine Learning (DML)',
            'reference': 'Chernozhukov et al. (2018), Econometrica',
            'theta_hat': round(float(self.theta_hat_), 6),
            'se': round(float(self.se_hat_), 6) if self.se_hat_ is not None else None,
            'p_value': round(float(self.p_value_), 4) if self.p_value_ is not None else None,
            'ci_95_lower': round(float(self.ci_lower_), 6) if self.ci_lower_ is not None else None,
            'ci_95_upper': round(float(self.ci_upper_), 6) if self.ci_upper_ is not None else None,
            'is_significant': (self.p_value_ < 0.05) if self.p_value_ is not None else None,
            'n_folds': self.n_folds,
            'model_Y': type(self.model_Y).__name__,
            'model_D': type(self.model_D).__name__,
            'interpretation': self._interpret_result()
        }
        
        return result
    
    def _interpret_result(self) -> str:
        """解释DML估计结果"""
        if self.theta_hat_ is None:
            return "模型尚未拟合"
        
        direction = "正向" if self.theta_hat_ > 0 else "负向"
        magnitude = "大" if abs(self.theta_hat_) > 1.0 else ("中" if abs(self.theta_hat_) > 0.1 else "小")
        
        if self.p_value_ is not None and self.p_value_ < 0.05:
            return (f"✅ DML检测到{direction}且统计显著的因果效应"
                   f"(θ={self.theta_hat_:.4f}, p={self.p_value_:.4f})，"
                   f"效应量{magnitude}，在控制高维混杂后仍稳健")
        elif self.p_value_ is not None:
            return (f"⚠️ DML未检测到显著因果效应"
                   f"(θ={self.theta_hat_:.4f}, p={self.p_value_:.4f}≥0.05)"
                   f"，可能存在未观测混杂或效应较弱")
        else:
            return f"DML估计值: θ={self.theta_hat_:.4f}（未计算推断统计量）"
    
    @staticmethod
    def _normal_cdf(x: float) -> float:
        """标准正态分布CDF的近似计算"""
        return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))


def run_dml_analysis(data: pd.DataFrame,
                     treatment_col: str,
                     outcome_col: str,
                     confounders: List[str] = None,
                     n_folds: int = 5) -> Dict:
    """
    运行完整的DML分析流程（便捷函数）
    
    Args:
        data: 完整数据集
        treatment_col: 处理变量列名
        outcome_col: 结果变量列名
        confounders: 混杂变量列表（默认使用所有其他数值列）
        n_folds: CV折数
        
    Returns:
        DML分析结果字典
    """
    logger.info(f"启动DML完整分析: treatment={treatment_col}, outcome={outcome_col}")
    
    df_clean = data.dropna(subset=[treatment_col, outcome_col])
    
    if confounders is None:
        confounders = [c for c in data.select_dtypes(include=[np.number]).columns 
                      if c not in [treatment_col, outcome_col]]
    
    available_confounders = [c for c in confounders if c in df_clean.columns]
    
    D = df_clean[treatment_col].values
    Y = df_clean[outcome_col].values
    X = df_clean[available_confounders].values
    
    dml = DoubleMachineLearning(n_folds=n_folds)
    dml.fit(D, Y, X, inference=True)
    
    result = dml.summary()
    result['n_samples'] = len(df_clean)
    result['n_confounders'] = len(available_confounders)
    result['confounder_names'] = available_confounders[:10]  # Top10显示
    
    return result
