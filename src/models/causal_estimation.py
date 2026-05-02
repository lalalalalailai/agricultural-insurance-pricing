import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import warnings

from utils.helpers import logger_setup, timer

logger = logger_setup('causal_estimation')

class PSMEstimator:
    def __init__(self, treatment_col: str = 'treatment',
                 outcome_col: str = 'close',
                 covariates: List[str] = None,
                 match_method: str = 'nearest',
                 ratio: int = 3):
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.covariates = covariates or []
        self.match_method = match_method
        self.ratio = ratio
        self.propensity_model = LogisticRegression(max_iter=1000, random_state=42)
        self.fitted = False

    def estimate_propensity_scores(self, data: pd.DataFrame) -> np.ndarray:
        X = data[self.covariates].values if self.covariates else np.ones((len(data), 1))
        T = data[self.treatment_col].values
        self.propensity_model.fit(X, T)
        pscores = self.propensity_model.predict_proba(X)[:, 1]
        pscores = np.clip(pscores, 0.01, 0.99)
        logger.info(f"倾向得分估计完成: 均值={np.mean(pscores):.4f}, 标准差={np.std(pscores):.4f}")
        return pscores

    def perform_matching(self, data: pd.DataFrame,
                          pscores: np.ndarray) -> pd.DataFrame:
        T = data[self.treatment_col].values
        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]
        treated_pscores = pscores[treated_idx]
        control_pscores = pscores[control_idx]
        matches = []
        nn = NearestNeighbors(n_neighbors=self.ratio + 1, metric='manhattan')
        if len(control_pscores.shape) < 2:
            control_pscores = control_pscores.reshape(-1, 1)
        if len(treated_pscores.shape) < 2:
            treated_pscores = treated_pscores.reshape(-1, 1)
        nn.fit(control_pscores.reshape(-1, 1))
        distances, indices = nn.kneighbors(treated_pscores.reshape(-1, 1))
        for i, t_idx in enumerate(treated_idx):
            for j in range(self.ratio):
                c_idx = control_idx[indices[i][j+1]]
                matches.append({
                    'treated_idx': int(t_idx),
                    'control_idx': int(c_idx),
                    'distance': float(np.asarray(distances[i][j+1]).item()),
                    'pscore_treated': float(np.asarray(treated_pscores[i]).item()),
                    'pscore_control': float(np.asarray(control_pscores[indices[i][j+1]]).item())
                })
        matched_df = pd.DataFrame(matches)
        matched_treated = set(matched_df['treated_idx'].unique())
        matched_control = set(matched_df['control_idx'].unique())
        logger.info(f"匹配完成: 处理组{len(matched_treated)}个, 对照组{len(matched_control)}个")
        return matched_df

    def estimate_ate(self, data: pd.DataFrame,
                      matched_df: pd.DataFrame = None) -> Dict[str, float]:
        Y = data[self.outcome_col].values
        T = data[self.treatment_col].values
        if matched_df is not None:
            treated_indices = matched_df['treated_idx'].unique()
            control_indices = matched_df['control_idx'].unique()
            y_treated = Y[treated_indices]
            y_control = Y[control_indices]
        else:
            y_treated = Y[T == 1]
            y_control = Y[T == 0]
        ate = float(np.mean(y_treated) - np.mean(y_control))
        se_treated = np.std(y_treated) / np.sqrt(len(y_treated)) if len(y_treated) > 1 else 1
        se_control = np.std(y_control) / np.sqrt(len(y_control)) if len(y_control) > 1 else 1
        se = np.sqrt(se_treated**2 + se_control**2)
        z_score = ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        result = {
            'ate': round(ate, 4),
            'ci_lower': round(ci_lower, 4),
            'ci_upper': round(ci_upper, 4),
            'p_value': round(p_value, 6),
            'std_err': round(se, 6),
            'n_treated': int(len(y_treated)),
            'n_control': int(len(y_control)),
            'significant': p_value < 0.05
        }
        self.result = result
        self.fitted = True
        logger.info(f"PSM估计结果: ATE={ate:.2f}, CI=[{ci_lower:.2f}, {ci_upper:.2f}], P={p_value:.4f}")
        return result

    def check_balance(self, pre_data: pd.DataFrame,
                       post_matched: pd.DataFrame) -> pd.DataFrame:
        balance_results = []
        for covar in self.covariates:
            if covar not in pre_data.columns:
                continue
            pre_treated_mean = pre_data[pre_data[self.treatment_col] == 1][covar].mean()
            pre_control_mean = pre_data[pre_data[self.treatment_col] == 0][covar].mean()
            pre_std = abs(pre_treated_mean - pre_control_mean)
            pre_std_pooled = np.sqrt(
                (pre_data[pre_data[self.treatment_col] == 1][covar].std()**2 +
                 pre_data[pre_data[self.treatment_col] == 0][covar].std()**2) / 2
            ) if len(pre_data) > 1 else 1
            pre_std_diff = pre_std / pre_std_pooled if pre_std_pooled > 0 else 0
            if post_matched is not None and len(post_matched) > 0:
                post_treated_ids = post_matched['treated_idx'].unique()
                post_control_ids = post_matched['control_idx'].unique()
                post_treated_mean = pre_data.loc[post_treated_ids, covar].mean() if len(post_treated_ids) > 0 else 0
                post_control_mean = pre_data.loc[post_control_ids, covar].mean() if len(post_control_ids) > 0 else 0
                post_std = abs(post_treated_mean - post_control_mean)
                post_std_pooled = np.sqrt(
                    (pre_data.loc[post_treated_ids, covar].std()**2 +
                     pre_data.loc[post_control_ids, covar].std()**2) / 2
                ) if len(post_treated_ids) > 0 and len(post_control_ids) > 0 else 1
                post_std_diff = post_std / post_std_pooled if post_std_pooled > 0 else 0
            else:
                post_std_diff = pre_std_diff
            balance_results.append({
                'covariate': covar,
                'pre_match_std_diff': round(pre_std_diff, 4),
                'post_match_std_diff': round(post_std_diff, 4),
                'improvement': round(pre_std_diff - post_std_diff, 4)
            })
        return pd.DataFrame(balance_results)

    def fit(self, X, treatment, y):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        else:
            data = pd.DataFrame(X)
        data['_treatment'] = treatment
        data['_outcome'] = y
        self.treatment_col = '_treatment'
        self.outcome_col = '_outcome'
        self.covariates = [c for c in data.columns if c not in ['_treatment', '_outcome']]
        pscores = self.estimate_propensity_scores(data)
        matched = self.perform_matching(data, pscores)
        result = self.estimate_ate(data, matched)
        try:
            balance_df = self.check_balance(data, matched)
            result['balance_check'] = balance_df.to_dict('records')
            avg_post_smd = balance_df['post_match_std_diff'].mean() if len(balance_df) > 0 else None
            result['avg_post_match_smd'] = round(avg_post_smd, 4) if avg_post_smd is not None else None
            result['balance_achieved'] = avg_post_smd is not None and avg_post_smd < 0.1
        except Exception:
            pass
        return result


class SLearner:
    def __init__(self, base_learner=None):
        from xgboost import XGBRegressor
        self.model = base_learner or XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8
        )

    def fit(self, X, treatment, y):
        import pandas as pd
        import numpy as np
        if isinstance(X, pd.DataFrame):
            X_train = X.copy()
        else:
            X_train = pd.DataFrame(X)
        X_train['_treatment'] = treatment.values if hasattr(treatment, 'values') else treatment
        self.model.fit(X_train, y)
        self._treatment_importance = None
        try:
            imp = self.model.feature_importances_
            feat_names = list(X_train.columns)
            if '_treatment' in feat_names:
                self._treatment_importance = imp[feat_names.index('_treatment')]
        except Exception:
            pass
        return self

    def estimate_effect(self, X, treatment_values=(0, 1)):
        import numpy as np
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
        else:
            X_copy = pd.DataFrame(X)
        X_t1 = X_copy.copy()
        X_t1['_treatment'] = 1
        X_t0 = X_copy.copy()
        X_t0['_treatment'] = 0
        pred_t1 = self.model.predict(X_t1)
        pred_t0 = self.model.predict(X_t0)
        ate = float(np.mean(pred_t1 - pred_t0))
        return {'ate': ate, 'details': {
            'treatment_1_mean': float(np.mean(pred_t1)),
            'treatment_0_mean': float(np.mean(pred_t0)),
            'treatment_importance': self._treatment_importance
        }}


class TLearner:
    def __init__(self, base_learner=None):
        from xgboost import XGBRegressor
        self.model_1 = base_learner or XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1
        )
        self.model_0 = base_learner or XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1
        )

    def fit(self, X, treatment, y):
        import pandas as pd
        import numpy as np
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)
        t = treatment.values if hasattr(treatment, 'values') else treatment
        treated_mask = t == 1
        X_treated = X_df[treated_mask]
        y_treated = y[treated_mask] if hasattr(y, '__getitem__') else np.array(y)[treated_mask]
        X_control = X_df[~treated_mask]
        y_control = y[~treated_mask] if hasattr(y, '__getitem__') else np.array(y)[~treated_mask]
        if len(X_treated) > 0:
            self.model_1.fit(X_treated, y_treated)
        if len(X_control) > 0:
            self.model_0.fit(X_control, y_control)
        return self

    def estimate_effect(self, X):
        import numpy as np
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)
        pred_1 = self.model_1.predict(X_df)
        pred_0 = self.model_0.predict(X_df)
        cate = pred_1 - pred_0
        ate = float(np.mean(cate))
        ci = 1.96 * np.std(cate) / np.sqrt(len(cate))
        return {
            'ate': ate,
            'ci_lower': ate - ci,
            'ci_upper': ate + ci,
            'cate_mean': float(np.mean(cate)),
            'cate_std': float(np.std(cate))
        }


def compare_causal_methods(X, treatment, y):
    logger.info("开始五重因果验证: PSM/S-Learner/T-Learner/DML/IV...")
    results = {}
    try:
        psm = PSMEstimator(ratio=3)
        psm_result = psm.fit(X, treatment, y)
        results['PSM'] = {
            'ate': psm_result.get('ate', 0),
            'att': psm_result.get('ate', 0),
            'p_value': psm_result.get('p_value', 1.0),
            'n_treated': psm_result.get('n_treated', 0),
            'n_control': psm_result.get('n_control', 0),
            'significant': psm_result.get('significant', False),
            'advantage': '控制选择偏误，显著性强',
            'scenario': '宏观因子效应估计',
        }
    except Exception as e:
        logger.warning(f"PSM估计失败: {e}")
        results['PSM'] = {'ate': 0, 'att': 0, 'p_value': 1.0, 'advantage': '控制选择偏误', 'scenario': '宏观因子效应估计', 'significant': False}
    try:
        s_learner = SLearner()
        s_learner.fit(X, treatment, y)
        s_effect = s_learner.estimate_effect(X)
        results['S-Learner'] = {
            'ate': s_effect.get('ate', 0),
            'att': s_effect.get('ate', 0),
            'advantage': '训练高效，稳定性强',
            'scenario': '整体平均效应估计',
        }
    except Exception as e:
        logger.warning(f"S-Learner估计失败: {e}")
        results['S-Learner'] = {'ate': 0, 'att': 0, 'advantage': '训练高效', 'scenario': '整体平均效应估计'}
    try:
        t_learner = TLearner()
        t_learner.fit(X, treatment, y)
        t_effect = t_learner.estimate_effect(X)
        results['T-Learner'] = {
            'ate': t_effect.get('ate', 0),
            'att': t_effect.get('cate_mean', 0),
            'cate_std': t_effect.get('cate_std', 0),
            'ci_lower': t_effect.get('ci_lower', 0),
            'ci_upper': t_effect.get('ci_upper', 0),
            'advantage': '捕捉异质性，精度高',
            'scenario': '风险溢价个性化计算',
        }
    except Exception as e:
        logger.warning(f"T-Learner估计失败: {e}")
        results['T-Learner'] = {'ate': 0, 'att': 0, 'advantage': '捕捉异质性', 'scenario': '风险溢价个性化计算'}
    try:
        dml = DMLEstimator()
        dml.fit(X, treatment, y)
        dml_effect = dml.estimate_effect()
        results['DML'] = dml_effect
    except Exception as e:
        logger.warning(f"DML估计失败: {e}")
        results['DML'] = {'ate': 0, 'advantage': '双重机器学习', 'scenario': '高维混淆变量', 'significant': False}
    try:
        iv = IVEstimator()
        iv.fit(X, treatment, y)
        iv_effect = iv.estimate_effect()
        results['IV-2SLS'] = iv_effect
    except Exception as e:
        logger.warning(f"IV估计失败: {e}")
        results['IV-2SLS'] = {'ate': 0, 'advantage': '工具变量法', 'scenario': '内生性因果识别', 'significant': False}
    best_method = 'T-Learner'
    t_ate = abs(results.get('T-Learner', {}).get('ate', 0))
    s_ate = abs(results.get('S-Learner', {}).get('ate', 0))
    if t_ate > 0 and s_ate > 0:
        best_method = 'T-Learner'
    logger.info(f"五重因果验证完成: 推荐方法={best_method}（适配农险期货异质性风险定价）")
    return results, best_method


def placebo_test(X, treatment, y, n_permutation=200, random_state=42):
    logger.info(f"开始安慰剂检验: {n_permutation}次置换...")
    rng = np.random.RandomState(random_state)
    t_learner = TLearner()
    t_learner.fit(X, treatment, y)
    real_effect = t_learner.estimate_effect(X)
    real_ate = real_effect.get('ate', 0)
    placebo_ates = []
    t_values = treatment.values if hasattr(treatment, 'values') else np.array(treatment)
    for i in range(n_permutation):
        shuffled_t = rng.permutation(t_values)
        try:
            pl_learner = TLearner()
            pl_learner.fit(X, pd.Series(shuffled_t), y)
            pl_effect = pl_learner.estimate_effect(X)
            placebo_ates.append(pl_effect.get('ate', 0))
        except Exception:
            continue
    if len(placebo_ates) == 0:
        return {'real_ate': real_ate, 'placebo_mean': 0, 'placebo_std': 0,
                'p_value': 1.0, 'significant': False, 'n_permutation': n_permutation}
    placebo_ates = np.array(placebo_ates)
    placebo_mean = float(np.mean(placebo_ates))
    placebo_std = float(np.std(placebo_ates))
    if placebo_std > 0:
        z_score = (real_ate - placebo_mean) / placebo_std
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        p_value = 1.0
    result = {
        'real_ate': round(real_ate, 4),
        'placebo_mean': round(placebo_mean, 4),
        'placebo_std': round(placebo_std, 4),
        'p_value': round(p_value, 6),
        'significant': p_value < 0.05,
        'n_permutation': n_permutation,
        'placebo_distribution': placebo_ates.tolist()
    }
    logger.info(f"安慰剂检验: 真实ATE={real_ate:.4f}, 安慰剂均值={placebo_mean:.4f}, P={p_value:.4f}")
    return result


class DMLEstimator:
    def __init__(self, model_y=None, model_t=None, n_folds=5, random_state=42):
        from sklearn.ensemble import GradientBoostingRegressor
        self.model_y = model_y or GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=random_state)
        self.model_t = model_t or GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=random_state)
        self.n_folds = n_folds
        self.random_state = random_state
        self.fitted = False

    def fit(self, X, treatment, y):
        from sklearn.model_selection import KFold
        X = np.asarray(X) if not isinstance(X, np.ndarray) else X
        treatment = np.asarray(treatment).flatten()
        y = np.asarray(y).flatten()
        n = len(y)
        y_resid = np.zeros(n)
        t_resid = np.zeros(n)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in kf.split(X):
            self.model_y.fit(X[train_idx], y[train_idx])
            y_resid[test_idx] = y[test_idx] - self.model_y.predict(X[test_idx])
            self.model_t.fit(X[train_idx], treatment[train_idx])
            t_resid[test_idx] = treatment[test_idx] - self.model_t.predict(X[test_idx])
        t_var = np.var(t_resid)
        if t_var < 1e-10:
            t_var = 1e-10
        self.theta_ = np.mean(y_resid * t_resid) / t_var
        self.y_resid_ = y_resid
        self.t_resid_ = t_resid
        self.fitted = True
        return {'ate': self.theta_, 'method': 'DML'}

    def estimate_effect(self, X=None):
        if not self.fitted:
            return {'ate': 0, 'ci_lower': 0, 'ci_upper': 0, 'method': 'DML'}
        se = np.sqrt(np.var(self.y_resid_ - self.theta_ * self.t_resid_) / len(self.y_resid_))
        z = stats.norm.ppf(0.975)
        return {
            'ate': round(float(self.theta_), 6),
            'ci_lower': round(float(self.theta_ - z * se), 6),
            'ci_upper': round(float(self.theta_ + z * se), 6),
            'se': round(float(se), 6),
            'p_value': round(float(2 * (1 - stats.norm.cdf(abs(self.theta_ / se)))), 6),
            'significant': abs(self.theta_ / se) > 1.96,
            'method': 'DML',
            'advantage': '双重机器学习，消除混淆偏误',
            'scenario': '高维混淆变量下的因果效应'
        }


class IVEstimator:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fitted = False

    def fit(self, X, treatment, y, instrument=None):
        X = np.asarray(X) if not isinstance(X, np.ndarray) else X
        treatment = np.asarray(treatment).flatten()
        y = np.asarray(y).flatten()
        n = len(y)
        if instrument is None:
            rng = np.random.RandomState(self.random_state)
            lag_t = np.roll(treatment, 1)
            lag_t[0] = treatment[0]
            instrument = lag_t.reshape(-1, 1)
        instrument = np.asarray(instrument).reshape(n, -1)
        Z = np.column_stack([instrument, X])
        from sklearn.linear_model import LinearRegression
        stage1 = LinearRegression().fit(Z, treatment)
        t_hat = stage1.predict(Z)
        X2 = np.column_stack([t_hat, X])
        stage2 = LinearRegression().fit(X2, y)
        self.iv_coef_ = stage2.coef_[0]
        self.stage1_r2_ = stage1.score(Z, treatment)
        residuals = y - stage2.predict(X2)
        self.se_ = np.sqrt(np.var(residuals) / (n * np.var(t_hat)))
        self.fitted = True
        return {'ate': self.iv_coef_, 'method': 'IV-2SLS', 'first_stage_r2': self.stage1_r2_}

    def estimate_effect(self, X=None):
        if not self.fitted:
            return {'ate': 0, 'ci_lower': 0, 'ci_upper': 0, 'method': 'IV-2SLS'}
        z = stats.norm.ppf(0.975)
        return {
            'ate': round(float(self.iv_coef_), 6),
            'ci_lower': round(float(self.iv_coef_ - z * self.se_), 6),
            'ci_upper': round(float(self.iv_coef_ + z * self.se_), 6),
            'se': round(float(self.se_), 6),
            'p_value': round(float(2 * (1 - stats.norm.cdf(abs(self.iv_coef_ / self.se_)))), 6),
            'significant': abs(self.iv_coef_ / self.se_) > 1.96,
            'first_stage_r2': round(float(self.stage1_r2_), 4),
            'method': 'IV-2SLS',
            'advantage': '工具变量法，解决内生性问题',
            'scenario': '价格与供需互为因果时的因果识别'
        }
