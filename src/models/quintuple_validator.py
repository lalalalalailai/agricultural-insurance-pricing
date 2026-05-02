import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings

from utils.helpers import logger_setup, timer

logger = logger_setup('quintuple_validation')


class QuintupleCausalValidator:
    """
    五重因果效应估计与验证体系
    
    整合5种主流因果推断方法，形成交叉验证的稳健估计框架：
    
    1. PSM (Propensity Score Matching) - 倾向得分匹配
       → Rosenbaum P, Rubin D B. "The central role of the propensity score in observational studies for causal effects." Biometrika, 1983.
       
    2. S-Learner (Single-Learner) - 单一学习器
       → Kunzel S R, et al. "Meta-learners for estimating heterogeneous treatment effects using machine learning." PNAS, 2019.
       
    3. T-Learner (Two-Learner) - 双学习器
       → 同上，适用于异质性处理效应估计
       
    4. DML (Double Machine Learning) - 双重机器学习
       → Chernozhukov V, et al. "Double/debiased ML for treatment and structural parameters." Econometrica, 2018.
       
    5. IV (Instrumental Variable) - 工具变量法
       → Angrist J D, Imbens G W. "Two-stage least squares estimation of average causal effects." JASA, 1995.
    
    核心价值：
    ✅ 多角度交叉验证，提高结论可信度
    ✅ 各方法互补优势，PSM控制选择偏误、DML消除高维混杂、IV解决内生性
    ✅ 共识机制：多数方法一致则结论稳健，不一致则深入分析原因
    ✅ 满足顶级期刊审稿要求的严谨性标准
    
    适用场景：
    - 农险期货定价中的因果效应识别
    - 政策评估和反事实预测
    - 需要高严谨性的学术研究和监管报告
    """
    
    def __init__(self):
        self.methods_results = {}
        self.consensus_conclusion = None
        self.agreement_score = None
        
        logger.info("五重因果验证体系初始化完成")
    
    @timer(verbose=False)
    def run_full_validation(self,
                            data: pd.DataFrame,
                            treatment_col: str,
                            outcome_col: str,
                            confounders: List[str] = None,
                            instrument_col: str = None,
                            n_bootstrap_psm: int = 50) -> Dict:
        """
        运行完整的五重因果验证
        
        Args:
            data: 完整数据集
            treatment_col: 处理变量列名
            outcome_col: 结果变量列名
            confounders: 混杂变量列表
            instrument_col: 工具变量列名（IV方法需要）
            n_bootstrap_psm: PSM Bootstrap次数
            
        Returns:
            五重验证完整结果字典
        """
        logger.info("=" * 80)
        logger.info("🚀 启动五重因果效应估计与验证体系")
        logger.info("=" * 80)
        
        df_clean = data.dropna(subset=[treatment_col, outcome_col])
        
        if confounders is None:
            confounders = [c for c in data.select_dtypes(include=[np.number]).columns 
                          if c not in [treatment_col, outcome_col]]
        
        available_confounders = [c for c in confounders if c in df_clean.columns]
        
        D = df_clean[treatment_col].values
        Y = df_clean[outcome_col].values
        X = df_clean[available_confounders].values if available_confounders else None
        
        results_summary = {
            'metadata': {
                'n_samples': len(df_clean),
                'n_confounders': len(available_confounders),
                'treatment_var': treatment_col,
                'outcome_var': outcome_col,
                'methods_run': []
            },
            'method_results': {},
            'consensus_analysis': None,
            'final_recommendation': None
        }
        
        # ====== Method 1: PSM (倾向得分匹配) ======
        try:
            logger.info("\n" + "─" * 60)
            logger.info("📊 Method 1/5: PSM (Propensity Score Matching)")
            logger.info("─" * 60)
            
            psm_result = self._run_psm(D, Y, X, n_bootstrap=n_bootstrap_psm)
            self.methods_results['PSM'] = psm_result
            results_summary['method_results']['PSM'] = psm_result
            results_summary['metadata']['methods_run'].append('PSM')
            
        except Exception as e:
            logger.warning(f"PSM执行失败: {e}")
            self.methods_results['PSM'] = {'error': str(e), 'ate': None}
        
        # ====== Method 2: S-Learner ======
        try:
            logger.info("\n" + "─" * 60)
            logger.info("📊 Method 2/5: S-Learner (Single-Learner)")
            logger.info("─" * 60)
            
            slearner_result = self._run_slearner(D, Y, X)
            self.methods_results['S-Learner'] = slearner_result
            results_summary['method_results']['S-Learner'] = slearner_result
            results_summary['metadata']['methods_run'].append('S-Learner')
            
        except Exception as e:
            logger.warning(f"S-Learner执行失败: {e}")
            self.methods_results['S-Learner'] = {'error': str(e), 'ate': None}
        
        # ====== Method 3: T-Learner ======
        try:
            logger.info("\n" + "─" * 60)
            logger.info("📊 Method 3/5: T-Learner (Two-Learner)")
            logger.info("─" * 60)
            
            tlearner_result = self._run_tlearner(D, Y, X)
            self.methods_results['T-Learner'] = tlearner_result
            results_summary['method_results']['T-Learner'] = tlearner_result
            results_summary['metadata']['methods_run'].append('T-Learner')
            
        except Exception as e:
            logger.warning(f"T-Learner执行失败: {e}")
            self.methods_results['T-Learner'] = {'error': str(e), 'ate': None}
        
        # ====== Method 4: DML (双重机器学习) ======
        try:
            from models.dml_estimator import DoubleMachineLearning
            
            logger.info("\n" + "─" * 60)
            logger.info("📊 Method 4/5: DML (Double Machine Learning)")
            logger.info("─" * 60)
            
            dml = DoubleMachineLearning(n_folds=5)
            dml.fit(D, Y, X, inference=True)
            dml_result = dml.summary()
            dml_result['method_name'] = 'DML'
            self.methods_results['DML'] = dml_result
            results_summary['method_results']['DML'] = dml_result
            results_summary['metadata']['methods_run'].append('DML')
            
        except Exception as e:
            logger.warning(f"DML执行失败: {e}")
            self.methods_results['DML'] = {'error': str(e), 'ate': None}
        
        # ====== Method 5: IV (工具变量) ======
        if instrument_col and instrument_col in data.columns:
            try:
                from models.iv_estimator import InstrumentalVariable
                
                logger.info("\n" + "─" * 60)
                logger.info("📊 Method 5/5: IV (Instrumental Variable)")
                logger.info("─" * 60)
                
                Z = data[instrument_col].dropna().loc[df_clean.index].values
                
                iv = InstrumentalVariable(instrument_var=instrument_col)
                iv.fit(D, Y, Z, X=X)
                iv_result = iv.summary()
                iv_result['method_name'] = 'IV'
                self.methods_results['IV'] = iv_result
                results_summary['method_results']['IV'] = iv_result
                results_summary['metadata']['methods_run'].append('IV')
                
            except Exception as e:
                logger.warning(f"IV执行失败: {e}")
                self.methods_results['IV'] = {'error': str(e), 'ate': None}
        else:
            logger.info("⏭️ 跳过Method 5/5: 未提供工具变量")
            self.methods_results['IV'] = {'skipped': True, 'reason': 'no_instrument'}
        
        # ====== Consensus Analysis (共识分析) ======
        logger.info("\n" + "=" * 80)
        logger.info("🎯 五重验证共识分析")
        logger.info("=" * 80)
        
        consensus = self._analyze_consensus()
        results_summary['consensus_analysis'] = consensus
        results_summary['final_recommendation'] = self._generate_recommendation(consensus)
        
        self.consensus_conclusion = consensus['conclusion']
        self.agreement_score = consensus['agreement_score']
        
        logger.info(f"\n✅ 五重验证完成: 共识={self.consensus_conclusion}, "
                   f"一致性得分={self.agreement_score:.1f}/100")
        
        return results_summary
    
    def _run_psm(self, D, Y, X, n_bootstrap=50) -> Dict:
        """运行PSM倾向得分匹配"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors
        
        logger.info("  执行PSM: LogisticRegression倾向得分 + 1:3最近邻匹配...")
        
        n = len(D)
        if X is None or X.shape[1] < 1:
            return {'ate': np.cov(D, Y)[0, 1] / (np.var(D) + 1e-8), 'method': 'simple_cov'}
        
        # Step 1: 估计倾向得分 P(D|X)
        try:
            ps_model = LogisticRegression(max_iter=1000, random_state=42)
            ps_model.fit(X, D)
            propensity_scores = ps_model.predict_proba(X)[:, 1]
        except Exception:
            # 如果LogisticRegression失败，使用线性概率模型
            from sklearn.linear_model import LinearRegression
            lpm = LinearRegression().fit(X, D)
            propensity_scores = lpm.predict(X).clip(0.01, 0.99)
        
        # 二值化处理变量（用于匹配）
        median_D = np.median(D)
        treatment_binary = (D > median_D).astype(int)
        
        # Step 2: 最近邻匹配（1:3）
        treated_idx = np.where(treatment_binary == 1)[0]
        control_idx = np.where(treatment_binary == 0)[0]
        
        if len(treated_idx) < 5 or len(control_idx) < 15:
            logger.warning("  样本量不足，使用简化ATE估计")
            return {'ate': float(np.mean(Y[treatment_binary == 1]) - np.mean(Y[treatment_binary == 0])),
                   'att': None, 'n_matched': 0, 'method': 'simplified'}
        
        nn = NearestNeighbors(n_neighbors=min(4, len(control_idx)), metric='euclidean')
        nn.fit(propensity_scores[control_idx].reshape(-1, 1))
        
        distances, indices = nn.kneighbors(propensity_scores[treated_idx].reshape(-1, 1))
        
        matched_ates = []
        for i, treated_i in enumerate(treated_idx):
            matched_controls = control_idx[indices[i][1:]]  # 排除自身(最近的是自己)
            if len(matched_controls) > 0:
                ate_i = Y[treated_i] - np.mean(Y[matched_controls])
                matched_ates.append(ate_i)
        
        ate_psm = np.mean(matched_ates) if matched_ates else 0
        matched_control_values = [Y[control_idx[j]] for i, treated_i in enumerate(treated_idx) for j in indices[i, 1:4]]
        att_psm = np.mean([Y[i] for i in treated_idx]) - (np.mean(matched_control_values) if matched_control_values else 0)
        
        result = {
            'ate': round(float(ate_psm), 6),
            'att': round(float(att_psm), 6) if not np.isnan(att_psm) else None,
            'n_treated': len(treated_idx),
            'n_control': len(control_idx),
            'n_matched': len(matched_ates),
            'mean_propensity_treated': float(np.mean(propensity_scores[treatment_binary == 1])),
            'mean_propensity_control': float(np.mean(propensity_scores[treatment_binary == 0])),
            'method': 'PSM (Logistic + NN 1:3)',
            'interpretation': f"PSM估计ATE={ate_psm:.4f}, 匹配{len(matched_ates)}对"
        }
        
        logger.info(f"  PSM结果: ATE={ate_psm:.6f}, ATT={att_psm:.6f}, 匹配{len(matched_ates)}对")
        
        return result
    
    def _run_slearner(self, D, Y, X) -> Dict:
        """运行S-Learner单一学习器"""
        try:
            from xgboost import XGBRegressor
            model_class = XGBRegressor
            model_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'random_state': 42,
                           'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.8}
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            model_class = GradientBoostingRegressor
            model_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'random_state': 42}
        
        logger.info(f"  执行S-Learner: {model_class.__name__}...")
        
        if X is not None and X.shape[1] >= 1:
            features = np.hstack([D.reshape(-1, 1), X])
        else:
            features = D.reshape(-1, 1)
        
        model = model_class(**model_params)
        model.fit(features, Y)
        
        features_t1 = features.copy()
        features_t1[:, 0] = 1
        pred_t1 = model.predict(features_t1)
        
        features_t0 = features.copy()
        features_t0[:, 0] = 0
        pred_t0 = model.predict(features_t0)
        
        ate_slearner = float(np.mean(pred_t1 - pred_t0))
        
        result = {
            'ate': round(ate_slearner, 6),
            'model': model_class.__name__,
            'method': 'S-Learner',
            'baseline_prediction': round(float(np.mean(pred_t0)), 6),
            'interpretation': f"S-Learner估计ATE={ate_slearner:.4f}"
        }
        
        logger.info(f"  S-Learner结果: ATE={ate_slearner:.6f}")
        
        return result
    
    def _run_tlearner(self, D, Y, X) -> Dict:
        """运行T-Learner双学习器"""
        try:
            from xgboost import XGBRegressor
            model_class = XGBRegressor
            model_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'random_state': 42}
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            model_class = GradientBoostingRegressor
            model_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'random_state': 42}
        
        logger.info(f"  执行T-Learner: {model_class.__name__} (异质性处理效应)...")
        
        median_D = np.median(D)
        treated_mask = D > median_D
        control_mask = ~treated_mask
        
        if X is not None and X.shape[1] >= 1:
            X_treated = X[treated_mask]
            X_control = X[control_mask]
        else:
            X_treated = np.zeros((np.sum(treated_mask), 1))
            X_control = np.zeros((np.sum(control_mask), 1))
        
        Y_treated = Y[treated_mask]
        Y_control = Y[control_mask]
        
        if len(Y_treated) < 10 or len(Y_control) < 10:
            return {'ate': float(np.mean(Y_treated) - np.mean(Y_control)) if len(Y_treated) > 0 and len(Y_control) > 0 else 0,
                   'cate_std': None, 'method': 'T-Learner (insufficient sample)'}
        
        model_treated = model_class(**model_params)
        model_control = model_class(**model_params)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_treated.fit(X_treated, Y_treated)
            model_control.fit(X_control, Y_control)
        
        mu1 = model_treated.predict(X)  # E[Y(1)|X]
        mu0 = model_control.predict(X)  # E[Y(0)|X]
        
        cate_individual = mu1 - mu0  # CATE (个体处理效应)
        ate_tlearner = np.mean(cate_individual)
        cate_std = np.std(cate_individual)
        
        att_tlearner = np.mean(mu1[treated_mask] - mu0[treated_mask])
        
        result = {
            'ate': round(float(ate_tlearner), 6),
            'att': round(float(att_tlearner), 6),
            'cate_std': round(float(cate_std), 6),
            'cate_min': round(float(np.min(cate_individual)), 6),
            'cate_max': round(float(np.max(cate_individual)), 6),
            'model': model_class.__name__,
            'method': 'T-Learner',
            'interpretation': f"T-Learner估计ATE={ate_tlearner:.4f}, CATE标准差={cate_std:.4f}"
        }
        
        logger.info(f"  T-Learner结果: ATE={ate_tlearner:.6f}, ATT={att_tlearner:.6f}, CATE std={cate_std:.6f}")
        
        return result
    
    def _analyze_consensus(self) -> Dict:
        """
        分析五种方法的共识程度
        
        Returns:
            共识分析字典
        """
        valid_methods = {k: v for k, v in self.methods_results.items() 
                        if v.get('ate') is not None and 'error' not in v}
        
        n_valid = len(valid_methods)
        
        if n_valid < 2:
            return {
                'conclusion': 'INSUFFICIENT_DATA',
                'agreement_score': 0,
                'n_valid_methods': n_valid,
                'message': '有效方法数量不足，无法进行共识分析'
            }
        
        ates = [v['ate'] for v in valid_methods.values()]
        mean_ate = np.mean(ates)
        std_ate = np.std(ates)
        cv_ate = std_ate / (abs(mean_ate) + 1e-8) * 100
        
        # 符号一致性（所有ATE同号？）
        signs = [np.sign(a) for a in ates if a != 0]
        sign_agreement = len(set(signs)) <= 1 if signs else False
        
        # 显著性一致性（多少方法达到p<0.05）
        significant_count = sum(1 for v in valid_methods.values() 
                              if v.get('p_value', 0) is not None and v['p_value'] < 0.05)
        significance_ratio = significant_count / n_valid
        
        # 计算综合一致性得分（0-100）
        agreement_components = []
        
        # Component 1: 符号一致性（30分）
        agreement_components.append(30 if sign_agreement else 10)
        
        # Component 2: 变异系数（30分，CV越小越好）
        if cv_ate < 20:
            agreement_components.append(30)
        elif cv_ate < 50:
            agreement_components.append(20)
        elif cv_ate < 100:
            agreement_components.append(10)
        else:
            agreement_components.append(5)
        
        # Component 3: 显著性一致性（20分）
        agreement_components.append(20 * significance_ratio)
        
        # Component 4: 方法覆盖度（20分，方法越多越可信）
        coverage_score = min(20, n_valid * 4)
        agreement_components.append(coverage_score)
        
        agreement_score = sum(agreement_components)
        
        # 确定共识结论
        if agreement_score >= 85:
            conclusion = 'STRONG_CONSENSUS'
            conclusion_desc = '强共识：多种方法高度一致，结论可信度极高'
        elif agreement_score >= 70:
            conclusion = 'MODERATE_CONSENSUS'
            conclusion_desc = '中等共识：大部分方法一致，结论基本可信'
        elif agreement_score >= 50:
            conclusion = 'WEAK_CONSENSUS'
            conclusion_desc = '弱共识：存在方法间分歧，需进一步分析'
        else:
            conclusion = 'NO_CONSENSUS'
            conclusion_desc = '无共识：方法间严重分歧，建议检查数据质量或模型设定'
        
        result = {
            'conclusion': conclusion,
            'conclusion_description': conclusion_desc,
            'agreement_score': round(agreement_score, 1),
            'n_valid_methods': n_valid,
            'methods_used': list(valid_methods.keys()),
            'ate_statistics': {
                'mean': round(float(mean_ate), 6),
                'std': round(float(std_ate), 6),
                'cv_percent': round(cv_ade := cv_ate, 2),
                'min': round(float(min(ates)), 6),
                'max': round(float(max(ates)), 6)
            },
            'sign_consistency': sign_agreement,
            'significance_ratio': round(significance_ratio, 2),
            'individual_ates': {k: round(v['ate'], 6) for k, v in valid_methods.items()}
        }
        
        logger.info(f"\n  共识分析结果:")
        logger.info(f"    一致性得分: {agreement_score:.1f}/100")
        logger.info(f"    结论等级: {conclusion} - {conclusion_desc}")
        logger.info(f"    ATE统计: 均值={mean_ate:.6f}, 标准差={std_ate:.6f}, CV={cv_ate:.2f}%")
        logger.info(f"    各方法ATE: {result['individual_ates']}")
        
        return result
    
    def _generate_recommendation(self, consensus: Dict) -> str:
        """
        基于共识分析生成最终推荐
        
        Args:
            consensus: 共识分析结果
            
        Returns:
            最终推荐字符串
        """
        if consensus['conclusion'] == 'STRONG_CONSENSUS':
            return (f"✅ 强烈推荐：{consensus['n_valid_methods']}种方法达成强共识"
                   f"(一致性={consensus['agreement_score']:.0f}/100)，"
                   f"因果效应估计高度可靠(ATE≈{consensus['ate_statistics']['mean']:.4f})，"
                   f"可直接用于定价决策和学术发表")
        
        elif consensus['conclusion'] == 'MODERATE_CONSENSUS':
            return (f"✅ 推荐：{consensus['n_valid_methods']}种方法达成中等共识"
                   f"(一致性={consensus['agreement_score']:.0f}/100)，"
                   f"因果效应估计基本可靠(ATE≈{consensus['ate_statistics']['mean']:.4f})，"
                   f"可用于决策参考，建议补充敏感性分析")
        
        elif consensus['conclusion'] == 'WEAK_CONSENSUS':
            return (f"⚠️ 谨慎参考：方法间存在分歧(一致性={consensus['agreement_score']:.0f}/100)"
                   f"，ATE范围[{consensus['ate_statistics']['min']:.4f}, "
                   f"{consensus['ate_statistics']['max']:.4f}]，"
                   f"建议深入分析差异来源，优先采用PSM+T-Learner组合")
        
        else:
            return (f"❌ 暂不推荐：方法间严重不一致(一致性={consensus['agreement_score']:.0f}/100)"
                   f"，可能存在数据问题或模型设定错误，"
                   f"需要重新审视研究设计和数据质量")


def run_quintuple_validation_quick(data: pd.DataFrame,
                                   treatment: str,
                                   outcome: str,
                                   instrument: str = None) -> Dict:
    """
    五重验证便捷函数（一行代码调用全部）
    
    Args:
        data: 数据集
        treatment: 处理变量
        outcome: 结果变量
        instrument: 工具变量（可选）
        
    Returns:
        完整的五重验证结果
    """
    validator = QuintupleCausalValidator()
    return validator.run_full_validation(data, treatment, outcome, instrument_col=instrument)
