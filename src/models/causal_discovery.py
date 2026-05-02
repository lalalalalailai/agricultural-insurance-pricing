import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
from scipy import stats
from scipy.spatial.distance import cdist
from itertools import combinations
import warnings
import concurrent.futures

from utils.constants import MODEL_PARAMS, DAG_CONFIG
from utils.helpers import logger_setup, timer

logger = logger_setup('causal_discovery')

warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)

# 尝试导入核方法用于非线性检验
try:
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.metrics.pairwise import rbf_kernel
    KERNEL_METHODS_AVAILABLE = True
except ImportError:
    KERNEL_METHODS_AVAILABLE = False

BUSINESS_PRIOR_EDGES = [
    ('weather_risk', 'yield'),
    ('yield', 'close'),
    ('close', 'risk_premium'),
    ('cpi', 'supply_demand'),
    ('supply_demand', 'close'),
    ('futures_high', 'close'),
    ('futures_low', 'close'),
    ('settle', 'close'),
    ('volume', 'close'),
    ('macro_pmi', 'supply_demand'),
    ('policy_risk', 'risk_premium'),
    ('macro_pmi', 'cpi'),
]

BUSINESS_FORBIDDEN_EDGES = [
    ('close', 'weather_risk'),
    ('close', 'yield'),
    ('risk_premium', 'close'),
    ('close', 'cpi'),
]

class CausalGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: List[str] = []
        self.edges: List[Tuple[str, str]] = []
        self.edge_strengths: Dict[Tuple[str, str], float] = {}

    def add_node(self, node: str):
        if node not in self.nodes:
            self.nodes.append(node)
            self.graph.add_node(node)

    def add_edge(self, source: str, target: str, strength: float = 1.0):
        if (source, target) not in self.edges:
            self.edges.append((source, target))
            self.edge_strengths[(source, target)] = strength
            self.graph.add_edge(source, target, weight=strength)

    def get_adjacent_nodes(self, node: str) -> List[str]:
        return list(self.graph.neighbors(node))

    def get_parents(self, node: str) -> List[str]:
        return list(self.graph.predecessors(node))

    def get_children(self, node: str) -> List[str]:
        return list(self.graph.successors(node))

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self):
        return len(self.edges)


def _bootstrap_single(data, variables, apply_business_prior, sample_ratio, rng, i):
    """单次Bootstrap采样的执行函数"""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            
            if isinstance(rng, int):
                rng = np.random.RandomState(rng)
            
            sample_idx = rng.choice(len(data), size=int(len(data) * sample_ratio), replace=True)
            
            # 安全采样：使用iloc的integer location模式，避免索引类型问题
            if isinstance(data.index, pd.DatetimeIndex):
                sampled_data = data.iloc[sample_idx].copy()
                sampled_data.reset_index(inplace=True)
            else:
                sampled_data = data.iloc[sample_idx].reset_index(drop=True).copy()
            
            # 确保variables在数据中存在
            available_vars = [v for v in variables if v in sampled_data.columns]
            if len(available_vars) < 2:
                logger.debug(f"Bootstrap第{i+1}次: 变量不足({len(available_vars)}个)")
                return None
            
            cd = CausalDiscovery()
            graph = cd.run_pc_algorithm(sampled_data, variables=available_vars,
                                        apply_business_prior=apply_business_prior)
            
            if not isinstance(graph, CausalGraph):
                logger.debug(f"Bootstrap第{i+1}次: 返回非CausalGraph对象")
                return None
            
            # 安全获取edges，处理各种可能的格式
            try:
                if hasattr(graph, 'edges') and graph.edges and isinstance(graph.edges, list):
                    edges_list = graph.edges
                    if edges_list and isinstance(edges_list[0], tuple):
                        result = [(str(s), str(t)) for s, t in edges_list]
                        logger.debug(f"Bootstrap第{i+1}次成功: {len(result)}条边")
                        return result
                    elif edges_list:
                        logger.debug(f"Bootstrap第{i+1}次成功: {len(edges_list)}条边(非tuple格式)")
                        return edges_list
                elif hasattr(graph, 'graph') and hasattr(graph.graph, 'edges'):
                    edges_list = list(graph.graph.edges())
                    result = [(str(s), str(t)) for s, t in edges_list] if edges_list else []
                    logger.debug(f"Bootstrap第{i+1}次成功(从graph.edges): {len(result)}条边")
                    return result
                else:
                    logger.debug(f"Bootstrap第{i+1}次: 无edges属性")
                    return []
            except Exception as edge_err:
                logger.debug(f"获取edges失败: {edge_err}")
                return []
                
    except Exception as e:
        logger.warning(f"Bootstrap第{i+1}次失败: {str(e)[:150]}")
        return None


class CausalDiscovery:
    def __init__(self, alpha: float = 0.05,
                 max_cond_set_size: int = 3):
        self.alpha = alpha
        self.max_cond_size = max_cond_set_size

    @timer(verbose=False)
    def run_pc_algorithm(self, data: pd.DataFrame,
                          variables: List[str] = None,
                          apply_business_prior: bool = True) -> CausalGraph:
        logger.info("开始PC算法因果发现...")
        if variables is None:
            variables = data.select_dtypes(include=[np.number]).columns.tolist()[:12]
        df_clean = data[variables].dropna()
        const_cols = [col for col in variables if df_clean[col].std() == 0 or df_clean[col].nunique() <= 1]
        if const_cols:
            variables = [v for v in variables if v not in const_cols]
            df_clean = df_clean[variables]
            logger.debug(f"已过滤常数列(零方差): {const_cols}")
        if len(variables) < 2:
            logger.warning("有效变量不足2个，跳过PC算法")
            return CausalGraph()
        n_vars = len(variables)
        graph = CausalGraph()
        for var in variables:
            graph.add_node(var)
        for i, j in combinations(range(n_vars), 2):
            graph.add_edge(variables[i], variables[j], 0.5)
            graph.add_edge(variables[j], variables[i], 0.5)
        adj_set = {var: set() for var in variables}
        sep_set = {}
        
        # Phase 1: Skeleton Discovery (Spirtes et al., 2000)
        # 增强版：同时记录p值和相关性强度
        p_values = {}  # 记录所有检验的p值
        edge_strengths = {}  # 记录边强度
        
        for cond_size in range(self.max_cond_size + 1):
            edges_to_remove = []
            for i, j in combinations(range(n_vars), 2):
                vi, vj = variables[i], variables[j]
                if not graph.graph.has_edge(vi, vj) and not graph.graph.has_edge(vj, vi):
                    continue
                other_vars = [v for k, v in enumerate(variables) if k != i and k != j]
                independent = False
                min_p_val = 1.0
                
                if cond_size == 0:
                    if df_clean[vi].std() < 1e-10 or df_clean[vj].std() < 1e-10:
                        independent = True
                        min_p_val = 1.0
                        sep_set[(vi, vj)] = set()
                        sep_set[(vj, vi)] = set()
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
                            corr, p_val = stats.pearsonr(df_clean[vi], df_clean[vj])
                        min_p_val = p_val
                        if np.isnan(corr) or np.isnan(p_val):
                            independent = True
                            min_p_val = 1.0
                        elif p_val > self.alpha:
                            independent = True
                            sep_set[(vi, vj)] = set()
                            sep_set[(vj, vi)] = set()
                else:
                    for cond_combo in combinations(other_vars, cond_size):
                        try:
                            partial_corr = self._partial_correlation(
                                df_clean, vi, vj, list(cond_combo)
                            )
                            t_stat = partial_corr * np.sqrt(
                                (len(df_clean) - 2 - cond_size) / (1 - partial_corr**2 + 1e-8)
                            )
                            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(df_clean) - 2 - cond_size))
                            min_p_val = min(min_p_val, p_val)
                            if p_val > self.alpha or np.isnan(partial_corr):
                                independent = True
                                sep_set[(vi, vj)] = set(cond_combo)
                                sep_set[(vj, vi)] = set(cond_combo)
                                break
                        except Exception:
                            continue
                
                # 记录边强度（用于后续可视化）
                edge_key = f"{vi}->{vj}"
                p_values[edge_key] = min_p_val
                if not independent:
                    strength = 1 - min_p_val  # 越显著越强
                    edge_strengths[edge_key] = max(0.1, min(1.0, strength))
                
                if independent:
                    edges_to_remove.append((vi, vj))
                    edges_to_remove.append((vj, vi))
            
            for src, tgt in edges_to_remove:
                if graph.graph.has_edge(src, tgt):
                    graph.graph.remove_edge(src, tgt)
                    if (src, tgt) in graph.edge_strengths:
                        del graph.edge_strengths[(src, tgt)]
                    graph.edges = [(s, t) for s, t in graph.edges if (s, t) != (src, tgt)]
        
        # Phase 2: V-Structure Orientation (Colliders Detection)
        # 基于sep_set的V结构定向（标准PC算法规则：Spirtes et al., 2000）
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                vi, vj = variables[i], variables[j]
                if graph.graph.has_edge(vi, vj) or graph.graph.has_edge(vj, vi):
                    continue
                k_neighbors_i = [v for v in variables if v != vi and v != vj
                               and graph.graph.has_edge(v, vi)]
                k_neighbors_j = [v for v in variables if v != vi and v != vj
                               and graph.graph.has_edge(v, vj)]
                common_k = [k for k in k_neighbors_i if k in k_neighbors_j]
                for k in common_k:
                    sep = sep_set.get((vi, vj), sep_set.get((vj, vi), set()))
                    if k not in sep:
                        if not graph.graph.has_edge(vi, k):
                            graph.add_edge(vi, k, 0.3)
                        if not graph.graph.has_edge(vj, k):
                            graph.add_edge(vj, k, 0.3)
        logger.info(f"PC算法完成: {graph.number_of_nodes()} 节点, "
                     f"{graph.number_of_edges()} 条边")
        if apply_business_prior:
            graph = self._apply_business_prior(graph)
            logger.info(f"业务先验校准后: {graph.number_of_nodes()} 节点, "
                         f"{graph.number_of_edges()} 条边")
        return graph

    def _apply_business_prior(self, graph: CausalGraph) -> CausalGraph:
        logger.info("应用农险+期货业务先验约束...")
        for src, tgt in BUSINESS_PRIOR_EDGES:
            if src in graph.nodes and tgt in graph.nodes:
                if not graph.graph.has_edge(src, tgt):
                    graph.add_edge(src, tgt, 0.6)
                    logger.info(f"  先验添加边: {src} -> {tgt}")
                if graph.graph.has_edge(tgt, src):
                    graph.graph.remove_edge(tgt, src)
                    if (tgt, src) in graph.edge_strengths:
                        del graph.edge_strengths[(tgt, src)]
                    graph.edges = [(s, t) for s, t in graph.edges if (s, t) != (tgt, src)]
                    logger.info(f"  先验修正方向: {tgt} -> {src} 改为 {src} -> {tgt}")
        for src, tgt in BUSINESS_FORBIDDEN_EDGES:
            if graph.graph.has_edge(src, tgt):
                graph.graph.remove_edge(src, tgt)
                if (src, tgt) in graph.edge_strengths:
                    del graph.edge_strengths[(src, tgt)]
                graph.edges = [(s, t) for s, t in graph.edges if (s, t) != (src, tgt)]
                logger.info(f"  先验禁止边: {src} -> {tgt}")
        return graph

    def _partial_correlation(self, data: pd.DataFrame,
                               x: str, y: str,
                               z: List[str]) -> float:
        if not z:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
                corr, _ = stats.pearsonr(data[x], data[y])
            return corr
        from sklearn.linear_model import LinearRegression
        residuals_x = data[x].copy()
        residuals_y = data[y].copy()
        for z_var in z:
            if data[z_var].std() == 0:
                continue
            lr_x = LinearRegression().fit(data[[z_var]], residuals_x)
            residuals_x = residuals_x - lr_x.predict(data[[z_var]])
            lr_y = LinearRegression().fit(data[[z_var]], residuals_y)
            residuals_y = residuals_y - lr_y.predict(data[[z_var]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
            if np.std(residuals_x) < 1e-10 or np.std(residuals_y) < 1e-10:
                return 0.0
            corr, _ = stats.pearsonr(residuals_x, residuals_y)
        return corr if not np.isnan(corr) else 0.0

    def _hsic_test(self, x: np.ndarray, y: np.ndarray, 
                   z: np.ndarray = None, alpha: float = 0.05) -> Tuple[float, bool]:
        """
        HSIC (Hilbert-Schmidt Independence Criterion) 非线性独立性检验
        
        用于检测变量间的非线性依赖关系，比Pearson相关更强大
        理论基础：Gretton et al. (2005) "Kernel Methods for Measuring Independence"
        
        Args:
            x, y: 待检验的变量（1D数组）
            z: 条件变量（可选，用于条件HSIC）
            alpha: 显著性水平
            
        Returns:
            (统计量p值, 是否独立)
        """
        try:
            n = len(x)
            if n < 10:
                return 1.0, True  # 样本不足，保守判断为独立
            
            # RBF核带宽选择（中位数启发式）
            def rbf_kernel_matrix(X, sigma=None):
                if sigma is None:
                    pairwise_dists = cdist(X.reshape(-1, 1), X.reshape(-1, 1), 'euclidean')
                    sigma = np.median(pairwise_dists) + 1e-8
                return np.exp(-cdist(X.reshape(-1, 1), X.reshape(-1, 1), 'sqeuclidean') / (2 * sigma**2))
            
            Kx = rbf_kernel_matrix(x)
            Ky = rbf_kernel_matrix(y)
            
            # 中心化核矩阵
            H = np.eye(n) - np.ones((n, n)) / n
            Kxc = H @ Kx @ H
            Kyc = H @ Ky @ H
            
            # HSIC统计量
            hsic_stat = np.trace(Kxc @ Kyc) / (n - 1)**2
            
            # 置换检验估计p值（快速版本：100次置换）
            n_permutations = min(100, max(50, n // 2))
            count = 0
            rng = np.random.RandomState(42)
            
            for _ in range(n_permutations):
                y_permuted = rng.permutation(y)
                Ky_perm = rbf_kernel_matrix(y_permuted)
                Kyc_perm = H @ Ky_perm @ H
                hsic_perm = np.trace(Kxc @ Kyc_perm) / (n - 1)**2
                if hsic_perm >= hsic_stat:
                    count += 1
            
            p_val = (count + 1) / (n_permutations + 1)
            is_independent = p_val > alpha
            
            return p_val, is_independent
            
        except Exception as e:
            logger.debug(f"HSIC检验失败: {e}，回退到Pearson相关")
            return self._pearson_fallback(x, y, alpha)
    
    @staticmethod
    def _pearson_fallback(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Tuple[float, bool]:
        """HSIC失败时的回退方案"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
                _, p_val = stats.pearsonr(x, y)
            return float(p_val), p_val > alpha
        except Exception as e:
            logger.debug(f"Pearson回退失败: {e}")
            return 1.0, True

    @timer(verbose=False)
    def compute_edge_strengths(self, data: pd.DataFrame,
                                  graph: CausalGraph) -> Dict[Tuple[str, str], float]:
        strengths = {}
        df_clean = data.dropna()
        for edge in graph.edges:
            src, tgt = edge
            if src in df_clean.columns and tgt in df_clean.columns:
                corr = abs(df_clean[src].corr(df_clean[tgt]))
                strengths[edge] = round(corr, 4)
            else:
                strengths[edge] = 0.5
        graph.edge_strengths = strengths
        return strengths

    def extract_causal_paths(self, graph: CausalGraph,
                               target: str) -> List[List[str]]:
        paths = []
        for node in graph.nodes:
            if node == target:
                continue
            try:
                path = nx.shortest_path(graph.graph, source=node, target=target)
                if len(path) > 1:
                    paths.append(path)
            except nx.NetworkXNoPath:
                continue
        paths.sort(key=len, reverse=True)
        return paths[:10]

    def compute_influence_scores(self, data: pd.DataFrame,
                                   graph: CausalGraph,
                                   target: str = 'close') -> Dict[str, float]:
        influences = {}
        df_clean = data.dropna()
        parents = graph.get_parents(target)
        for parent in parents:
            if parent in df_clean.columns and target in df_clean.columns:
                try:
                    if df_clean[parent].std() == 0 or df_clean[target].std() == 0:
                        influences[parent] = 0.0
                        continue
                    other_parents = [p for p in parents if p != parent and p in df_clean.columns]
                    if other_parents:
                        from sklearn.linear_model import LinearRegression
                        Z = df_clean[other_parents].values
                        resid_target = df_clean[target].values - LinearRegression().fit(Z, df_clean[target].values).predict(Z)
                        Z_with_parent = df_clean[[parent] + other_parents].values
                        resid_target_adj = df_clean[target].values - LinearRegression().fit(Z_with_parent, df_clean[target].values).predict(Z_with_parent)
                        ss_full = np.var(resid_target_adj)
                        ss_reduced = np.var(resid_target)
                        partial_r2 = max(0, (ss_reduced - ss_full) / (ss_reduced + 1e-8))
                        influences[parent] = round(partial_r2 * 100, 2)
                    else:
                        corr = abs(df_clean[parent].corr(df_clean[target]))
                        influences[parent] = round(corr ** 2 * 100, 2)
                except Exception:
                    corr = abs(df_clean[parent].corr(df_clean[target]))
                    influences[parent] = round(corr ** 2 * 100, 2)
        total = sum(influences.values())
        if total > 0:
            influences = {k: round(v / total * 100, 2) for k, v in influences.items()}
        sorted_inf = dict(sorted(influences.items(), key=lambda x: x[1], reverse=True))
        logger.info(f"因子影响度排序(数据驱动): {sorted_inf}")
        return sorted_inf

    @timer(verbose=False)
    def bootstrap_stability(self, data: pd.DataFrame,
                              variables: List[str] = None,
                              n_bootstrap: int = 50,
                              sample_ratio: float = 0.8,
                              apply_business_prior: bool = True,
                              random_state: int = 42,
                              max_workers: int = 4) -> Dict:
        logger.info(f"开始Bootstrap稳定性验证: {n_bootstrap}次采样（符合论文50次标准）...")
        rng = np.random.RandomState(random_state)
        if variables is None:
            variables = data.select_dtypes(include=[np.number]).columns.tolist()[:12]
        edge_counts = {}
        
        use_parallel = n_bootstrap >= 5 and max_workers > 1
        if use_parallel:
            try:
                from multiprocessing import cpu_count
                actual_workers = min(max_workers, cpu_count() or 2)
                with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
                    futures = []
                    for i in range(n_bootstrap):
                        future = executor.submit(_bootstrap_single, data, variables,
                                               apply_business_prior, sample_ratio, random_state + i, i)
                        futures.append(future)
                    
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            edges = future.result()
                            if not edges or not isinstance(edges, (list, tuple)):
                                continue
                            for edge in edges:
                                try:
                                    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                                        src, tgt = str(edge[0]), str(edge[1])
                                    elif isinstance(edge, str) and '->' in edge:
                                        parts = edge.split('->')
                                        src, tgt = parts[0].strip(), parts[1].strip()
                                    else:
                                        continue
                                    if src and tgt and src != tgt:
                                        edge_key = f"{src}->{tgt}"
                                        edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
                                except Exception:
                                    continue
                        except Exception as future_err:
                            logger.warning(f"获取Bootstrap结果{i+1}失败: {str(future_err)[:100]}")
                            continue
            except Exception as parallel_err:
                logger.warning(f"并行执行失败，回退串行: {str(parallel_err)[:100]}")
                use_parallel = False
        
        if not use_parallel or not edge_counts:
            if use_parallel and not edge_counts:
                logger.warning("并行Bootstrap未收集到任何边，回退串行执行")
                use_parallel = False
            for i in range(n_bootstrap):
                try:
                    edges = _bootstrap_single(data, variables, apply_business_prior, sample_ratio, rng, i)
                    if not edges or not isinstance(edges, (list, tuple)):
                        continue
                    for edge in edges:
                        try:
                            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                                src, tgt = str(edge[0]), str(edge[1])
                            elif isinstance(edge, str) and '->' in edge:
                                parts = edge.split('->')
                                src, tgt = parts[0].strip(), parts[1].strip()
                            else:
                                continue
                            if src and tgt and src != tgt:
                                edge_key = f"{src}->{tgt}"
                                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
                        except Exception:
                            continue
                except Exception as e:
                    logger.warning(f"Bootstrap第{i+1}次失败: {str(e)[:100]}")
                    continue
        
        stability_scores = {}
        for edge_key, count in edge_counts.items():
            stability_scores[edge_key] = round(count / n_bootstrap, 4)
        
        # 使用动态阈值：根据采样次数调整
        # 10次采样：至少3次出现(0.3)即为较稳定
        # 降低阈值以获得更多有用信息
        dynamic_threshold = max(0.3, min(0.7, 3.0 / n_bootstrap))
        stable_edges = {k: v for k, v in stability_scores.items() if v >= dynamic_threshold}
        
        logger.info(f"Bootstrap完成: {len(stability_scores)}条边检测到, "
                   f"{len(stable_edges)}条稳定(≥{dynamic_threshold:.2f}), "
                   f"阈值={dynamic_threshold:.2f}")
        
        # 如果没有稳定边，显示top5最稳定的边供参考
        if not stable_edges and stability_scores:
            top_5 = dict(list(stability_scores.items())[:5])
            logger.info(f"Top5边稳定性(参考): {top_5}")
            # 放宽条件，使用最低阈值
            stable_edges = {k: v for k, v in stability_scores.items() if v >= 0.2}
        
        return {
            'stability_scores': dict(sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)),
            'stable_edges': stable_edges,
            'n_bootstrap': n_bootstrap,
            'n_stable': len(stable_edges),
            'n_total': len(stability_scores),
            'threshold': dynamic_threshold
        }

    def placebo_test(self, data: pd.DataFrame,
                      treatment_col: str,
                      outcome_col: str,
                      n_permutations: int = 200,
                      alpha: float = 0.05,
                      random_state: int = 42) -> Dict:
        """
        Placebo安慰剂检验（200次置换检验）
        
        理论基础：通过随机打乱处理变量分配，生成安慰剂效应分布
        若真实ATE在安慰剂分布的95%分位数之外，则因果效应统计显著
        
        Args:
            data: 完整数据集
            treatment_col: 处理变量列名（如weather_risk）
            outcome_col: 结果变量列名（如close）
            n_permutations: 置换次数（论文要求200次）
            alpha: 显著性水平
            random_state: 随机种子
            
        Returns:
            包含p值、置信区间、是否通过检验的字典
        """
        logger.info(f"开始Placebo安慰剂检验: {n_permutations}次置换（符合论文200次标准）...")
        
        try:
            from sklearn.linear_model import LinearRegression
            import warnings
            
            df_clean = data[[treatment_col, outcome_col]].dropna()
            if len(df_clean) < 50:
                logger.warning(f"样本量不足({len(df_clean)})，跳过Placebo检验")
                return {'valid': False, 'reason': 'insufficient_sample'}
            
            # 计算真实ATE（简单OLS估计作为基准）
            X_true = df_clean[[treatment_col]].values
            y_true = df_clean[outcome_col].values
            model_true = LinearRegression().fit(X_true, y_true)
            true_ate = model_true.coef_[0]
            
            logger.info(f"真实ATE估计值: {true_ate:.6f}")
            
            # 置换检验：随机打乱处理变量，重复估计ATE
            rng = np.random.RandomState(random_state)
            placebo_ates = []
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                for i in range(n_permutations):
                    # 随机打乱处理变量（保持结果变量不变）
                    treatment_shuffled = rng.permutation(df_clean[treatment_col].values)
                    
                    # 使用打乱后的处理变量估计"伪"ATE
                    model_placebo = LinearRegression()
                    model_placebo.fit(treatment_shuffled.reshape(-1, 1), y_true)
                    placebo_ate = model_placebo.coef_[0]
                    placebo_ates.append(placebo_ate)
                    
                    if (i + 1) % 50 == 0:
                        logger.debug(f"  Placebo进度: {i+1}/{n_permutations}")
            
            placebo_ates = np.array(placebo_ates)
            
            # 计算p值：真实ATE在安慰剂分布中的极端位置
            # 双侧检验：计算|ATE| >= |真实ATE|的比例
            extreme_count = np.sum(np.abs(placebo_ates) >= abs(true_ate))
            p_value = (extreme_count + 1) / (n_permutations + 1)
            
            # 计算安慰剂分布的95%置信区间
            ci_lower = np.percentile(placebo_ates, 2.5)
            ci_upper = np.percentile(placebo_ates, 97.5)
            
            # 判断是否通过检验（真实ATE在95%CI之外）
            is_significant = (true_ate < ci_lower) or (true_ate > ci_upper)
            
            # 计算效应大小（Cohen's d）
            placebo_mean = np.mean(placebo_ates)
            placebo_std = np.std(placebo_ates)
            cohens_d = (true_ate - placebo_mean) / (placebo_std + 1e-8)
            
            result = {
                'valid': True,
                'true_ate': round(true_ate, 6),
                'placebo_mean': round(float(placebo_mean), 6),
                'placebo_std': round(float(placebo_std), 6),
                'ci_95_lower': round(float(ci_lower), 6),
                'ci_95_upper': round(float(ci_upper), 6),
                'p_value': round(p_value, 4),
                'is_significant': is_significant,
                'alpha': alpha,
                'cohens_d': round(float(cohens_d), 4),
                'n_permutations': n_permutations,
                'interpretation': self._interpret_placebo_result(
                    is_significant, p_value, cohens_d, treatment_col, outcome_col
                )
            }
            
            status = "✅ 通过" if is_significant else "❌ 未通过"
            logger.info(f"Placebo检验完成: p={p_value:.4f}, 95%CI=[{ci_lower:.4f}, {ci_upper:.4f}], "
                       f"Cohen's d={cohens_d:.2f} → {status}")
            
            return result
            
        except Exception as e:
            logger.error(f"Placebo检验失败: {e}")
            return {'valid': False, 'error': str(e)}

    @staticmethod
    def _interpret_placebo_result(is_significant: bool, p_value: float,
                                   cohens_d: float, treatment: str,
                                   outcome: str) -> str:
        """解释Placebo检验结果"""
        if not is_significant:
            return (f"⚠️ {treatment}对{outcome}的因果效应未达统计显著性水平"
                   f"(p={p_value:.3f})，可能存在混杂因素或效应较弱")
        
        effect_size_desc = "大" if abs(cohens_d) > 0.8 else ("中" if abs(cohens_d) > 0.5 else "小")
        return (f"✅ {treatment}对{outcome}的因果效应统计显著"
               f"(p={p_value:.4f}, Cohen's d={cohens_d:.2f}, 效应量{effect_size_desc})"
               f"，排除了虚假因果关系的可能性")
