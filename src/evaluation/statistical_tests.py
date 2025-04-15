import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

class StatisticalAnalyzer:
    """Class for performing statistical analysis on model results."""
    
    @staticmethod
    def paired_ttest(scores1: np.ndarray,
                     scores2: np.ndarray,
                     alpha: float = 0.05) -> Dict[str, float]:
        """Perform paired t-test between two sets of scores."""
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        # Calculate effect size (Cohen's d)
        d = np.mean(scores1 - scores2) / np.std(scores1 - scores2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': d
        }
    
    @staticmethod
    def wilcoxon_test(scores1: np.ndarray,
                      scores2: np.ndarray,
                      alpha: float = 0.05) -> Dict[str, float]:
        """Perform Wilcoxon signed-rank test."""
        stat, p_value = stats.wilcoxon(scores1, scores2)
        
        # Calculate effect size (r = Z / sqrt(N))
        z_score = stats.zscore(scores1 - scores2)
        r = np.mean(z_score) / np.sqrt(len(scores1))
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': r
        }
    
    @staticmethod
    def mcnemar_test(predictions1: np.ndarray,
                     predictions2: np.ndarray,
                     labels: np.ndarray) -> Tuple[float, float]:
        """
        Perform McNemar's test to compare two models' predictions.
        
        Args:
            predictions1: Predictions from first model
            predictions2: Predictions from second model
            labels: True labels
            
        Returns:
            tuple: (statistic, p-value)
        """
        # Count disagreements
        b = np.sum((predictions1 == labels) & (predictions2 != labels))
        c = np.sum((predictions1 != labels) & (predictions2 == labels))
        
        # If both models make identical predictions or if there's only one class,
        # return (0, 1.0) indicating no significant difference
        if b + c == 0 or len(np.unique(labels)) == 1:
            return 0, 1.0
        
        try:
            # Create contingency table
            table = np.array([[0, b], [c, 0]])
            # Use chi2_contingency instead of mcnemar
            from scipy.stats import chi2_contingency
            stat, p_value, _, _ = chi2_contingency(table, correction=True)
            return stat, p_value
        except Exception as e:
            print(f"Warning: McNemar test failed with error: {str(e)}")
            return 0, 1.0
    
    @staticmethod
    def bootstrap_ci(scores: np.ndarray,
                    n_bootstrap: int = 1000,
                    ci: float = 0.95) -> Dict[str, float]:
        """Calculate bootstrap confidence intervals."""
        bootstrap_samples = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_samples.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_samples, (1 - ci) * 100 / 2)
        ci_upper = np.percentile(bootstrap_samples, (1 + ci) * 100 / 2)
        
        return {
            'mean': np.mean(scores),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower
        }
    
    @staticmethod
    def compare_multiple_models(predictions: List[np.ndarray],
                              y_true: np.ndarray,
                              model_names: Optional[List[str]] = None,
                              alpha: float = 0.05) -> pd.DataFrame:
        """Compare multiple models using pairwise statistical tests."""
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(predictions))]
        
        n_models = len(predictions)
        results = []
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                # Perform McNemar's test
                mcnemar_results = StatisticalAnalyzer.mcnemar_test(
                    predictions[i], predictions[j], y_true)
                
                # Calculate AUC difference
                auc_i = roc_auc_score(y_true, predictions[i])
                auc_j = roc_auc_score(y_true, predictions[j])
                
                results.append({
                    'Model1': model_names[i],
                    'Model2': model_names[j],
                    'AUC_Diff': auc_i - auc_j,
                    'P_Value': mcnemar_results[1],
                    'Significant': mcnemar_results[1] < alpha,
                    'Odds_Ratio': mcnemar_results[0]
                })
        
        # Create DataFrame and adjust for multiple comparisons
        df = pd.DataFrame(results)
        df['Adjusted_P_Value'] = multipletests(df['P_Value'], alpha=alpha, method='fdr_bh')[1]
        df['Significant_Adjusted'] = df['Adjusted_P_Value'] < alpha
        
        return df
    
    @staticmethod
    def power_analysis(effect_size: float,
                      alpha: float = 0.05,
                      power: float = 0.8) -> int:
        """Calculate required sample size for desired statistical power."""
        from statsmodels.stats.power import TTestPower
        
        analysis = TTestPower()
        n = analysis.solve_power(effect_size=effect_size,
                               alpha=alpha,
                               power=power,
                               ratio=1.0)
        return int(np.ceil(n))
    
    @staticmethod
    def effect_size_analysis(scores1: np.ndarray,
                            scores2: np.ndarray) -> Dict[str, float]:
        """Calculate various effect size metrics."""
        # Cohen's d
        pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
        cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
        
        # Hedge's g (bias-corrected d)
        n = len(scores1) + len(scores2)
        hedges_g = cohens_d * (1 - 3 / (4 * n - 9))
        
        # Glass's delta
        glass_delta = (np.mean(scores1) - np.mean(scores2)) / np.std(scores2)
        
        return {
            'cohens_d': cohens_d,
            'hedges_g': hedges_g,
            'glass_delta': glass_delta
        }
