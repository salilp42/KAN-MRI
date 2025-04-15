import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from statsmodels.stats.power import TTestPower, GofChisquarePower
from statsmodels.stats.proportion import proportion_effectsize
import matplotlib.pyplot as plt
import os

class PowerAnalyzer:
    """Class for performing power analysis and sample size calculations."""
    
    def __init__(self, save_dir: str = 'results/power_analysis', alpha=0.05, power=0.8):
        """Initialize power analyzer."""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.alpha = alpha
        self.power = power
        self.analyzer = TTestPower()
    
    def sample_size_curve(self, effect_size, n_min=10, n_max=1000, n_points=50):
        """
        Generate a curve showing the relationship between sample size and power
        for a given effect size.
        
        Parameters:
        -----------
        effect_size : float
            The effect size to analyze
        n_min : int
            Minimum sample size to consider
        n_max : int
            Maximum sample size to consider
        n_points : int
            Number of points to plot
            
        Returns:
        --------
        dict containing:
            'n_samples' : array of sample sizes
            'power' : array of corresponding power values
            'required_n' : minimum sample size required for desired power
        """
        # Generate range of sample sizes
        n_samples = np.linspace(n_min, n_max, n_points)
        power_values = []
        
        # Calculate power for each sample size
        for n in n_samples:
            power = self.analyzer.power(
                effect_size=effect_size,
                nobs=n,
                alpha=self.alpha,
                alternative='two-sided'
            )
            power_values.append(power)
            
        # Find required sample size for desired power
        required_n = self.analyzer.solve_power(
            effect_size=effect_size,
            power=self.power,
            alpha=self.alpha,
            alternative='two-sided'
        )
        
        return {
            'n_samples': n_samples,
            'power': np.array(power_values),
            'required_n': required_n
        }

    def plot_power_curve(self, curve_data, title=None):
        """
        Plot the power curve.
        
        Parameters:
        -----------
        curve_data : dict
            Output from sample_size_curve()
        title : str, optional
            Title for the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(curve_data['n_samples'], curve_data['power'])
        plt.axhline(y=self.power, color='r', linestyle='--', label=f'Target Power ({self.power})')
        plt.axvline(x=curve_data['required_n'], color='g', linestyle='--', 
                   label=f'Required N ({int(curve_data["required_n"])})')
        
        plt.xlabel('Sample Size')
        plt.ylabel('Power')
        if title:
            plt.title(title)
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    
    def power_curve(self,
                   sample_sizes: np.ndarray,
                   effect_size: float,
                   alpha: float = 0.05) -> pd.DataFrame:
        """Calculate statistical power for different sample sizes."""
        analysis = TTestPower()
        powers = []
        
        for n in sample_sizes:
            power = analysis.power(
                effect_size=effect_size,
                nobs=n,
                alpha=alpha,
                ratio=1.0
            )
            powers.append(power)
        
        results = pd.DataFrame({
            'sample_size': sample_sizes,
            'power': powers
        })
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(sample_sizes, powers)
        plt.xlabel('Sample Size')
        plt.ylabel('Statistical Power')
        plt.title('Power vs Sample Size')
        plt.grid(True)
        plt.savefig(f'{self.save_dir}/power_curve.png')
        plt.close()
        
        return results
    
    @staticmethod
    def effect_size_from_proportions(p1: float, p2: float) -> float:
        """Calculate effect size from two proportions."""
        return proportion_effectsize(p1, p2)
    
    @staticmethod
    def effect_size_from_means(mean1: float,
                             mean2: float,
                             std1: float,
                             std2: float) -> float:
        """Calculate Cohen's d effect size from means and standard deviations."""
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        return (mean1 - mean2) / pooled_std
    
    def sensitivity_analysis(self,
                           sample_size: int,
                           alpha_range: np.ndarray,
                           effect_size: float) -> pd.DataFrame:
        """Perform sensitivity analysis for different alpha levels."""
        analysis = TTestPower()
        powers = []
        
        for alpha in alpha_range:
            power = analysis.power(
                effect_size=effect_size,
                nobs=sample_size,
                alpha=alpha,
                ratio=1.0
            )
            powers.append(power)
        
        results = pd.DataFrame({
            'alpha': alpha_range,
            'power': powers
        })
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_range, powers)
        plt.xlabel('Alpha Level')
        plt.ylabel('Statistical Power')
        plt.title('Power vs Alpha Level')
        plt.grid(True)
        plt.savefig(f'{self.save_dir}/sensitivity_analysis.png')
        plt.close()
        
        return results
    
    def compute_achieved_power(self,
                             observed_effect_size: float,
                             sample_size: int,
                             alpha: float = 0.05) -> Dict[str, float]:
        """Compute achieved power based on observed effect size."""
        analysis = TTestPower()
        power = analysis.power(
            effect_size=observed_effect_size,
            nobs=sample_size,
            alpha=alpha,
            ratio=1.0
        )
        
        return {
            'effect_size': observed_effect_size,
            'sample_size': sample_size,
            'alpha': alpha,
            'achieved_power': power
        }
    
    def power_analysis_report(self,
                            effect_size: float,
                            sample_size: int,
                            alpha: float = 0.05,
                            target_power: float = 0.8) -> Dict[str, Any]:
        """Generate comprehensive power analysis report."""
        analysis = TTestPower()
        
        # Calculate achieved power
        achieved_power = analysis.power(
            effect_size=effect_size,
            nobs=sample_size,
            alpha=alpha,
            ratio=1.0
        )
        
        # Calculate required sample size for target power
        required_n = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=target_power,
            ratio=1.0
        )
        
        # Generate report
        report = {
            'input_parameters': {
                'effect_size': effect_size,
                'sample_size': sample_size,
                'alpha': alpha,
                'target_power': target_power
            },
            'results': {
                'achieved_power': achieved_power,
                'required_sample_size': int(np.ceil(required_n)),
                'is_sufficient': achieved_power >= target_power,
                'power_deficit': target_power - achieved_power if achieved_power < target_power else 0
            },
            'recommendations': []
        }
        
        # Add recommendations
        if achieved_power < target_power:
            report['recommendations'].append(
                f"Increase sample size to {int(np.ceil(required_n))} to achieve target power"
            )
        if effect_size < 0.2:
            report['recommendations'].append(
                "Consider methods to increase effect size or plan for larger sample size"
            )
        if alpha > 0.01 and achieved_power > 0.9:
            report['recommendations'].append(
                "Consider using a more stringent alpha level"
            )
        
        return report
