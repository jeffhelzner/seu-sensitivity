"""
Prior Predictive Analysis for SEU Sensitivity Models

This module provides tools for performing prior predictive analysis on Bayesian Decision Theory models.
It allows researchers to understand the range of behaviors that a model can produce before fitting
to actual data, which is useful for:

1. Model validation - ensuring the model can produce sensible behaviors
2. Prior sensitivity analysis - understanding how prior choices affect predictions
3. Experimental design - determining if a study design can discriminate between competing models
4. Parameter recoverability assessment - checking if parameters can be recovered from simulated data

The module works by:
1. Loading or generating a study design
2. Sampling multiple parameter configurations from the prior distribution
3. For each parameter configuration, simulating multiple choice outcomes
4. Analyzing the results through visualizations and summary statistics

Examples:
    # Basic usage with defaults
    analysis = PriorPredictiveAnalysis()
    analysis.run()
    
    # Custom usage with specific parameters
    from utils.study_design import StudyDesign
    study = StudyDesign.load("path/to/design.json")
    analysis = PriorPredictiveAnalysis(
        model_path="models/custom_model.stan",
        study_design=study,
        output_dir="results/my_analysis",
        n_param_samples=500,
        n_choice_samples=10
    )
    samples = analysis.run()
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import datetime

# Add parent directory to path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.study_design import StudyDesign
from utils import (
    detect_model_name, get_model_sim_hyperparams, DEFAULT_PARAM_GENERATION
)

class PriorPredictiveAnalysis:
    """
    A class for performing prior predictive analysis on Bayesian Decision Theory models.
    
    This class handles:
    - Sampling from prior distributions of model parameters
    - Simulating choices based on those parameters
    - Analyzing and visualizing results
    - Saving outputs for later reference
    
    Attributes:
        model_path (str): Path to the Stan model file
        study_design (StudyDesign): Study design object containing experiment structure
        output_dir (str): Directory where results will be saved
        n_param_samples (int): Number of parameter configurations to sample
        n_choice_samples (int): Number of choice simulations per parameter set
        model (CmdStanModel): Compiled Stan model
        samples (pd.DataFrame): Results from prior predictive sampling
    """
    def __init__(
        self,
        model_path=None,
        study_design=None,
        output_dir=None,
        n_param_samples=100,   # Number of parameter configurations to sample
        n_choice_samples=5     # Number of choice simulations per parameter set
    ):
        """
        Initialize the prior predictive analysis.
        
        Parameters:
            model_path (str, optional): Path to the Stan model file. If None, uses default m_0_sim.stan.
            study_design (StudyDesign, optional): Study design object. If None, a new one will be generated.
            output_dir (str, optional): Directory to save results. If None, creates a timestamped directory.
            n_param_samples (int): Number of parameter configurations to sample from the prior.
            n_choice_samples (int): Number of choice simulations to run per parameter configuration.
        """
        # Set default model path if not provided
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "models", 
                "m_0_sim.stan"
            )
            
        self.model_path = model_path
        self.study_design = study_design
        self.n_param_samples = n_param_samples
        self.n_choice_samples = n_choice_samples
        
        # Set default output directory if not provided
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "prior_predictive",
                f"run_{timestamp}"
            )
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
            
        # Compile the model
        self.model = CmdStanModel(stan_file=self.model_path)
        
    def run(self):
        """
        Run the prior predictive analysis with multiple parameter samples.
        
        This method:
        1. Ensures a study design exists or creates one
        2. Generates multiple parameter samples from the prior
        3. For each parameter set, simulates multiple choice outcomes
        4. Analyzes and visualizes the results
        
        Returns:
            pd.DataFrame: The combined samples from all simulations
        """
        # Generate study design if not provided
        if self.study_design is None:
            self.study_design = StudyDesign()
            self.study_design.generate()
            
        # Save the study design
        design_path = os.path.join(self.output_dir, "study_design.json")
        self.study_design.save(design_path)
        
        # Get data dictionary for Stan
        data = self.study_design.get_data_dict()
        
        # Detect model name and add appropriate prior hyperparameters
        model_name = detect_model_name(self.model_path)
        required_hyperparams = get_model_sim_hyperparams(model_name)
        for hp in required_hyperparams:
            if hp not in data:
                data[hp] = DEFAULT_PARAM_GENERATION.get(hp, 1.0)
        
        # Run multiple independent Stan simulations (different parameter values each time)
        all_samples = []
        
        print(f"Generating {self.n_param_samples} parameter samples from prior...")
        for i in range(self.n_param_samples):
            if i % 10 == 0:
                print(f"  Sample {i}/{self.n_param_samples}")
                
            # Each run gets a different seed for parameter generation
            seed = 12345 + i
            
            # Sample from the model with fixed parameters
            fit = self.model.sample(
                data=data,
                seed=seed,
                iter_sampling=self.n_choice_samples,  # Multiple choice simulations per parameter set
                iter_warmup=0,            
                chains=1,            
                fixed_param=True,
                adapt_engaged=False
            )
            
            # Extract samples and add a parameter set ID
            samples = fit.draws_pd()
            samples['param_set'] = i
            
            all_samples.append(samples)
        
        # Combine all samples
        self.samples = pd.concat(all_samples, ignore_index=True)
        
        # Save the fit summary
        self.samples.to_csv(os.path.join(self.output_dir, "prior_samples.csv"))
        
        # Run analyses and create plots
        self._analyze_parameters()
        self._analyze_expected_utilities()
        self._analyze_choice_probabilities()
        self._analyze_choices()
        self._analyze_seu_maximizer_selection()
        
        # Return samples for further analysis if needed
        return self.samples
        
    def _analyze_parameters(self):
        """
        Analyze the generated parameter values from prior samples.
        
        This method:
        1. Creates visualizations of parameter distributions (alpha, beta, delta, utilities)
        2. Computes summary statistics for all parameters
        3. Saves plots and statistics to the output directory
        
        The analysis helps understand the range of parameter values covered by the prior.
        """
        # Create parameter plots directory
        param_dir = os.path.join(self.output_dir, "parameters")
        os.makedirs(param_dir, exist_ok=True)
        
        # Extract parameters
        alpha = self.samples['alpha']
        
        # Plot alpha distribution
        plt.figure(figsize=(10, 6))
        plt.hist(alpha, bins=30, alpha=0.7)
        plt.axvline(np.median(alpha), color='red', linestyle='--', 
                   label=f'Median: {np.median(alpha):.2f}')
        plt.title('Prior Distribution of Alpha (Sensitivity)')
        plt.xlabel('Alpha')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(param_dir, 'alpha_dist.png'))
        plt.close()
        
        # Plot omega distribution if present (m_2 or m_3)
        if 'omega' in self.samples.columns:
            omega = self.samples['omega']
            plt.figure(figsize=(10, 6))
            plt.hist(omega, bins=30, alpha=0.7)
            plt.axvline(np.median(omega), color='red', linestyle='--', 
                       label=f'Median: {np.median(omega):.2f}')
            plt.title('Prior Distribution of Omega (Risky Sensitivity)')
            plt.xlabel('Omega')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(os.path.join(param_dir, 'omega_dist.png'))
            plt.close()
        
        # Plot kappa distribution if present (m_3)
        if 'kappa' in self.samples.columns:
            kappa = self.samples['kappa']
            plt.figure(figsize=(10, 6))
            plt.hist(kappa, bins=30, alpha=0.7)
            plt.axvline(np.median(kappa), color='red', linestyle='--', 
                       label=f'Median: {np.median(kappa):.2f}')
            plt.title('Prior Distribution of Kappa (Risk-Uncertainty Association)')
            plt.xlabel('Kappa')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(os.path.join(param_dir, 'kappa_dist.png'))
            plt.close()
            
            # For m_3, also plot alpha vs omega scatter
            if 'omega' in self.samples.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(self.samples['alpha'], self.samples['omega'], alpha=0.3, s=10)
                plt.xlabel('Alpha')
                plt.ylabel('Omega (= Kappa * Alpha)')
                plt.title('Prior: Alpha vs Omega Relationship')
                lims = [0, max(self.samples['alpha'].max(), self.samples['omega'].max())]
                plt.plot(lims, lims, 'r--', alpha=0.5, label='omega = alpha (kappa=1)')
                plt.legend()
                plt.savefig(os.path.join(param_dir, 'alpha_vs_omega.png'))
                plt.close()
        
        # Extract and plot beta parameters
        K, D = self.study_design.K, self.study_design.D
        
        # Create a figure for all beta parameters
        plt.figure(figsize=(15, 10))
        idx = 1
        for k in range(1, K+1):
            for d in range(1, D+1):
                plt.subplot(K, D, idx)
                beta_col = f'beta[{k},{d}]'
                if beta_col in self.samples.columns:
                    plt.hist(self.samples[beta_col], bins=20)
                    plt.title(f'Beta[{k},{d}]')
                idx += 1
        plt.tight_layout()
        plt.savefig(os.path.join(param_dir, 'beta_dist.png'))
        plt.close()
        
        # Extract and plot delta parameters
        delta_cols = [col for col in self.samples.columns if col.startswith('delta[')]
        if delta_cols:
            plt.figure(figsize=(12, 8))
            for i, col in enumerate(delta_cols):
                plt.subplot(2, (len(delta_cols)+1)//2, i+1)
                plt.hist(self.samples[col], bins=20)
                plt.title(col)
            plt.tight_layout()
            plt.savefig(os.path.join(param_dir, 'delta_dist.png'))
            plt.close()
            
        # Plot utilities
        upsilon_cols = [col for col in self.samples.columns if col.startswith('upsilon[')]
        if upsilon_cols:
            plt.figure(figsize=(12, 8))
            for i, col in enumerate(upsilon_cols):
                plt.subplot(2, (len(upsilon_cols)+1)//2, i+1)
                plt.hist(self.samples[col], bins=20)
                plt.title(col)
            plt.tight_layout()
            plt.savefig(os.path.join(param_dir, 'utility_dist.png'))
            plt.close()
            
        # Create summary statistics
        param_summary = {
            'alpha': {
                'mean': float(np.mean(alpha)),
                'std': float(np.std(alpha)),
                'median': float(np.median(alpha)),
                'q05': float(np.quantile(alpha, 0.05)),
                'q95': float(np.quantile(alpha, 0.95))
            }
        }
        
        # Add beta, delta, upsilon, omega, kappa summaries
        for col in self.samples.columns:
            if (col.startswith('beta[') or col.startswith('delta[') or 
                col.startswith('upsilon[') or col in ('omega', 'kappa')):
                param_summary[col] = {
                    'mean': float(np.mean(self.samples[col])),
                    'std': float(np.std(self.samples[col])),
                    'median': float(np.median(self.samples[col])),
                    'q05': float(np.quantile(self.samples[col], 0.05)),
                    'q95': float(np.quantile(self.samples[col], 0.95))
                }
                
        # Save parameter summary
        with open(os.path.join(param_dir, 'parameter_summary.json'), 'w') as f:
            json.dump(param_summary, f, indent=2)
    
    def _analyze_expected_utilities(self):
        """
        Analyze the expected utilities generated for each alternative.
        
        This method:
        1. Creates boxplots showing the distribution of expected utilities for each alternative
           across all parameter samples
        2. Helps understand how much the prior influences expected utilities
        3. Shows whether certain alternatives consistently have higher expected utilities
        
        Results are saved to the output directory under 'expected_utilities/'.
        """
        # Create expected utilities plots directory
        eu_dir = os.path.join(self.output_dir, "expected_utilities")
        os.makedirs(eu_dir, exist_ok=True)
        
        # Extract problem_etas columns for all decision problems and alternatives
        eta_cols = [col for col in self.samples.columns if 'problem_etas' in col]
        
        # Get a subset of problems to visualize (max 10)
        problems = min(10, self.study_design.M)
        
        plt.figure(figsize=(15, 10))
        for m in range(1, problems+1):
            plt.subplot(3, (problems+2)//3, m)
            
            # Get columns for this problem
            problem_cols = [col for col in eta_cols if f'problem_etas[{m},' in col]
            
            # Only keep alternatives that exist (expected utility != -1e10)
            valid_cols = []
            for col in problem_cols:
                if np.median(self.samples[col]) > -1e9:  # Not the padding value
                    valid_cols.append(col)
            
            # Create box plots for valid alternatives
            if valid_cols:
                data = [self.samples[col] for col in valid_cols]
                plt.boxplot(data)
                plt.title(f'Problem {m}')
                plt.xlabel('Alternative')
                plt.ylabel('Expected Utility')
                plt.xticks(range(1, len(valid_cols)+1), 
                          [c.split(',')[1].strip(']') for c in valid_cols])
            
        plt.tight_layout()
        plt.savefig(os.path.join(eu_dir, 'expected_utilities.png'))
        plt.close()
    
    def _analyze_choice_probabilities(self):
        """
        Analyze the choice probabilities for each alternative.
        
        This method:
        1. Creates boxplots showing the distribution of choice probabilities for each alternative
           across all parameter samples
        2. Helps understand how parameter uncertainty affects choice probabilities
        3. Identifies alternatives that are consistently chosen with high or low probability
        
        Results are saved to the output directory under 'choice_probabilities/'.
        """
        # Create choice probabilities plots directory
        cp_dir = os.path.join(self.output_dir, "choice_probabilities")
        os.makedirs(cp_dir, exist_ok=True)
        
        # Extract choice_probabilities columns
        cp_cols = [col for col in self.samples.columns if 'choice_probabilities' in col]
        
        # Get a subset of problems to visualize (max 10)
        problems = min(10, self.study_design.M)
        
        plt.figure(figsize=(15, 10))
        for m in range(1, problems+1):
            plt.subplot(3, (problems+2)//3, m)
            
            # Get columns for this problem
            problem_cols = [col for col in cp_cols if f'choice_probabilities[{m},' in col]
            
            # Only keep alternatives that exist (probability > 0)
            valid_cols = []
            for col in problem_cols:
                if np.median(self.samples[col]) > 0.001:  # Not the padding value
                    valid_cols.append(col)
            
            # Create box plots for valid alternatives
            if valid_cols:
                data = [self.samples[col] for col in valid_cols]
                plt.boxplot(data)
                plt.title(f'Problem {m}')
                plt.xlabel('Alternative')
                plt.ylabel('Choice Probability')
                plt.xticks(range(1, len(valid_cols)+1), 
                          [c.split(',')[1].strip(']') for c in valid_cols])
            
        plt.tight_layout()
        plt.savefig(os.path.join(cp_dir, 'choice_probabilities.png'))
        plt.close()
        
    def _analyze_choices(self):
        """
        Analyze the actual choices simulated from the model.
        
        This method:
        1. Counts the frequency of each alternative being chosen across all simulations
        2. Creates bar plots showing the empirical choice distributions
        3. Saves choice counts for further analysis
        
        Results are saved to the output directory under 'choices/'.
        """
        # Create choices plots directory
        choices_dir = os.path.join(self.output_dir, "choices")
        os.makedirs(choices_dir, exist_ok=True)
        
        # Extract choice columns
        y_cols = [col for col in self.samples.columns if col.startswith('y[')]
        
        # Calculate the total number of choice simulations
        total_samples = self.n_param_samples * self.n_choice_samples
        
        # Count choices for each decision problem
        choice_counts = {}
        for m in range(1, self.study_design.M + 1):
            if f'y[{m}]' in self.samples.columns:
                choices = self.samples[f'y[{m}]']
                choice_counts[m] = choices.value_counts().to_dict()
        
        # Get problems with valid choice data (up to 10)
        valid_problems = []
        for m in range(1, self.study_design.M + 1):
            if m in choice_counts and choice_counts[m]:
                valid_problems.append(m)
                if len(valid_problems) >= 10:
                    break
    
        if not valid_problems:
            print("No valid choice data found.")
            return
    
        # Calculate optimal grid layout based on number of plots
        num_plots = len(valid_problems)
        num_cols = min(3, num_plots)
        num_rows = (num_plots + num_cols - 1) // num_cols
    
        # Create figure with appropriate size
        plt.figure(figsize=(5*num_cols, 4*num_rows))
    
        # Use a pleasing color palette
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
        for i, m in enumerate(valid_problems):
            ax = plt.subplot(num_rows, num_cols, i+1)
            
            # Get all alternatives that could be chosen
            max_choice = int(max(choice_counts[m].keys()))
            alternatives = list(range(1, max_choice+1))
            
            # Calculate probabilities
            probs = []
            for alt in alternatives:
                probs.append(choice_counts[m].get(alt, 0) / total_samples)
            
            # Create bar chart with improved formatting
            bars = ax.bar(
                alternatives, 
                probs,
                width=0.7,
                color=colors,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            
            # Add data labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:  # Only label bars with non-negligible probability
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.01,
                        f'{height:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        rotation=0
                    )
            
            # Improve formatting
            ax.set_title(f'Problem {m} Choice Distribution', fontsize=12)
            ax.set_xlabel('Alternative', fontsize=10)
            ax.set_ylabel('Choice Probability', fontsize=10)
            ax.set_ylim(0, min(1, max(probs)*1.2))  # Dynamic y-limit with 20% headroom
            ax.set_xticks(alternatives)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add text showing sample size
            ax.text(
                0.02, 0.95, 
                f'n={total_samples}', 
                transform=ax.transAxes,
                fontsize=9,
                va='top'
            )
        
        plt.tight_layout(pad=2.0)
        plt.savefig(os.path.join(choices_dir, 'choice_distribution.png'), dpi=150)
        plt.close()
        
        # Save choice counts
        with open(os.path.join(choices_dir, 'choice_counts.json'), 'w') as f:
            json.dump(choice_counts, f, indent=2)

    def _analyze_seu_maximizer_selection(self):
        """
        Analyze the selection of SEU maximizers across decision problems.
        
        This method:
        1. Calculates the probability of selecting an SEU maximizer for each problem
        2. Analyzes the distribution of total SEU maximizers selected across problems
        3. Creates visualizations showing SEU maximizer selection patterns
        
        Results are saved to the output directory under 'seu_maximizer_selection/'.
        """
        # Create SEU maximizer selection plots directory
        seu_dir = os.path.join(self.output_dir, "seu_maximizer_selection")
        os.makedirs(seu_dir, exist_ok=True)
        
        # Extract SEU maximizer selection indicators
        M = self.study_design.M
        
        # Calculate probability of selecting SEU maximizer for each problem
        prob_seu_max_by_problem = {}
        for m in range(1, M + 1):
            col = f'selected_seu_max[{m}]'
            if col in self.samples.columns:
                prob_seu_max_by_problem[m] = self.samples[col].mean()
        
        # Plot probability of selecting SEU maximizer by problem
        if prob_seu_max_by_problem:
            plt.figure(figsize=(12, 6))
            problems = list(prob_seu_max_by_problem.keys())
            probs = list(prob_seu_max_by_problem.values())
            
            plt.bar(problems, probs, alpha=0.7, edgecolor='black')
            plt.axhline(y=np.mean(probs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(probs):.3f}')
            plt.xlabel('Decision Problem')
            plt.ylabel('Probability of Selecting SEU Maximizer')
            plt.title('Probability of Selecting SEU Maximizer by Problem')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(seu_dir, 'prob_seu_max_by_problem.png'), dpi=150)
            plt.close()
        
        # Analyze distribution of total SEU maximizers selected
        if 'total_seu_max_selected' in self.samples.columns:
            total_seu_max = self.samples['total_seu_max_selected']
            
            # Create histogram
            plt.figure(figsize=(10, 6))
            plt.hist(total_seu_max, bins=range(0, M+2), align='left', 
                    rwidth=0.8, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(total_seu_max), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(total_seu_max):.2f}')
            plt.axvline(np.median(total_seu_max), color='blue', linestyle='--', 
                       label=f'Median: {np.median(total_seu_max):.0f}')
            plt.xlabel('Number of Problems Where SEU Maximizer Selected')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Total SEU Maximizers Selected (out of {M} problems)')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(seu_dir, 'total_seu_max_distribution.png'), dpi=150)
            plt.close()
            
            # Create summary statistics
            seu_max_summary = {
                'total_problems': M,
                'prob_seu_max_by_problem': prob_seu_max_by_problem,
                'overall_prob_seu_max': float(np.mean(list(prob_seu_max_by_problem.values()))),
                'total_seu_max_selected': {
                    'mean': float(np.mean(total_seu_max)),
                    'std': float(np.std(total_seu_max)),
                    'median': float(np.median(total_seu_max)),
                    'min': int(np.min(total_seu_max)),
                    'max': int(np.max(total_seu_max)),
                    'q25': float(np.quantile(total_seu_max, 0.25)),
                    'q75': float(np.quantile(total_seu_max, 0.75))
                }
            }
            
            # Save summary
            with open(os.path.join(seu_dir, 'seu_maximizer_summary.json'), 'w') as f:
                json.dump(seu_max_summary, f, indent=2)
            
            # Print summary to console
            print("\n" + "="*60)
            print("SEU MAXIMIZER SELECTION SUMMARY")
            print("="*60)
            print(f"Total problems: {M}")
            print(f"Overall probability of selecting SEU maximizer: {seu_max_summary['overall_prob_seu_max']:.3f}")
            print(f"\nDistribution of total SEU maximizers selected:")
            print(f"  Mean: {seu_max_summary['total_seu_max_selected']['mean']:.2f}")
            print(f"  Median: {seu_max_summary['total_seu_max_selected']['median']:.0f}")
            print(f"  Range: [{seu_max_summary['total_seu_max_selected']['min']}, "
                  f"{seu_max_summary['total_seu_max_selected']['max']}]")
            print(f"  IQR: [{seu_max_summary['total_seu_max_selected']['q25']:.1f}, "
                  f"{seu_max_summary['total_seu_max_selected']['q75']:.1f}]")
            print("="*60 + "\n")

if __name__ == "__main__":
    # Example usage
    analysis = PriorPredictiveAnalysis(n_param_samples=100, n_choice_samples=5)
    analysis.run()