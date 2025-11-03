"""
Prior Predictive Analysis Script

This script runs prior predictive analysis on the Bayesian Decision Theory models, 
which helps assess whether the prior distributions are reasonable before fitting 
models to actual data.

Purpose:
    - Sample parameters from prior distributions
    - Generate simulated choices and utilities from those parameters
    - Visualize the resulting distributions
    - Assess whether prior assumptions lead to reasonable behavior
    - Detect potential issues with priors before model fitting

Usage:
    python scripts/run_prior_predictive.py [--config CONFIG_PATH]
    
    The config file should be a JSON file with the following structure:
    {
        "model_path": "models/m_0_sim.stan",      # Path to simulation Stan model
        "study_design_path": "path/to/design.json", # Path to existing study design
        "output_dir": "results/prior_predictive/my_run", # Where to save results
        "n_param_samples": 100,   # Number of parameter samples to draw
        "n_choice_samples": 5     # Number of choice samples per parameter
    }
    
    Alternatively, you can specify a study_config_path instead of study_design_path
    to generate a new study design from a configuration file.
    
Examples:
    # Run with default configuration
    python scripts/run_prior_predictive.py
    
    # Run with a custom configuration file
    python scripts/run_prior_predictive.py --config configs/my_prior_config.json
"""
import os
import sys
import json
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.prior_predictive import PriorPredictiveAnalysis
from utils.study_design import StudyDesign

def run_from_config(config_path):
    """
    Run prior predictive analysis from a configuration file.
    
    Parameters:
        config_path (str): Path to the configuration file
        
    Returns:
        PriorPredictiveAnalysis: The analysis instance after running
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract parameters with defaults
    model_path = config.get('model_path', None)  # Use default if not specified
    output_dir = config.get('output_dir', None)
    
    # Support both new and old parameter naming
    n_param_samples = config.get('n_param_samples', config.get('n_samples', 100))
    n_choice_samples = config.get('n_choice_samples', 5)
    
    # Handle study design
    study_design = None
    study_design_path = config.get('study_design_path', None)
    if study_design_path:
        # Load existing study design
        study_design = StudyDesign.load(study_design_path)
        print(f"Loaded study design from: {study_design_path}")
    elif 'study_config_path' in config:
        # Create study design from config
        study_design = StudyDesign.from_config(config['study_config_path'])
        print(f"Created study design from config: {config['study_config_path']}")
    
    # Initialize and run analysis
    analysis = PriorPredictiveAnalysis(
        model_path=model_path,
        study_design=study_design,
        output_dir=output_dir,
        n_param_samples=n_param_samples,
        n_choice_samples=n_choice_samples
    )
    
    # Run the analysis
    samples = analysis.run()
    
    print(f"Prior predictive analysis completed.")
    print(f"Results saved to: {analysis.output_dir}")
    
    return analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prior predictive analysis")
    parser.add_argument('--config', type=str, default='configs/prior_analysis_config.json', 
                        help='Path to configuration file')
    args = parser.parse_args()
    
    run_from_config(args.config)