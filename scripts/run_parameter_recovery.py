"""
Parameter Recovery Analysis Script

This script evaluates how well the inference model can recover known parameter values
from simulated data, which is crucial for validating the model's identifiability
and estimation procedures.

Purpose:
    - Generate data from known ("true") parameter values
    - Attempt to recover these parameters using the inference model
    - Assess the bias, variance, and overall accuracy of parameter estimates
    - Identify which parameters are well-recovered and which are problematic
    - Validate the model's ability to make accurate inferences

Usage:
    python scripts/run_parameter_recovery.py [--config CONFIG_PATH]
    
    The config file should be a JSON file with the following structure:
    {
        "inference_model_path": "models/m_0.stan",   # Path to inference model
        "sim_model_path": "models/m_0_sim.stan",     # Path to simulation model
        "study_design_path": "path/to/design.json",  # Path to existing study design
        "output_dir": "results/parameter_recovery/my_run", # Where to save results
        "n_mcmc_samples": 2000,    # Number of posterior samples per chain
        "n_mcmc_chains": 4,        # Number of MCMC chains
        "n_iterations": 20         # Number of recovery iterations to perform
    }
    
    Alternatively, you can specify a study_config_path instead of study_design_path
    to generate a new study design from a configuration file.
    
Examples:
    # Run with default configuration
    python scripts/run_parameter_recovery.py
    
    # Run with a custom configuration file
    python scripts/run_parameter_recovery.py --config configs/my_recovery_config.json
"""
import os
import sys
import json
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.parameter_recovery import ParameterRecovery
from utils.study_design import StudyDesign

def run_from_config(config_path):
    """
    Run parameter recovery analysis from a configuration file.
    
    Parameters:
        config_path (str): Path to the configuration file
        
    Returns:
        ParameterRecovery: The analysis instance after running
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract parameters with defaults
    inference_model_path = config.get('inference_model_path', None)
    sim_model_path = config.get('sim_model_path', None)
    output_dir = config.get('output_dir', None)
    
    # Extract numerical parameters with defaults
    n_mcmc_samples = config.get('n_mcmc_samples', 2000)
    n_mcmc_chains = config.get('n_mcmc_chains', 4)
    n_iterations = config.get('n_iterations', 20)
    
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
    recovery = ParameterRecovery(
        inference_model_path=inference_model_path,
        sim_model_path=sim_model_path,
        study_design=study_design,
        output_dir=output_dir,
        n_mcmc_samples=n_mcmc_samples,
        n_mcmc_chains=n_mcmc_chains,
        n_iterations=n_iterations
    )
    
    # Run the analysis
    true_params, posterior_summaries = recovery.run()
    
    print(f"Parameter recovery analysis completed.")
    print(f"Results saved to: {recovery.output_dir}")
    
    return recovery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameter recovery analysis")
    parser.add_argument('--config', type=str, default='configs/parameter_recovery_config.json', 
                        help='Path to configuration file')
    args = parser.parse_args()
    
    run_from_config(args.config)