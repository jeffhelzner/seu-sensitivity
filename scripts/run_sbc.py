"""
Simulation-Based Calibration (SBC) Analysis Script

This script performs simulation-based calibration to validate that the model's
posterior sampling procedure is correctly implemented and that the model is
well-calibrated.

Purpose:
    - Test whether parameters drawn from the prior can be recovered by the model
    - Validate that the posterior distribution is correctly sampled
    - Check for issues in the model implementation or computation
    - Ensure that the model's credible intervals have proper coverage
    - Provide stronger validation than simple parameter recovery

Usage:
    python scripts/run_sbc.py [--config CONFIG_PATH]
    
    The config file should be a JSON file with the following structure:
    {
        "sbc_model_path": "models/m_0_sbc.stan", # Path to SBC Stan model
        "study_design_path": "path/to/design.json", # Path to existing study design
        "output_dir": "results/sbc/my_run",     # Where to save results
        "n_sbc_sims": 100,        # Number of SBC simulations to perform
        "n_mcmc_samples": 1000,   # Number of posterior samples per chain
        "n_mcmc_chains": 4,       # Number of MCMC chains
        "thin": 10                # Thinning factor for posterior samples
    }
    
    Alternatively, you can specify a study_config_path instead of study_design_path
    to generate a new study design from a configuration file.
    
Examples:
    # Run with default configuration
    python scripts/run_sbc.py
    
    # Run with a custom configuration file
    python scripts/run_sbc.py --config configs/my_sbc_config.json

Notes:
    SBC is more rigorous than parameter recovery, as it checks not just point
    estimates but the entire posterior distribution. If SBC is successful,
    ranks of true parameters within posterior samples should follow a uniform
    distribution.
"""
import os
import sys
import json
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.sbc import SimulationBasedCalibration
from utils.study_design import StudyDesign
from utils.study_design_m1 import StudyDesignM1

def run_from_config(config_path):
    """
    Run simulation-based calibration analysis from a configuration file.
    
    Parameters:
        config_path (str): Path to the configuration file
        
    Returns:
        SimulationBasedCalibration: The analysis instance after running
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract parameters with defaults
    sbc_model_path = config.get('sbc_model_path', None)  # Use default if not specified
    output_dir = config.get('output_dir', None)
    
    # Extract numerical parameters with defaults
    n_sbc_sims = config.get('n_sbc_sims', 100)
    n_mcmc_samples = config.get('n_mcmc_samples', 1000)
    n_mcmc_chains = config.get('n_mcmc_chains', 4)
    thin = config.get('thin', 1)  # Default to no thinning
    
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
    elif 'study_design_config' in config:
        # Create study design from embedded config
        design_config = config['study_design_config']
        
        # Check if this is an m_1 model (has both M and N)
        if 'N' in design_config:
            # Create m_1 study design
            study_design = StudyDesignM1(
                M=design_config.get('M', 20),
                N=design_config.get('N', 20),
                K=design_config.get('K', 3),
                D=design_config.get('D', 2),
                R=design_config.get('R', 10),
                S=design_config.get('S', 8),
                min_alts_per_problem=design_config.get('min_alts_per_problem', 2),
                max_alts_per_problem=design_config.get('max_alts_per_problem', 5),
                risky_probs=design_config.get('risky_probs', 'fixed'),
                feature_dist=design_config.get('feature_dist', 'normal'),
                feature_params=design_config.get('feature_params', {"loc": 0, "scale": 1})
            )
            print(f"Created m_1 study design: M={design_config['M']}, N={design_config['N']}")
        else:
            # Create standard study design
            study_design = StudyDesign(
                M=design_config.get('M', 20),
                K=design_config.get('K', 3),
                D=design_config.get('D', 2),
                R=design_config.get('R', 10),
                min_alts_per_problem=design_config.get('min_alts_per_problem', 2),
                max_alts_per_problem=design_config.get('max_alts_per_problem', 5),
                feature_dist=design_config.get('feature_dist', 'normal'),
                feature_params=design_config.get('feature_params', {"loc": 0, "scale": 1})
            )
            print(f"Created study design: M={design_config['M']}")
        
        # Generate the design
        study_design.generate()
    
    # Initialize and run analysis
    sbc = SimulationBasedCalibration(
        sbc_model_path=sbc_model_path,
        study_design=study_design,
        output_dir=output_dir,
        n_sbc_sims=n_sbc_sims,
        n_mcmc_samples=n_mcmc_samples,
        n_mcmc_chains=n_mcmc_chains,
        thin=thin
    )
    
    # Run the analysis
    ranks, pars = sbc.run()
    
    print(f"Simulation-based calibration analysis completed.")
    print(f"Results saved to: {sbc.output_dir}")
    
    return sbc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation-based calibration analysis")
    parser.add_argument('--config', type=str, default='configs/sbc_config.json', 
                        help='Path to configuration file')
    args = parser.parse_args()
    
    run_from_config(args.config)