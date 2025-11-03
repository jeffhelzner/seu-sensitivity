"""
Study Design Generation Script

This script creates and manages experimental study designs for the Bayesian 
Decision Theory models, defining the structure of decision problems and 
alternatives that will be used in all subsequent analyses.

Purpose:
    - Generate study designs with specific characteristics (number of problems, alternatives, etc.)
    - Create feature spaces for alternatives with controlled distributions
    - Analyze the properties of generated study designs
    - Save study designs for consistent use across all analyses
    - Ensure reproducible experimental conditions across model evaluations

Usage:
    python scripts/run_study_design.py [--config CONFIG_PATH] [--output OUTPUT_PATH]
    
    The config file should be a JSON file with the following structure:
    {
        "M": 30,                   # Number of decision problems
        "K": 4,                    # Number of possible consequences
        "D": 3,                    # Dimensions to describe alternatives
        "R": 15,                   # Number of distinct alternatives
        "min_alts_per_problem": 2, # Minimum alternatives per problem
        "max_alts_per_problem": 6, # Maximum alternatives per problem
        "feature_dist": "uniform", # Distribution type for features
        "feature_params": {        # Parameters for the feature distribution
            "low": -2,
            "high": 2
        },
        "design_name": "my_design", # Name for the design
        "generate_on_load": true    # Whether to generate design immediately
    }
    
Examples:
    # Run with default configuration
    python scripts/run_study_design.py
    
    # Run with a custom configuration file
    python scripts/run_study_design.py --config configs/my_study_config.json
    
    # Specify output location
    python scripts/run_study_design.py --output results/designs/custom_design.json
    
    # Load and analyze an existing design
    python scripts/run_study_design.py --load results/designs/existing_design.json
"""
import os
import sys
import json
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.study_design import StudyDesign

def generate_from_config(config_path, output_path):
    """
    Generate a study design from a configuration file.
    
    Parameters:
        config_path (str): Path to the configuration file
        output_path (str): Path where the study design will be saved
        
    Returns:
        StudyDesign: The generated study design
    """
    # Create design from configuration
    design = StudyDesign.from_config(config_path)
    
    # Analyze the design properties
    design.analyze()
    
    # Save the design
    design.save(output_path)
    
    print(f"Study design generated from {config_path}")
    print(f"Results saved to: {output_path}")
    
    return design

def load_and_analyze(load_path, output_path=None):
    """
    Load an existing study design and analyze it.
    
    Parameters:
        load_path (str): Path to the existing study design
        output_path (str, optional): Path to save the analyzed design
        
    Returns:
        StudyDesign: The loaded study design
    """
    # Load existing design
    design = StudyDesign.load(load_path)
    
    # Analyze the design properties
    design.analyze()
    
    # Save if output path is provided
    if output_path:
        design.save(output_path)
        print(f"Study design saved to: {output_path}")
    
    print(f"Study design loaded from {load_path} and analyzed")
    
    return design

def create_default(output_path):
    """
    Create a study design with default parameters.
    
    Parameters:
        output_path (str): Path where the study design will be saved
        
    Returns:
        StudyDesign: The generated study design
    """
    # Create with default parameters
    design = StudyDesign()
    design.generate()
    
    # Analyze the design properties
    design.analyze()
    
    # Save the design
    design.save(output_path)
    
    print(f"Default study design generated")
    print(f"Results saved to: {output_path}")
    
    return design

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or manage study designs")
    
    # Define the mutually exclusive options for the source
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--config', type=str, default='configs/study_config.json',
                        help='Path to configuration file for generating a new design')
    source_group.add_argument('--load', type=str,
                        help='Path to existing study design to load and analyze')
    source_group.add_argument('--default', action='store_true',
                        help='Create a design with default parameters')
    
    # Define the output path
    parser.add_argument('--output', type=str, default='results/designs/study.json',
                        help='Path where the study design will be saved')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Process based on selected option
    if args.load:
        load_and_analyze(args.load, args.output)
    elif args.default:
        create_default(args.output)
    else:  # Default to config
        generate_from_config(args.config, args.output)