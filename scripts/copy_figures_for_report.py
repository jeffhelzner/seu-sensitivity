"""
Copy figures from analyses to reports directory.

This script copies the relevant plots from analysis output directories
to the reports/figures directory for inclusion in reports.
"""
import os
import shutil
from pathlib import Path
import argparse

# Define source and destination directories
project_root = Path(__file__).parent.parent
dest_dir = project_root / "reports" / "figures"

# Create destination directory if it doesn't exist
dest_dir.mkdir(parents=True, exist_ok=True)

def copy_prior_predictive_figures(source_dir_name="example_prior_analysis"):
    """Copy prior predictive analysis figures."""
    source_dir = project_root / "results" / "prior_predictive" / source_dir_name
    
    files_to_copy = {
        # Study design files
        "study_design.json": "../study_design.json",
        "study_design_plots/alt_frequency.png": "alt_frequency.png",
        "study_design_plots/alts_per_problem.png": "alts_per_problem.png",
        
        # Prior analysis plots
        "parameters/alpha_dist.png": "alpha_dist.png",
        "expected_utilities/expected_utilities.png": "expected_utilities.png",
        "choices/choice_distribution.png": "choice_distribution.png",
        "seu_maximizer_selection/prob_seu_max_by_problem.png": "prob_seu_max_by_problem.png",
        "seu_maximizer_selection/total_seu_max_distribution.png": "total_seu_max_distribution.png",
    }
    
    print("Copying prior predictive analysis figures...")
    copy_files(source_dir, files_to_copy)

def copy_parameter_recovery_figures(source_dir_name="example_prior_recovery"):
    """Copy parameter recovery analysis figures."""
    recovery_base = project_root / "results" / "parameter_recovery"
    
    if source_dir_name is None:
        # Find the most recent parameter recovery run
        if not recovery_base.exists():
            print("Warning: No parameter recovery results found")
            return
        
        # Look for any subdirectories (both "run_*" and other naming conventions)
        run_dirs = [d for d in recovery_base.iterdir() if d.is_dir()]
        if not run_dirs:
            print("Warning: No parameter recovery runs found")
            return
        
        source_dir = max(run_dirs, key=lambda d: d.stat().st_mtime) / "recovery_summary"
        print(f"Using most recent run: {source_dir.parent.name}")
    else:
        source_dir = recovery_base / source_dir_name / "recovery_summary"
    
    files_to_copy = {
        # Summary table
        "summary_table.csv": "../recovery_summary_table.csv",
        
        # Alpha coverage plot
        "alpha_coverage_intervals.png": "recovery_alpha_coverage.png",
    }
    
    # Add beta coverage plots - adjust based on your K and D values
    # Assuming K=3, D=2 from the study design
    for k in range(1, 4):  # K=3
        for d in range(1, 3):  # D=2
            files_to_copy[f"beta_{k}_{d}_coverage.png"] = f"recovery_beta_{k}_{d}_coverage.png"
    
    # Add delta coverage plots - adjust based on K-1
    for k in range(1, 3):  # K-1 = 2 for K=3
        files_to_copy[f"delta_{k}_coverage.png"] = f"recovery_delta_{k}_coverage.png"
    
    print("\nCopying parameter recovery figures...")
    copy_files(source_dir, files_to_copy)

def copy_sample_size_figures(source_dir_name="custom_run"):
    """Copy sample size estimation figures."""
    sample_size_base = project_root / "results" / "sample_size_estimation"
    
    if source_dir_name is None:
        # Find the most recent sample size run
        if not sample_size_base.exists():
            print("Warning: No sample size estimation results found")
            return
        
        run_dirs = [d for d in sample_size_base.iterdir() if d.is_dir()]
        if not run_dirs:
            print("Warning: No sample size estimation runs found")
            return
        
        source_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
        print(f"Using most recent run: {source_dir.name}")
    else:
        source_dir = sample_size_base / source_dir_name
    
    if not source_dir.exists():
        print(f"Warning: Sample size directory not found: {source_dir}")
        return
    
    files_to_copy = {
        # Summary data
        "alpha_interval_widths.csv": "../sample_size_alpha_widths.csv",
        "recovery_summary_vs_M.csv": "../sample_size_recovery_summary.csv",
        
        # Main comparison plots
        "alpha_precision_vs_M.png": "sample_size_alpha_precision.png",
        "ci_width_vs_M_comparison.png": "sample_size_ci_width_comparison.png",
        "rmse_vs_M_comparison.png": "sample_size_rmse_comparison.png",
        "mae_vs_M_comparison.png": "sample_size_mae_comparison.png",
        "coverage_vs_M_comparison.png": "sample_size_coverage_comparison.png",
    }
    
    print("\nCopying sample size estimation figures...")
    copy_files(source_dir, files_to_copy)

def copy_files(source_dir, files_to_copy):
    """Helper function to copy files."""
    for source_file, dest_file in files_to_copy.items():
        source_path = source_dir / source_file
        
        # Handle relative paths for destination
        if dest_file.startswith("../"):
            dest_path = dest_dir.parent / dest_file[3:]
        else:
            dest_path = dest_dir / dest_file
        
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            print(f"  ✓ Copied: {source_file} -> {dest_path.relative_to(project_root)}")
        else:
            print(f"  ⚠ Source file not found: {source_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy analysis figures to reports directory")
    parser.add_argument('--analysis', type=str, 
                        choices=['prior', 'recovery', 'sample_size', 'all'], 
                        default='all', help='Which analysis figures to copy')
    parser.add_argument('--prior-dir', type=str, default='example_prior_analysis',
                        help='Prior predictive analysis directory name')
    parser.add_argument('--recovery-dir', type=str, default='example_prior_recovery',
                        help='Parameter recovery directory name')
    parser.add_argument('--sample-size-dir', type=str, default='custom_run',
                        help='Sample size estimation directory name')
    args = parser.parse_args()
    
    if args.analysis in ['prior', 'all']:
        copy_prior_predictive_figures(args.prior_dir)
    
    if args.analysis in ['recovery', 'all']:
        copy_parameter_recovery_figures(args.recovery_dir)
    
    if args.analysis in ['sample_size', 'all']:
        copy_sample_size_figures(args.sample_size_dir)
    
    print(f"\n✓ All requested figures copied to reports directory")
