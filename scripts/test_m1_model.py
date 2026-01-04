#!/usr/bin/env python3
"""
Test and validation script for m_1 model implementation.

This script validates the m_1 model by:
1. Generating a small study design
2. Simulating data from the model
3. Running inference on the simulated data
4. Checking that the model runs without errors
5. Validating output formats and dimensions

Usage:
    python scripts/test_m1_model.py
    python scripts/test_m1_model.py --verbose
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.study_design_m1 import StudyDesignM1


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_study_design(verbose=False):
    """Test study design generation for m_1."""
    print_section("TEST 1: Study Design Generation")
    
    # Create a small test design
    design = StudyDesignM1(
        M=5,   # 5 uncertain problems
        N=5,   # 5 risky problems
        K=3,   # 3 consequences
        D=2,   # 2 feature dimensions
        R=6,   # 6 uncertain alternatives
        S=5,   # 5 risky alternatives
        min_alts_per_problem=2,
        max_alts_per_problem=3,
        risky_probs='fixed'
    )
    
    design.generate()
    data = design.get_data_dict()
    
    # Validate uncertain problem structure
    assert data['M'] == 5, "Incorrect number of uncertain problems"
    assert data['K'] == 3, "Incorrect number of consequences"
    assert data['D'] == 2, "Incorrect feature dimensions"
    assert data['R'] == 6, "Incorrect number of uncertain alternatives"
    assert len(data['w']) == 6, "Incorrect w array length"
    assert len(data['I']) == 5, "Incorrect I array length"
    
    # Validate risky problem structure
    assert data['N'] == 5, "Incorrect number of risky problems"
    assert data['S'] == 5, "Incorrect number of risky alternatives"
    assert len(data['x']) == 5, "Incorrect x array length"
    assert len(data['J']) == 5, "Incorrect J array length"
    
    # Validate probability simplexes sum to 1
    for i, simplex in enumerate(data['x']):
        simplex_sum = sum(simplex)
        assert abs(simplex_sum - 1.0) < 1e-6, f"Simplex {i} does not sum to 1: {simplex_sum}"
    
    print("✓ Study design generation: PASSED")
    
    if verbose:
        print(f"\n  Uncertain problems: {data['M']}")
        print(f"  Risky problems: {data['N']}")
        print(f"  Total problems: {data['M'] + data['N']}")
        print(f"  Consequences: {data['K']}")
        print(f"  Uncertain alternatives: {data['R']}")
        print(f"  Risky alternatives: {data['S']}")
    
    return design, data


def test_simulation_model(design, data, verbose=False):
    """Test the m_1_sim.stan simulation model."""
    print_section("TEST 2: Simulation Model")
    
    # Find and compile the simulation model
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "m_1_sim.stan"
    )
    
    if not os.path.exists(model_path):
        print(f"✗ Simulation model not found at: {model_path}")
        return None
    
    print(f"  Compiling simulation model from: {model_path}")
    sim_model = CmdStanModel(stan_file=model_path)
    print("  ✓ Model compiled successfully")
    
    # Add parameter generation controls to data
    sim_data = data.copy()
    sim_data.update({
        'alpha_mean': 0.0,
        'alpha_sd': 1.0,
        'beta_sd': 1.0
    })
    
    # Run simulation
    print("  Running simulation...")
    sim_fit = sim_model.sample(
        data=sim_data,
        seed=12345,
        iter_sampling=1,
        iter_warmup=0,
        chains=1,
        fixed_param=True,
        adapt_engaged=False
    )
    
    # Extract results
    sim_df = sim_fit.draws_pd()
    
    # Validate outputs
    assert 'alpha' in sim_df.columns, "alpha not in output"
    assert 'delta[1]' in sim_df.columns, "delta not in output"
    
    # Check that choices were generated
    y_cols = [f'y[{i+1}]' for i in range(data['M'])]
    z_cols = [f'z[{i+1}]' for i in range(data['N'])]
    
    for col in y_cols:
        assert col in sim_df.columns, f"Missing uncertain choice: {col}"
    
    for col in z_cols:
        assert col in sim_df.columns, f"Missing risky choice: {col}"
    
    print("✓ Simulation model: PASSED")
    
    if verbose:
        print(f"\n  Generated parameters:")
        print(f"    alpha: {sim_df['alpha'].values[0]:.3f}")
        print(f"    delta: {[sim_df[f'delta[{i+1}]'].values[0] for i in range(data['K']-1)]}")
        print(f"  Generated {data['M']} uncertain choices and {data['N']} risky choices")
    
    return sim_fit, sim_df


def test_inference_model(design, data, sim_df, verbose=False):
    """Test the m_1.stan inference model."""
    print_section("TEST 3: Inference Model")
    
    # Find and compile the inference model
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "m_1.stan"
    )
    
    if not os.path.exists(model_path):
        print(f"✗ Inference model not found at: {model_path}")
        return None
    
    print(f"  Compiling inference model from: {model_path}")
    inf_model = CmdStanModel(stan_file=model_path)
    print("  ✓ Model compiled successfully")
    
    # Prepare inference data (add simulated choices)
    inf_data = data.copy()
    
    # Extract choices from simulation
    y = [int(sim_df[f'y[{i+1}]'].values[0]) for i in range(data['M'])]
    z = [int(sim_df[f'z[{i+1}]'].values[0]) for i in range(data['N'])]
    
    inf_data['y'] = y
    inf_data['z'] = z
    
    # Run inference (short run for testing)
    print("  Running inference (test run: 100 samples)...")
    inf_fit = inf_model.sample(
        data=inf_data,
        seed=54321,
        iter_sampling=100,
        iter_warmup=100,
        chains=2,
        show_console=verbose
    )
    
    # Extract results
    inf_df = inf_fit.draws_pd()
    
    # Validate outputs
    assert 'alpha' in inf_df.columns, "alpha not in output"
    assert 'delta[1]' in inf_df.columns, "delta not in output"
    assert 'upsilon[1]' in inf_df.columns, "upsilon not in output"
    
    # Check generated quantities
    assert 'log_lik_total' in inf_df.columns, "log_lik_total not in output"
    assert 'log_lik_uncertain[1]' in inf_df.columns, "log_lik_uncertain not in output"
    assert 'log_lik_risky[1]' in inf_df.columns, "log_lik_risky not in output"
    assert 'y_pred[1]' in inf_df.columns, "y_pred not in output"
    assert 'z_pred[1]' in inf_df.columns, "z_pred not in output"
    
    print("✓ Inference model: PASSED")
    
    if verbose:
        # Compare true vs estimated parameters
        true_alpha = sim_df['alpha'].values[0]
        est_alpha_mean = inf_df['alpha'].mean()
        
        print(f"\n  Parameter comparison (true vs. estimated mean):")
        print(f"    alpha: {true_alpha:.3f} vs. {est_alpha_mean:.3f}")
        
        print(f"\n  Posterior summary statistics:")
        summary = inf_fit.summary()
        print(f"    Total parameters: {len(summary)}")
        print(f"    R-hat max: {summary['R_hat'].max():.3f}")
        print(f"    ESS bulk min: {summary['ess_bulk'].min():.0f}")
    
    return inf_fit, inf_df


def test_sbc_model(design, data, verbose=False):
    """Test the m_1_sbc.stan model."""
    print_section("TEST 4: SBC Model")
    
    # Find and compile the SBC model
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "m_1_sbc.stan"
    )
    
    if not os.path.exists(model_path):
        print(f"✗ SBC model not found at: {model_path}")
        return None
    
    print(f"  Compiling SBC model from: {model_path}")
    sbc_model = CmdStanModel(stan_file=model_path)
    print("  ✓ Model compiled successfully")
    
    # Add parameter generation controls to data
    sbc_data = data.copy()
    sbc_data.update({
        'alpha_mean': 0.0,
        'alpha_sd': 1.0,
        'beta_sd': 1.0
    })
    
    # Run SBC (short run for testing)
    print("  Running SBC test...")
    sbc_fit = sbc_model.sample(
        data=sbc_data,
        seed=99999,
        iter_sampling=50,
        iter_warmup=50,
        chains=1,
        show_console=verbose
    )
    
    # Extract results
    sbc_df = sbc_fit.draws_pd()
    
    # Validate outputs
    assert 'pars_[1]' in sbc_df.columns, "pars_ not in output"
    assert 'y_[1]' in sbc_df.columns, "y_ not in output"
    assert 'z_[1]' in sbc_df.columns, "z_ not in output"
    
    print("✓ SBC model: PASSED")
    
    if verbose:
        print(f"\n  Generated {data['M']} uncertain choices (y_)")
        print(f"  Generated {data['N']} risky choices (z_)")
        n_params = data['K'] * data['D'] + data['K']  # beta + delta + alpha
        print(f"  Flattened {n_params} parameters into pars_")
    
    return sbc_fit, sbc_df


def run_all_tests(verbose=False):
    """Run all tests and provide summary."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "M_1 MODEL VALIDATION SUITE" + " "*27 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {
        'study_design': False,
        'simulation': False,
        'inference': False,
        'sbc': False
    }
    
    try:
        # Test 1: Study Design
        design, data = test_study_design(verbose)
        results['study_design'] = True
        
        # Test 2: Simulation Model
        sim_fit, sim_df = test_simulation_model(design, data, verbose)
        if sim_fit is not None:
            results['simulation'] = True
        
        # Test 3: Inference Model
        if sim_fit is not None:
            inf_fit, inf_df = test_inference_model(design, data, sim_df, verbose)
            if inf_fit is not None:
                results['inference'] = True
        
        # Test 4: SBC Model
        sbc_fit, sbc_df = test_sbc_model(design, data, verbose)
        if sbc_fit is not None:
            results['sbc'] = True
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        if verbose:
            traceback.print_exc()
    
    # Print summary
    print_section("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name.replace('_', ' ').title():<20} {status}")
    
    print(f"\n  Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n  🎉 All tests PASSED! The m_1 model is ready to use.")
        return 0
    else:
        print("\n  ⚠️  Some tests failed. Please review the output above.")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Test and validate m_1 model implementation'
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed output')
    
    args = parser.parse_args()
    
    exit_code = run_all_tests(verbose=args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
