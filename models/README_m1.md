# m_1 Model Implementation Guide

**Date:** December 6, 2025  
**Model:** m_1 - Combined Risky and Uncertain Choice  
**Purpose:** Separate identification of utility and subjective probability parameters

---

## Overview

The `m_1` model extends the base `m_0` model by combining two types of choice data:

1. **Uncertain choice problems**: Probabilities derived from features via β (like m_0)
2. **Risky choice problems**: Objective probabilities provided as data (new)

This combination enables separate identification of:
- **Utility function** (υ/δ) - identified primarily from risky choices
- **Subjective probability mapping** (β) - identified from uncertain choices given the utility function
- **Sensitivity parameter** (α) - shared across both contexts

---

## Key Assumptions

- Same α (sensitivity) applies in both risky and uncertain contexts
- Same utility function (υ) applies in both contexts
- Risky choices involve only known objective probabilities
- Uncertain choices involve only feature-derived probabilities

---

## Files Created

### Stan Models

1. **`models/m_1.stan`** - Main inference model
   - Estimates parameters from observed choices
   - Combines uncertain and risky likelihood components
   - Outputs: posterior samples, log-likelihoods, posterior predictions

2. **`models/m_1_sim.stan`** - Simulation model
   - Generates synthetic data for testing
   - Takes features (w) and risky probabilities (x) as input
   - Outputs: simulated choices (y, z) and true parameters

3. **`models/m_1_sbc.stan`** - Simulation-based calibration model
   - For validating inference via SBC
   - Follows rstan conventions with trailing underscores

### Python Modules

4. **`utils/study_design_m1.py`** - Study design generator
   - Extends base StudyDesign class
   - Generates both uncertain (w, I) and risky (x, J) components
   - Supports multiple risky probability generation methods
   - Creates visualization plots

5. **`scripts/run_m1_study_design.py`** - CLI for design generation
   - Command-line interface for creating study designs
   - Supports config files or direct parameters
   - Saves designs with metadata and plots

6. **`scripts/test_m1_model.py`** - Comprehensive validation suite
   - Tests all three Stan models
   - Validates study design generation
   - Checks data formats and dimensions
   - Provides detailed test reports

### Configuration Files

7. **`configs/m1_study_config.json`** - Study design configuration
8. **`configs/m1_parameter_recovery_config.json`** - Parameter recovery settings
9. **`configs/m1_sbc_config.json`** - SBC configuration
10. **`configs/m1_prior_analysis_config.json`** - Prior predictive analysis

---

## Data Structure

### Uncertain Choice Problems (from m_0)

```json
{
  "M": 20,              // number of uncertain problems
  "K": 3,               // number of consequences
  "D": 2,               // feature dimensions
  "R": 10,              // number of uncertain alternatives
  "w": [[...], ...],    // R feature vectors of dimension D
  "I": [[...], ...],    // M×R indicator matrix
  "y": [1, 2, ...]      // M choices (for inference only)
}
```

### Risky Choice Problems (new)

```json
{
  "N": 20,              // number of risky problems
  "S": 8,               // number of risky alternatives
  "x": [[...], ...],    // S probability simplexes of dimension K
  "J": [[...], ...],    // N×S indicator matrix
  "z": [1, 2, ...]      // N choices (for inference only)
}
```

---

## Quick Start

### 1. Generate a Study Design

```bash
# Using configuration file
python scripts/run_m1_study_design.py --config configs/m1_study_config.json

# Using command-line arguments
python scripts/run_m1_study_design.py \
    --M 30 --N 30 --K 3 --D 2 --R 15 --S 12 \
    --risky-probs fixed \
    --output my_m1_design.json
```

### 2. Run Validation Tests

```bash
# Quick test
python scripts/test_m1_model.py

# Detailed output
python scripts/test_m1_model.py --verbose
```

### 3. Use in Analysis Scripts

```python
from utils.study_design_m1 import StudyDesignM1
from cmdstanpy import CmdStanModel

# Generate design
design = StudyDesignM1(M=20, N=20, K=3, D=2, R=10, S=8)
design.generate()
data = design.get_data_dict()

# Simulate data
sim_model = CmdStanModel(stan_file="models/m_1_sim.stan")
sim_fit = sim_model.sample(
    data=data,
    fixed_param=True,
    iter_sampling=1,
    chains=1
)

# Extract choices
sim_df = sim_fit.draws_pd()
y = [int(sim_df[f'y[{i+1}]'].values[0]) for i in range(data['M'])]
z = [int(sim_df[f'z[{i+1}]'].values[0]) for i in range(data['N'])]

# Run inference
data['y'] = y
data['z'] = z
inf_model = CmdStanModel(stan_file="models/m_1.stan")
inf_fit = inf_model.sample(data=data, chains=4)
```

---

## Parameters

### Model Parameters

| Parameter | Type | Dimension | Prior | Description |
|-----------|------|-----------|-------|-------------|
| α (alpha) | real+ | scalar | lognormal(0,1) | Sensitivity to utility differences (shared) |
| β (beta) | matrix | K×D | std_normal() | Feature-to-probability mapping (uncertain only) |
| δ (delta) | simplex | K-1 | dirichlet(1) | Utility increments (shared) |

### Derived Quantities

| Quantity | Type | Dimension | Description |
|----------|------|-----------|-------------|
| υ (upsilon) | ordered | K | Cumulative utilities: cumsum([0, δ]) |
| ψ (psi) | simplex | varies | Subjective probabilities for uncertain alts |
| η_uncertain | vector | varies | Expected utilities for uncertain alts |
| η_risky | vector | varies | Expected utilities for risky alts |
| χ_uncertain | simplex | varies | Choice probabilities for uncertain problems |
| χ_risky | simplex | varies | Choice probabilities for risky problems |

---

## Risky Probability Generation Methods

The `risky_probs` parameter in StudyDesignM1 controls how objective probabilities are generated:

### 1. `"uniform"` 
All consequences equally likely: [1/K, 1/K, ..., 1/K]

### 2. `"fixed"` (Recommended for K=3)
Common probability values designed for good coverage:
- [0.25, 0.50, 0.25]
- [0.50, 0.25, 0.25]
- [0.25, 0.25, 0.50]
- [0.33, 0.33, 0.34]
- [0.20, 0.60, 0.20]
- [0.60, 0.20, 0.20]
- [0.20, 0.20, 0.60]
- [0.10, 0.80, 0.10]

### 3. `"random"`
Random simplexes from Dirichlet(1, 1, ..., 1)

---

## Model Comparison

| Feature | m_0 | m_01 | m_1 |
|---------|-----|------|-----|
| Uncertain choices | ✓ | ✓ | ✓ |
| Risky choices | ✗ | ✗ | ✓ |
| β identifiable | ✗ | ✗ | ✓ |
| υ identifiable | ✗ | ✗ | ✓ |
| α identifiable | ~ | ~ | ✓ |
| Prior on δ | dirichlet(1) | dirichlet(5) | dirichlet(1) |

---

## Expected Outcomes

### Identification Benefits

1. **Utility function (υ)** directly identified from risky choices where probabilities are known
2. **Subjective probability mapping (β)** identified from uncertain choices given υ
3. **No confounding** between utility curvature and probability distortion
4. **Better α estimation** due to more varied choice contexts

### Parameter Recovery

Expected to see:
- Tighter credible intervals for all parameters vs. m_0
- Unbiased recovery of β coefficients
- Unbiased recovery of utility differences
- Good coverage (95% intervals contain true value ~95% of the time)

### Validation Metrics

Monitor via SBC:
- Rank statistics should be uniform
- ECDF should stay within tolerance bands
- No systematic bias in any direction

---

## Troubleshooting

### Common Issues

**Issue:** "y[m] must be <= N_uncertain[m]"
- **Cause:** Choice index out of bounds for problem m
- **Solution:** Check that y values are 1-indexed and ≤ number of alternatives in that problem

**Issue:** "Simplex does not sum to 1"
- **Cause:** Risky probability vectors don't sum to 1
- **Solution:** Ensure all x[s] vectors are valid simplexes

**Issue:** Slow sampling
- **Cause:** Large number of problems or alternatives
- **Solution:** Start with smaller designs (M=10, N=10) and scale up

**Issue:** Poor mixing
- **Cause:** Weak identification or complex posterior
- **Solution:** Check design has enough variation in both uncertain and risky contexts

---

## Next Steps

1. **Run validation tests:**
   ```bash
   python scripts/test_m1_model.py --verbose
   ```

2. **Generate initial design:**
   ```bash
   python scripts/run_m1_study_design.py --config configs/m1_study_config.json
   ```

3. **Perform parameter recovery:**
   - Adapt `analysis/parameter_recovery.py` to use m_1 models
   - Run with `configs/m1_parameter_recovery_config.json`

4. **Run SBC:**
   - Adapt `analysis/sbc.py` to use m_1_sbc.stan
   - Run with `configs/m1_sbc_config.json`

5. **Compare with m_0:**
   - Run same design through both models
   - Compare parameter recovery performance
   - Document identification improvements

---

## References

- Original instructions: `instructions.md`
- Base model: `models/m_0.stan`
- Study design utilities: `utils/study_design.py`
- CmdStanPy documentation: https://mc-stan.org/cmdstanpy/

---

**Questions or Issues?** Review the test output from `test_m1_model.py` or check the Stan model comments for detailed documentation.
