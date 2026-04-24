# SEU Sensitivity

![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

**Note**: This is an active research project. Code and documentation are evolving.

A Bayesian framework for modeling and analyzing decision-making behavior through the lens of Subjective Expected Utility (SEU) theory. This project provides tools for generating experimental designs, fitting computational models using Stan, and assessing the rationality of decision makers—including Large Language Models (LLMs).

## Development Status

🚧 **This project is currently under active development** 🚧

Current status:
- ✅ Core theoretical framework established
- ✅ Base Stan model (m_0) for uncertain choice implemented and tested
- ✅ Combined model (m_1) with risky and uncertain choice implemented and tested
- ✅ Separate sensitivity model (m_2) implemented and tested
- ✅ Proportional sensitivity model (m_3) implemented and tested
- ✅ Calibrated prior variants (m_01, m_02, m_11, m_21, m_31) implemented
- ✅ Study design tools functional (m_0 and m_1)
- ✅ Analysis pipeline complete (parameter recovery, SBC, prior/posterior predictive)
- ✅ Quarto-based documentation and reports
- ✅ Temperature study application complete
- ✅ Temperature study with EU prompt complete
- ✅ Temperature study with risky alternatives complete
- ✅ Ellsberg study (Claude 3.5 Sonnet) complete
- ✅ Claude insurance study complete
- ✅ GPT-4o Ellsberg study complete
- ✅ 2×2 factorial synthesis complete
- ✅ Hierarchical model (h_m01) with cell-level regression on α implemented and tested
- ✅ Alignment study application scaffolded (6 model × 3 prompt factorial)
- 📝 Documentation being expanded
- 🔬 Empirical validation ongoing

**Note for users**: While the core functionality is stable, API and features may change as the project evolves. Feedback and contributions are welcome!

## Overview

This framework implements a computational approach to understanding decision makers who form beliefs and make choices under uncertainty. The core insight is that we can measure an agent's "rationality" by estimating their sensitivity parameter (α), which governs how consistently they maximize subjective expected utility.

**Key Features:**

- **Theoretical Foundation**: Framework for softmax choice models with SEU
- **Stan Models**: Bayesian inference for rationality parameters using Hamiltonian Monte Carlo
- **Study Design Tools**: Generate and analyze experimental designs for decision studies
- **LLM Benchmarking**: Assess the rationality of Large Language Models through text-based decision problems
- **Visualization**: Create comprehensive plots and diagnostics for model results

## Project Structure

```
seu-sensitivity/
├── reports/                 # Quarto-based documentation and reports
│   ├── _quarto.yml         # Quarto project configuration
│   ├── _metadata.yml       # Shared metadata for reports
│   ├── index.qmd           # Main documentation index
│   ├── report_utils.py     # Shared Python utilities for reports
│   ├── references.bib      # Bibliography
│   ├── foundations/        # Foundational theoretical reports
│   │   ├── 01_abstract_formulation.qmd   # Mathematical framework
│   │   ├── 02_concrete_implementation.qmd # Implementation details
│   │   ├── 03_prior_analysis.qmd         # Prior predictive analysis
│   │   ├── 04_parameter_recovery.qmd     # Parameter recovery study
│   │   ├── 05_adding_risky_choices.qmd   # m_1 model development
│   │   ├── 06_sbc_validation.qmd         # Simulation-based calibration
│   │   ├── 07_generalizing_sensitivity.qmd # m_2 and m_3 model development
│   │   ├── 08_hierarchical_formulation.qmd   # Hierarchical model theory
│   │   ├── 09_hierarchical_implementation.qmd # h_m01 Stan implementation
│   │   ├── 10_hierarchical_prior_analysis.qmd # Hierarchical prior predictive
│   │   ├── 11_hierarchical_parameter_recovery.qmd # Hierarchical recovery
│   │   └── 12_hierarchical_sbc_validation.qmd # Hierarchical SBC
│   ├── applications/       # Applied research reports
│   │   ├── temperature_study/
│   │   ├── temperature_study_with_eu_prompt/
│   │   ├── temperature_study_with_risky_alts/
│   │   ├── ellsberg_study/
│   │   ├── claude_insurance_study/
│   │   ├── gpt4o_ellsberg_study/
│   │   └── factorial_synthesis/
│   ├── blog/               # Blog-style posts
│   ├── styles/             # Custom CSS/SCSS styles
│   └── legacy/             # Archived legacy reports
├── models/                  # Stan model implementations
│   ├── m_0.stan            # Base SEU model (uncertain choice only)
│   ├── m_0_sim.stan        # m_0 simulation model
│   ├── m_0_sbc.stan        # m_0 SBC model
│   ├── m_01.stan           # m_0 with calibrated priors
│   ├── m_01_sbc.stan       # m_01 SBC model
│   ├── m_01_sim.stan       # m_01 simulation model
│   ├── m_02.stan           # m_0 with calibrated priors for Ellsberg (K=4)
│   ├── m_02_sbc.stan       # m_02 SBC model
│   ├── m_02_sim.stan       # m_02 simulation model
│   ├── m_1.stan            # Combined model (risky + uncertain, shared α)
│   ├── m_11.stan           # m_1 with calibrated priors
│   ├── m_1_sim.stan        # m_1 simulation model
│   ├── m_1_sbc.stan        # m_1 SBC model
│   ├── m_2.stan            # Separate sensitivity model (α for uncertain, ω for risky)
│   ├── m_21.stan           # m_2 with calibrated priors
│   ├── m_2_sim.stan        # m_2 simulation model
│   ├── m_2_sbc.stan        # m_2 SBC model
│   ├── m_3.stan            # Proportional sensitivity model (ω = κα)
│   ├── m_31.stan           # m_3 with calibrated priors
│   ├── m_3_sim.stan        # m_3 simulation model
│   ├── m_3_sbc.stan        # m_3 SBC model
│   ├── h_m01.stan          # Hierarchical model (regression on log-α across cells)
│   ├── h_m01_sim.stan      # h_m01 simulation model
│   ├── h_m01_sbc.stan      # h_m01 SBC model
│   └── README_m1.md        # m_1 implementation guide
├── utils/                   # Core utilities
│   ├── __init__.py         # Shared utilities, model detection
│   ├── study_design.py     # Experimental design generation (m_0)
│   ├── study_design_m1.py  # Extended design for m_1
│   ├── study_design_hierarchical.py # Hierarchical design (stacked cells)
│   └── README.md           # Utils documentation
├── analysis/                # Analysis scripts
│   ├── model_estimation.py # Model fitting utilities
│   ├── parameter_recovery.py # Parameter recovery analysis
│   ├── posterior_predictive_checks.py # Posterior predictive checks
│   ├── prior_predictive.py # Prior predictive checks
│   ├── sbc.py              # Simulation-based calibration
│   ├── sample_size_estimation.py # Sample size planning
│   ├── hierarchical_prior_predictive.py # Prior predictive for h_m01
│   ├── hierarchical_parameter_recovery.py # Recovery for h_m01
│   └── hierarchical_sbc.py # SBC for h_m01
├── applications/            # Applied research projects
│   ├── temperature_study/  # LLM temperature effects on sensitivity
│   ├── temperature_study_with_eu_prompt/ # Temperature study with EU-maximization prompt
│   ├── temperature_study_with_risky_alts/ # Risky choice collection for m_1/m_2/m_3
│   ├── ellsberg_study/     # Claude 3.5 Sonnet on Ellsberg urn gambles
│   ├── claude_insurance_study/ # Claude 3.5 Sonnet on insurance claims triage
│   ├── gpt4o_ellsberg_study/ # GPT-4o on Ellsberg urn gambles
│   ├── factorial_synthesis/ # Cross-LLM × cross-task synthesis inputs and outputs
│   └── alignment_study/    # 6-model × 3-prompt hierarchical alignment study
├── scripts/                 # Executable scripts
│   ├── run_study_design.py # Generate study designs
│   ├── run_m1_study_design.py # Generate m_1 study designs
│   ├── run_model_estimation.py # Fit models
│   ├── run_parameter_recovery.py # Run recovery analysis
│   ├── run_prior_predictive.py # Prior predictive analysis
│   ├── run_prior_predictive_grid.py # Prior predictive grid search (m_0)
│   ├── run_prior_predictive_grid_augmented.py # Prior predictive grid search (m_1/m_2/m_3)
│   ├── run_sbc.py          # SBC validation
│   ├── run_sample_size_estimation.py # Sample size analysis
│   ├── run_hierarchical_prior_predictive.py # Hierarchical prior predictive
│   ├── run_hierarchical_parameter_recovery.py # Hierarchical recovery
│   ├── run_hierarchical_sbc.py # Hierarchical SBC
│   ├── run_temperature_analysis.py # Temperature study analysis
│   ├── run_ellsberg_study.py # Ellsberg study entry point
│   ├── refit_with_ppc.py   # Refit models with posterior predictive checks
│   ├── extract_report_data.py # Extract parameter draws for reports
│   ├── freeze_report_data.py # Snapshot factorial cell data for reports
│   ├── freeze_eu_prompt_report_data.py # Snapshot EU prompt study outputs
│   ├── generate_primary_analysis.py # Generate primary analysis JSON for factorial cells
│   ├── generate_ellsberg_primary_analysis.py # Generate primary analysis for Ellsberg study
│   ├── copy_figures_for_report.py # Copy figures into reports
│   ├── cleanup_temp_files.py # Clean up temporary files
│   ├── smoke_test_design.py # Study design smoke test
│   ├── smoke_test_sim.py   # Simulation smoke test
│   ├── smoke_test_inference.py # Inference smoke test
│   ├── smoke_test_recovery.py # Parameter recovery smoke test
│   ├── smoke_test_sbc.py   # SBC smoke test
│   └── test_m1_model.py    # m_1 model tests
├── configs/                 # Configuration files for studies
├── results/                 # Generated results and outputs
│   ├── designs/            # Study designs
│   ├── parameter_recovery/ # Recovery analysis results
│   ├── prior_predictive/   # Prior predictive results
│   ├── sample_size_estimation/ # Sample size results
│   └── sbc/                # SBC results
├── prompts/                 # LLM prompt templates
├── environment.yml          # Conda environment specification
├── requirements.txt         # Pip requirements (alternative)
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.10+ (Python 3.10 recommended)
- Conda (recommended) or pip
- Stan (installed via CmdStanPy)

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd seu-sensitivity

# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate seu-sensitivity

# Install Stan
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd seu-sensitivity

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Stan
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

### For LLM Benchmarking

The project supports both OpenAI and Anthropic APIs for LLM studies:

```bash
# Set up API keys (create .env file or export directly)
echo "OPENAI_API_KEY=your-openai-key-here" >> .env
echo "ANTHROPIC_API_KEY=your-anthropic-key-here" >> .env
```

See `.env.example` for a template.

## Quick Start

### 1. Generate a Study Design

```python
from utils.study_design import StudyDesign

# Create a study design with 20 decision problems
design = StudyDesign(M=20, K=3, D=2, R=10)
design.generate()
design.analyze()
design.save("results/designs/my_study.json")
```

### 2. Fit the Model

```python
from cmdstanpy import CmdStanModel
import json

# Load study design
with open("results/designs/my_study.json", "r") as f:
    stan_data = json.load(f)

# Compile and fit model
model = CmdStanModel(stan_file="models/m_0.stan")
fit = model.sample(data=stan_data)

# Extract rationality parameter
alpha = fit.stan_variable("alpha")
print(f"Estimated sensitivity (α): {alpha.mean():.2f}")
```

### 3. Combined Risky and Uncertain Choice (m_1 Model)

For better parameter identification, use the m_1 model which combines risky (known probabilities) and uncertain (feature-derived probabilities) choice problems:

```python
from utils.study_design_m1 import StudyDesignM1
from cmdstanpy import CmdStanModel

# Create design with both risky (N) and uncertain (M) problems
design = StudyDesignM1(M=20, N=20, K=3, D=2, R=10, S=8)
design.generate()
design.save("results/designs/m1_study.json")

# Fit model
model = CmdStanModel(stan_file="models/m_1.stan")
fit = model.sample(data=design.get_data_dict())
```

See [models/README_m1.md](models/README_m1.md) for detailed m_1 documentation.

### 4. Run an Application Study

The application modules each contain their own workflow documentation and entry points. Start with one of the active study directories:

- [applications/temperature_study/README.md](applications/temperature_study/README.md)
- [applications/temperature_study_with_eu_prompt/README.md](applications/temperature_study_with_eu_prompt/README.md)
- [applications/temperature_study_with_risky_alts/README.md](applications/temperature_study_with_risky_alts/README.md)

## Analysis Pipeline

The project includes a complete analysis pipeline accessible via scripts:

```bash
# Generate a study design
python scripts/run_study_design.py --config configs/study_config.json

# Run prior predictive analysis
python scripts/run_prior_predictive.py --config configs/prior_analysis_config.json

# Run parameter recovery study
python scripts/run_parameter_recovery.py --config configs/parameter_recovery_config.json

# Run simulation-based calibration
python scripts/run_sbc.py --config configs/sbc_config.json

# Run sample size estimation
python scripts/run_sample_size_estimation.py --config configs/sample_size_config.json
```

For m_1, m_2, and m_3 models, use the corresponding `m1_*`, `m2_*`, and `m3_*` config files. For the hierarchical model h_m01, use the `run_hierarchical_*` scripts with `h_m01_*` configs:

```bash
# Hierarchical prior predictive
python scripts/run_hierarchical_prior_predictive.py --config configs/h_m01_prior_analysis_config.json

# Hierarchical parameter recovery
python scripts/run_hierarchical_parameter_recovery.py --config configs/h_m01_parameter_recovery_config.json

# Hierarchical SBC
python scripts/run_hierarchical_sbc.py --config configs/h_m01_sbc_config.json
```

## Theoretical Background

Overview of three fundamental properties of SEU sensitivity:

1. **Monotonicity**: Higher sensitivity increases probability of choosing value-maximizing alternatives
2. **Perfect Rationality Limit**: As α → ∞, agents deterministically choose optimal alternatives
3. **Random Choice Limit**: As α → 0, agents choose uniformly at random

These properties hold for any value function, with SEU providing the substantive behavioral interpretation.

**Key Result**: With utilities normalized to [0,1], the sensitivity parameter α has a precise interpretation as the log-odds change per unit of standardized SEU difference.

**SEU Maximizer Selection**: The prior predictive analysis tracks the probability of selecting SEU-maximizing alternatives for each problem, providing a diagnostic for model rationality under the prior.

See [reports/foundations/01_abstract_formulation.qmd](reports/foundations/01_abstract_formulation.qmd) for complete mathematical details and proofs. To render and view the reports locally:

```bash
cd reports
quarto render
open _output/index.html
```

## Model m_0 Specification

The base model (`models/m_0.stan`) implements a softmax choice model for uncertain choice problems:

- **Subjective probabilities** determined by alternative features through softmax transformation
- **Ordered utilities** with incremental differences on unit scale
- **Choice probabilities** following softmax of expected utilities scaled by sensitivity α

**Parameters:**
- `alpha`: Sensitivity to expected utility (≥ 0)
- `beta`: Feature-to-probability mapping (K × D matrix)
- `delta`: Utility increments on unit scale ((K-1)-simplex)

**Data Requirements:**
- `M`: Number of decision problems
- `K`: Number of possible consequences
- `D`: Feature dimensions
- `R`: Number of distinct alternatives
- `w`: Feature vectors for each alternative
- `I`: Indicator array (which alternatives in which problems)
- `y`: Observed choices

## Calibrated Prior Variants

- **m_01** (`models/m_01.stan`): Calibrated for K=3 insurance triage — α ~ lognormal(3.0, 0.75)
- **m_02** (`models/m_02.stan`): Calibrated for K=4 Ellsberg gambles — α ~ lognormal(3.5, 0.75)

These are structurally identical to m_0 but use priors calibrated via prior predictive analysis for their respective task domains. Corresponding calibrated variants exist for m_1 (m_11), m_2 (m_21), and m_3 (m_31).

## Model m_1 Specification

The combined model (`models/m_1.stan`) extends m_0 by adding risky choice problems with known objective probabilities:

**Additional Parameters for m_1:**
- `N`: Number of risky choice problems
- `S`: Number of risky alternatives
- `x`: Objective probability vectors for risky alternatives
- `J`: Indicator array for risky problems
- `z`: Observed risky choices

**Key Advantage:** Separate identification of utility function (from risky choices) and subjective probability mapping (from uncertain choices).

See [models/README_m1.md](models/README_m1.md) for detailed m_1 documentation.

## Model m_2 Specification

The separate-sensitivity model (`models/m_2.stan`) extends m_1 by allowing independent sensitivity parameters for uncertain and risky choices:

- `alpha`: Sensitivity for uncertain choices
- `omega`: Sensitivity for risky choices (independent of α)
- Shared utility function across both choice types

**Use Case:** Testing whether decision makers exhibit different levels of sensitivity when probabilities are known (risky) vs. derived from features (uncertain).

## Model m_3 Specification

The proportional-sensitivity model (`models/m_3.stan`) introduces a proportional relationship between sensitivities:

- `alpha`: Sensitivity for uncertain choices
- `kappa`: Association parameter (ω = κα)
- `omega`: Sensitivity for risky choices (derived, not free)

When κ = 1, m_3 reduces to m_1 (shared α). When κ ≠ 1, risky sensitivity differs proportionally from uncertain sensitivity.

## Study Design Tools

The `utils/study_design.py` module provides comprehensive tools for creating experimental designs:

```python
# Generate from configuration file
design = StudyDesign.from_config("configs/my_config.json")

# Save with metadata and visualizations
design.save(
    "results/designs/my_design.json",
    include_metadata=True,
    include_plots=True
)

# Load existing design
loaded = StudyDesign.load("results/designs/my_design.json")
loaded.analyze()
```

**Features:**
- Flexible feature generation (normal, uniform distributions)
- Configurable problem complexity
- Comprehensive metadata and diagnostics
- Automatic visualization generation

See [utils/README.md](utils/README.md) for complete documentation.

## Applications

### Temperature Study

Investigate how LLM sampling temperature affects estimated sensitivity (α) to expected utility maximization.

**Research Question**: How does LLM temperature affect the rationality parameter α?

**Key Features:**
- Controlled experiment across multiple temperature levels (0.0, 0.3, 0.7, 1.0, 1.5)
- Position counterbalancing and transparent NA handling
- Deliberative embeddings

See [applications/temperature_study/README.md](applications/temperature_study/README.md) for the full experimental design.

### Temperature Study with EU Prompt

Extend the temperature study by adding an explicit expected utility maximization instruction to the choice prompts. This tests whether explicitly framing decisions in EU terms changes estimated sensitivity, interacting with temperature effects.

**Research Question**: Does adding an explicit EU-maximization instruction to the prompt change estimated α, and does this interact with temperature?

**Key Features:**
- Reuses problems, assessments, and embeddings from the base temperature study
- Only collects new choices under the EU-framing prompt — minimal additional API cost
- Fits the m_0 model separately at each temperature for comparison with the base study

See [applications/temperature_study_with_eu_prompt/](applications/temperature_study_with_eu_prompt/) for module details.

### Temperature Study with Risky Alternatives

Extend the temperature study by collecting risky choice data — decisions among alternatives with explicit probability distributions over consequences. The risky data merges with the existing uncertain data to produce augmented Stan data packages for models m_1, m_2, and m_3.

**Research Question**: Does LLM sensitivity to expected utility differ between risky choices (known probabilities) and uncertain choices (belief-derived probabilities), and how does temperature modulate each?

**Key Features:**
- 30 hand-crafted risky alternatives with K = 3 probability simplexes
- 100 risky problems (2–4 alternatives each) with position counterbalancing
- Merges with existing uncertain data — no re-collection needed
- Produces augmented Stan data for m_1 (shared α), m_2 (separate α and ω), and m_3 (proportional κ·α)

See [applications/temperature_study_with_risky_alts/README.md](applications/temperature_study_with_risky_alts/README.md) for the full design and CLI reference.

### Ellsberg Study

Study how temperature affects SEU sensitivity using Ellsberg-style urn gambles (K=4) with Claude 3.5 Sonnet. Tests whether the GPT-4o temperature–α relationship generalizes to a different LLM and task domain.

**Research Question**: Does the temperature–sensitivity relationship observed with GPT-4o on insurance triage replicate with Claude on Ellsberg gambles?

**Key Features:**
- Claude 3.5 Sonnet with temperature range [0.0, 0.2, 0.5, 0.8, 1.0]
- K=4 consequences (Ellsberg urn gambles)
- Uses m_02 model variant with Lognormal(3.5, 0.75) prior on α

### Claude Insurance Study

Part of the 2×2 factorial design (Claude × Insurance cell). Tests Claude 3.5 Sonnet on the same insurance claims triage task as the initial temperature study, isolating the LLM effect.

**Research Question**: Does Claude exhibit a temperature–sensitivity relationship on insurance triage?

**Key Features:**
- Claude 3.5 Sonnet on insurance claims triage (K=3)
- Same m_01 model and prior as initial temperature study
- Enables LLM × task factorial comparison

### GPT-4o Ellsberg Study

Part of the 2×2 factorial design (GPT-4o × Ellsberg cell). Tests GPT-4o on Ellsberg urn gambles, isolating the task effect.

**Research Question**: Does GPT-4o's temperature–sensitivity relationship generalize to the Ellsberg task domain?

**Key Features:**
- GPT-4o on Ellsberg urn gambles (K=4)
- Uses m_02 model variant with Lognormal(3.5, 0.75) prior
- Completes the 2×2 factorial design with the other three cells

### Factorial Synthesis

Synthesizes the full 2×2 factorial design crossing LLM family and task domain.

**Research Question**: Are temperature effects on estimated SEU sensitivity primarily driven by LLM, task, or their interaction?

**Key Features:**
- Integrates all four factorial cells
- Formal cross-cell comparison of temperature slopes and monotonicity
- Final synthesis of LLM-specific versus task-specific effects

### Alignment Study

A 6-model × 3-prompt factorial study measuring how LLM identity and prompt framing affect SEU sensitivity. Uses the hierarchical model h_m01 to estimate regression coefficients on log(α) across experimental cells.

**Research Question**: Do different LLMs and prompt framings produce systematically different levels of decision-theoretic rationality?

**Key Features:**
- 18 experimental cells: 6 LLMs (GPT-4o, GPT-4o-mini, o3-mini, Claude Sonnet 4, Claude 3.5 Haiku, Claude 3.7 Sonnet) × 3 prompts (neutral, EU-maximizing, deliberative)
- Hierarchical model with treatment-coded design matrix regressing on log(α)
- Non-centered parameterization for cell-level α
- Support for reasoning models (o3-mini) and extended thinking (Claude 3.7)

## Model h_m01 Specification

The hierarchical model (`models/h_m01.stan`) extends m_01 to pool information across J experimental cells via a regression on log-sensitivity:

- `log(α_j) = γ₀ + X_j·γ + σ_cell·z_j` (non-centered parameterization)
- Cell-specific `beta[j]` for feature-to-probability mapping
- Shared `delta` (utility increments) across all cells
- Stacked data with `cell[m]` membership vector

**Parameters:**
- `gamma0`: Intercept of log-α regression
- `gamma[P]`: Regression coefficients (treatment effects on log-α)
- `sigma_cell`: Cell-level residual SD
- `alpha[J]`: Cell-specific sensitivity (derived)
- `beta[J]`: Cell-specific feature weights (K × D per cell)
- `delta`: Shared utility increments ((K-1)-simplex)

## Configuration Files

Study designs can be specified via JSON configuration files:

```json
{
  "M": 30,
  "K": 4,
  "D": 3,
  "R": 15,
  "min_alts_per_problem": 2,
  "max_alts_per_problem": 6,
  "feature_dist": "uniform",
  "feature_params": {
    "low": -2,
    "high": 2
  },
  "design_name": "uniform_large_study"
}
```

## Output and Results

All results are organized in timestamped subdirectories:

```
results/
├── designs/              # Study designs
│   ├── my_study.json
│   └── my_study_plots/  # Visualizations
└── run_YYYYMMDD_HHMMSS/ # Benchmark runs
    ├── raw_choices.json
    ├── embeddings.npz
    ├── stan_data_*.json
    └── run_metadata.json
```

## Advanced Usage

### Custom Feature Distributions

```python
design = StudyDesign(
    feature_dist="uniform",
    feature_params={"low": -2, "high": 2}
)
```

### Model Diagnostics

```python
# Check convergence
print(fit.diagnose())

# Extract posterior samples
alpha_samples = fit.stan_variable("alpha")
beta_samples = fit.stan_variable("beta")

# Posterior predictive checks
y_pred = fit.stan_variable("y_pred")
```

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

### AI Tools

This project has been developed with significant assistance from AI tools, which have contributed to code development, documentation, mathematical derivations, and research design:

- **Claude Opus 4.5** (Anthropic) — Primary AI assistant for complex reasoning, mathematical formulations, and code architecture
- **Claude Sonnet 4.5** (Anthropic) — Used for code implementation, debugging, and documentation
- **GitHub Copilot** — Code completion and suggestions during development

We acknowledge that AI-assisted development is an evolving practice, and we have endeavored to verify AI-generated content for correctness and appropriateness throughout the project.

