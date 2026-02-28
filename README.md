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
- ✅ Calibrated prior variant (m_01) implemented
- ✅ Study design tools functional (m_0 and m_1)
- ✅ Analysis pipeline complete (parameter recovery, SBC, prior/posterior predictive)
- ✅ Quarto-based documentation and reports
- 🔄 Prompt framing study application in progress
- 🔄 Temperature study application in progress
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
│   │   └── 06_sbc_validation.qmd         # Simulation-based calibration
│   ├── applications/       # Applied research reports
│   │   ├── prompt_framing_study/
│   │   └── temperature_study/
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
│   ├── m_1.stan            # Combined model (risky + uncertain, shared α)
│   ├── m_1_sim.stan        # m_1 simulation model
│   ├── m_1_sbc.stan        # m_1 SBC model
│   ├── m_2.stan            # Separate sensitivity model (α for uncertain, ω for risky)
│   ├── m_2_sim.stan        # m_2 simulation model
│   ├── m_2_sbc.stan        # m_2 SBC model
│   ├── m_3.stan            # Proportional sensitivity model (ω = κα)
│   ├── m_3_sim.stan        # m_3 simulation model
│   ├── m_3_sbc.stan        # m_3 SBC model
│   └── README_m1.md        # m_1 implementation guide
├── utils/                   # Core utilities
│   ├── __init__.py         # Shared utilities, model detection
│   ├── study_design.py     # Experimental design generation (m_0)
│   ├── study_design_m1.py  # Extended design for m_1
│   └── README.md           # Utils documentation
├── analysis/                # Analysis scripts
│   ├── model_estimation.py # Model fitting utilities
│   ├── parameter_recovery.py # Parameter recovery analysis
│   ├── posterior_predictive_checks.py # Posterior predictive checks
│   ├── prior_predictive.py # Prior predictive checks
│   ├── sbc.py              # Simulation-based calibration
│   └── sample_size_estimation.py # Sample size planning
├── applications/            # Applied research projects
│   ├── prompt_framing_study/ # Prompt framing effects on LLM rationality
│   ├── temperature_study/  # LLM temperature effects on sensitivity
│   └── llm_rationality/    # Legacy LLM benchmarking (deprecated)
├── scripts/                 # Executable scripts
│   ├── run_study_design.py # Generate study designs
│   ├── run_m1_study_design.py # Generate m_1 study designs
│   ├── run_model_estimation.py # Fit models
│   ├── run_parameter_recovery.py # Run recovery analysis
│   ├── run_prior_predictive.py # Prior predictive analysis
│   ├── run_prior_predictive_grid.py # Prior predictive grid search
│   ├── run_sbc.py          # SBC validation
│   ├── run_sample_size_estimation.py # Sample size analysis
│   ├── run_temperature_analysis.py # Temperature study analysis
│   ├── refit_with_ppc.py   # Refit models with posterior predictive checks
│   ├── copy_figures_for_report.py # Copy figures into reports
│   ├── cleanup_temp_files.py # Clean up temporary files
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
git clone https://github.com/your-org/seu-sensitivity.git
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
git clone https://github.com/your-org/seu-sensitivity.git
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

### 4. Benchmark LLM Rationality

```bash
cd applications/prompt_framing_study

# Run the full study pipeline
python -m prompt_framing_study
```

See [applications/prompt_framing_study/README.md](applications/prompt_framing_study/README.md) for detailed workflow on investigating how prompt framing affects LLM rationality.

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

For m_1, m_2, and m_3 models, use the corresponding `m1_*`, `m2_*`, and `m3_*` config files.

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

### Prompt Framing Study

Investigate how prompt framing (rationality emphasis) affects an LLM's sensitivity to expected utility maximization.

**Research Question**: Does explicitly framing a decision problem in terms of utility maximization change how "rational" an LLM appears to be?

**Key Features:**
- Contextualized embeddings that capture prompt-specific claim perception
- Multiple prompt variants from minimal to maximal rationality emphasis
- Robustness analysis across embedding models and dimensions

See [applications/prompt_framing_study/README.md](applications/prompt_framing_study/README.md) for complete workflow.

### Temperature Study

Investigate how LLM sampling temperature affects estimated sensitivity (α) to expected utility maximization.

**Research Question**: How does LLM temperature affect the rationality parameter α?

**Key Features:**
- Controlled experiment across multiple temperature levels (0.0, 0.3, 0.7, 1.0, 1.5)
- Position counterbalancing and transparent NA handling
- Deliberative embeddings

See [applications/temperature_study/README.md](applications/temperature_study/README.md) for the full experimental design.

### Legacy: LLM Rationality Benchmarking (Deprecated)

The original `llm_rationality` module provides basic LLM benchmarking capabilities. This module is being superseded by `prompt_framing_study` and `temperature_study` which offer improved methodology.

See [applications/llm_rationality/README.md](applications/llm_rationality/README.md) for legacy documentation.

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

