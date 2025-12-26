# SEU Sensitivity

![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

**Note**: This is an active research project. Code and documentation are evolving.

A Bayesian framework for modeling and analyzing decision-making behavior through the lens of Subjective Expected Utility (SEU) theory. This project provides tools for generating experimental designs, fitting computational models, and assessing the rationality of decision makers (including Large Language Models).

## Development Status

🚧 **This project is currently under active development** 🚧

Current status:
- ✅ Core theoretical framework established
- ✅ Base Stan model (m_0) implemented and tested
- ✅ Combined model (m_1) with risky and uncertain choice - implemented and tested
- ✅ Study design tools functional
- 🔄 LLM benchmarking application in progress
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
├── theory/                  # Theoretical foundations and proofs
│   └── m_0_theory.md       # Mathematical framework for the base model
├── models/                  # Stan model implementations
│   ├── m_0.stan            # Base SEU sensitivity model (uncertain choice only)
│   ├── m_1.stan            # Combined model (risky + uncertain choice)
│   └── README_m1.md        # m_1 implementation guide
├── utils/                   # Core utilities
│   ├── study_design.py     # Experimental design generation (m_0)
│   ├── study_design_m1.py  # Extended design for m_1 (risky + uncertain)
│   └── README.md           # Utils documentation
├── applications/            # Applied research projects
│   └── llm_rationality/    # LLM rationality benchmarking
│       ├── claim_design.py
│       ├── claim_embedding.py
│       ├── llm_client.py
│       ├── run_benchmark.py
│       └── README.md
├── configs/                 # Configuration files for studies
├── results/                 # Generated results and outputs
├── environment.yml          # Conda environment specification
├── requirements.txt         # Pip requirements (alternative)
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended) or pip
- Stan (will be installed via CmdStanPy)

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

```bash
# Set up OpenAI API key
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

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
cd applications/llm_rationality

# Generate decision problems
python generate_property_problems.py

# Run benchmark
python run_benchmark.py
```

See [applications/llm_rationality/README.md](applications/llm_rationality/README.md) for detailed LLM benchmarking workflow.

## Theoretical Background

Overview of three fundamental properties of SEU sensitivity:

1. **Monotonicity**: Higher sensitivity increases probability of choosing value-maximizing alternatives
2. **Perfect Rationality Limit**: As α → ∞, agents deterministically choose optimal alternatives
3. **Random Choice Limit**: As α → 0, agents choose uniformly at random

These properties hold for any value function, with SEU providing the substantive behavioral interpretation.

**Key Result**: With utilities normalized to [0,1], the sensitivity parameter α has a precise interpretation as the log-odds change per unit of standardized SEU difference.

**SEU Maximizer Selection**: The prior predictive analysis tracks the probability of selecting SEU-maximizing alternatives for each problem, providing a diagnostic for model rationality under the prior.

See [theory/m_0_theory.md](theory/m_0_theory.md) for complete mathematical details.

## Model m_0 Specification

The base model (`models/m_0.stan`) implements:

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

### LLM Rationality Benchmarking

Assess how "rational" Large Language Models are by measuring their sensitivity to expected utility maximization in text-based decision problems.

**Workflow:**
1. Create base claim descriptions
2. Generate decision problems
3. Collect LLM choices
4. Generate embeddings for alternatives
5. Fit m_0 model to estimate rationality parameters
6. Compare across models

**Example Results:**
```
GPT-4:         α = 2.8 (highly sensitive to SEU)
GPT-3.5-Turbo: α = 1.5 (moderately sensitive)
```

See [applications/llm_rationality/README.md](applications/llm_rationality/README.md) for complete workflow.

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

