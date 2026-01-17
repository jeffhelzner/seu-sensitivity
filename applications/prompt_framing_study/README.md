# Prompt Framing Study

A module for studying how prompt framing affects LLM rationality in SEU-based decision tasks.

## Overview

This module implements an experimental framework to investigate whether explicit rationality
cues in prompts influence an LLM's sensitivity parameter (α) when making decisions that can
be analyzed using Subjective Expected Utility (SEU) theory. The key innovation is the use of
**contextualized embeddings**: the same claim receives different embeddings under different
prompt framings, capturing how context affects the LLM's internal representation.

## Research Design

### Core Hypothesis

When prompted with explicit rationality cues (e.g., "choose the option that maximizes expected
value"), LLMs will exhibit higher sensitivity (α) to SEU-optimal choices compared to minimal
prompts. This is measured by fitting the m_0 model to choice data collected under different
prompt conditions.

### Prompt Variants

The study uses four levels of rationality emphasis:

| Level | Name | Description |
|-------|------|-------------|
| 0 | Minimal | Simple task instruction, no rationality cues |
| 1 | Baseline | Brief mention of considering consequences |
| 2 | Enhanced | Explicit expected value reasoning |
| 3 | Maximal | Detailed decision-theoretic framework |

### Contextualized Embeddings

Unlike traditional approaches where claim embeddings are fixed, this module computes embeddings
within the context of each prompt variant. This captures the hypothesis that the same claim
"means" something different when presented with rationality framing versus minimal framing.

### Decision Structure

- **Domain**: Insurance claims triage
- **Alternatives**: Recommendations for claim handling
- **Consequences (K=3)**: 
  - Both parties agree with the recommendation
  - One party agrees, one disagrees  
  - Neither party agrees

## Installation

The module is part of the seu-sensitivity project. To install dependencies:

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For full functionality:
```bash
pip install openai anthropic  # LLM API clients
pip install cmdstanpy         # Stan model fitting
pip install matplotlib seaborn  # Visualization
```

## Configuration

### Main Configuration (`configs/study_config.yaml`)

```yaml
# Study parameters
num_problems: 200
K: 3  # Number of consequence dimensions
min_alternatives: 2
max_alternatives: 4
temperature: 0.7
num_repetitions: 1

# Embedding configuration
target_dim: 32
reduction_method: "pca"  # or "random"

# API configuration
provider: "openai"  # or "anthropic"
model: "gpt-4o"
embedding_model: "text-embedding-3-small"

# Output
output_dir: "results/prompt_framing"
checkpoint_interval: 50
```

### Prompt Variants (`configs/prompt_variants.yaml`)

Define custom prompt variants with system prompts and choice templates.

### Claims Data (`data/claims.json`)

```json
{
  "claims": [
    {
      "id": "C001",
      "description": "Auto collision claim for $5,000...",
      "complexity": "medium"
    }
  ]
}
```

## Usage

### Command Line Interface

```bash
# Validate configuration
python -m prompt_framing_study.cli validate

# Estimate costs before running
python -m prompt_framing_study.cli estimate

# Run full study
python -m prompt_framing_study.cli run

# Resume interrupted study
python -m prompt_framing_study.cli resume --checkpoint latest

# Generate visualizations
python -m prompt_framing_study.cli visualize
```

### Python API

```python
from prompt_framing_study import (
    StudyRunner,
    PromptManager,
    ContextualizedEmbeddingManager,
    ProblemGenerator,
    ChoiceCollector
)

# Initialize components
prompt_manager = PromptManager("configs/prompt_variants.yaml")
embedding_manager = ContextualizedEmbeddingManager(
    api_key="your-api-key",
    target_dim=32,
    reduction_method="pca"
)

# Run full pipeline
runner = StudyRunner(config_path="configs/study_config.yaml")
results = runner.run()

# Access results
for variant_name, stan_data in results.items():
    print(f"{variant_name}: {stan_data['M']} problems")
```

### Step-by-Step Usage

```python
import json
from prompt_framing_study import (
    PromptManager,
    ContextualizedEmbeddingManager,
    ProblemGenerator,
    ChoiceCollector
)

# 1. Load claims
with open("data/claims.json") as f:
    claims_data = json.load(f)
claims = claims_data["claims"]

# 2. Initialize components
prompt_manager = PromptManager("configs/prompt_variants.yaml")
embedding_manager = ContextualizedEmbeddingManager(
    api_key="your-api-key",
    target_dim=32
)

# 3. Generate contextualized embeddings for each variant
for variant in prompt_manager.variants:
    embeddings = embedding_manager.compute_embeddings(
        claims=claims,
        prompt_variant=variant,
        prompt_manager=prompt_manager
    )
    print(f"{variant.name}: {len(embeddings)} embeddings computed")

# 4. Generate problems
problem_generator = ProblemGenerator(
    claims=claims,
    K=3,
    min_alternatives=2,
    max_alternatives=4
)
problems = problem_generator.generate(num_problems=100)

# 5. Collect choices for each variant
choice_collector = ChoiceCollector(
    api_key="your-api-key",
    model="gpt-4o"
)

for variant in prompt_manager.variants:
    choices = choice_collector.collect(
        problems=problems,
        prompt_variant=variant,
        prompt_manager=prompt_manager
    )
```

## Output Structure

```
results/prompt_framing/
├── checkpoint_0050.json      # Intermediate checkpoint
├── checkpoint_0100.json
├── embeddings/
│   ├── minimal_embeddings.npy
│   ├── baseline_embeddings.npy
│   ├── enhanced_embeddings.npy
│   └── maximal_embeddings.npy
├── choices/
│   ├── minimal_choices.json
│   ├── baseline_choices.json
│   ├── enhanced_choices.json
│   └── maximal_choices.json
├── stan_data/
│   ├── minimal_stan_data.json
│   ├── baseline_stan_data.json
│   ├── enhanced_stan_data.json
│   └── maximal_stan_data.json
├── analysis/
│   ├── alpha_estimates.json
│   └── robustness_results.json
└── figures/
    ├── alpha_comparison.png
    ├── posterior_distributions.png
    └── embedding_space.png
```

## Analysis with m_0 Model

The collected data is formatted for the project's m_0 Stan model:

```python
from cmdstanpy import CmdStanModel
import json

# Load Stan data
with open("results/prompt_framing/stan_data/minimal_stan_data.json") as f:
    stan_data = json.load(f)

# Fit model
model = CmdStanModel(stan_file="models/m_0.stan")
fit = model.sample(data=stan_data)

# Extract alpha estimates
alpha_samples = fit.stan_variable("alpha")
print(f"α posterior mean: {alpha_samples.mean():.3f}")
print(f"α 95% CI: [{np.percentile(alpha_samples, 2.5):.3f}, {np.percentile(alpha_samples, 97.5):.3f}]")
```

## Robustness Analysis

The module includes tools for testing robustness:

```python
from prompt_framing_study import RobustnessAnalyzer

analyzer = RobustnessAnalyzer(results_dir="results/prompt_framing")

# Test sensitivity to PCA dimensions
pca_results = analyzer.analyze_pca_sensitivity(
    dims=[16, 32, 64, 128],
    claims=claims,
    prompt_manager=prompt_manager
)

# Compare embedding models
model_results = analyzer.compare_embedding_models(
    models=["text-embedding-3-small", "text-embedding-3-large"],
    claims=claims,
    prompt_manager=prompt_manager
)
```

## Cost Estimation

Before running a full study, estimate API costs:

```python
from prompt_framing_study import CostEstimator

estimator = CostEstimator()
estimate = estimator.estimate_study_cost(
    num_claims=20,
    num_problems=200,
    num_variants=4,
    embedding_model="text-embedding-3-small",
    chat_model="gpt-4o"
)

print(f"Estimated total cost: ${estimate['total']:.2f}")
print(f"  Embeddings: ${estimate['embeddings']:.2f}")
print(f"  Chat completions: ${estimate['completions']:.2f}")
```

## Validation

The module includes comprehensive validation:

```python
from prompt_framing_study import (
    validate_config,
    validate_claims_file,
    validate_stan_data
)

# Validate configuration
warnings = validate_config(config)
for warning in warnings:
    print(f"Warning: {warning}")

# Validate claims file
claims_data = validate_claims_file("data/claims.json")

# Validate generated Stan data
validate_stan_data(stan_data, model="m_0")
```

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest applications/prompt_framing_study/tests/

# Run with coverage
pytest applications/prompt_framing_study/tests/ --cov=prompt_framing_study

# Run specific test file
pytest applications/prompt_framing_study/tests/test_validation.py -v
```

## Module Structure

```
prompt_framing_study/
├── __init__.py              # Public API exports
├── llm_client.py            # LLM API clients (OpenAI, Anthropic)
├── prompt_manager.py        # Prompt variant management
├── contextualized_embedding.py  # Context-aware embeddings
├── problem_generator.py     # Decision problem generation
├── choice_collector.py      # LLM choice collection
├── cost_estimator.py        # API cost estimation
├── validation.py            # Input/output validation
├── study_runner.py          # Main pipeline orchestrator
├── robustness_analysis.py   # Sensitivity analysis
├── visualization.py         # Plotting utilities
├── cli.py                   # Command-line interface
├── configs/
│   ├── study_config.yaml
│   ├── prompt_variants.yaml
│   └── embedding_config.yaml
├── data/
│   └── claims.json
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_pipeline.py
    └── test_validation.py
```

## References

This module implements methods described in:

- The SEU sensitivity framework documented in the project's foundations reports
- The m_0 Stan model specification in `models/README_m1.md`

## License

See project LICENSE file.
