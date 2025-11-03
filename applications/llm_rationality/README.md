# LLM Rationality Benchmark

This application assesses the "rationality" of Large Language Models (LLMs) by measuring how consistently they maximize expected utility when making decisions across text-based problems.

## Overview

The benchmark framework:
1. Creates base descriptions of decision alternatives (e.g., insurance claims)
2. Generates decision problems by sampling from these descriptions
3. Collects LLM choices for each problem 
4. Converts alternatives to vector representations using embeddings
5. Prepares data for the m_0 model to estimate rationality parameters
6. Enables analysis and visualization of results

## Directory Structure

```
llm_rationality/
├── claim_design.py          # Creates decision problems from base claims
├── claim_embedding.py       # Generates embeddings for base claims
├── generate_property_problems.py  # Generates property claim problems
├── llm_client.py            # Interfaces with LLM APIs
├── run_benchmark.py         # Main script for running benchmarks
├── visualization.py         # Creates plots and visualizations
├── data/
│   └── property_claims.json # Base claim descriptions
├── problems/
│   └── property_problems.json # Generated decision problems
└── results/
    └── run_YYYYMMDD_HHMMSS/ # Run-specific results
        ├── embeddings.npz   # Claim embeddings
        ├── raw_choices.json # Raw choices from all models
        ├── run_metadata.json # Run metadata
        └── stan_data_*.json # Stan model data packages
```

## Installation

```bash
# Create a conda environment
conda create -n llm-rationality python=3.8
conda activate llm-rationality

# Install required packages
conda install -c conda-forge scikit-learn numpy
pip install openai python-dotenv

# Set your OpenAI API key
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Workflow

### 1. Create Base Claims

Create or edit `data/property_claims.json`:

```json
{
  "context": "Your job is to provide triage to incoming property claims...",
  "consequences": ["bad", "neutral", "good"],
  "claims": [
    {
      "id": "PC001",
      "description": "Residential house fire reported 3 hours ago..."
    },
    {
      "id": "PC002",
      "description": "Water damage in commercial retail space reported yesterday..."
    }
  ]
}
```

The `context` field provides the decision-making scenario presented to LLMs.
The `consequences` field defines the K outcomes for the m_0 model.

### 2. Generate Decision Problems

Run the problem generator:

```bash
python generate_property_problems.py
```

This creates `problems/property_problems.json` containing decision problems that combine base claims in different configurations.

You can customize problem generation:

```python
from claim_design import ClaimDesignGenerator

generator = ClaimDesignGenerator("data/property_claims.json")
problems = generator.generate_problems(
    num_problems=20,     # Generate 20 problems
    min_alts=2,          # Minimum 2 alternatives per problem
    max_alts=5,          # Maximum 5 alternatives per problem
    seed=42              # For reproducibility
)
generator.save_problems(problems, "problems/property_problems.json")
```

### 3. Run the Benchmark

Execute the benchmark:

```bash
python run_benchmark.py
```

This:
1. Presents each decision problem to multiple LLMs
2. Collects their choices
3. Generates embeddings for base claims (with dimension reduction)
4. Creates Stan data packages for analyzing rationality

Results are saved in a timestamped subfolder under `results/`.

You can customize the LLM models:

```python
# In run_benchmark.py
llm_clients = {
    "GPT-4": OpenAIClient(model="gpt-4", temperature=0.0),
    "GPT-3.5-Turbo": OpenAIClient(model="gpt-3.5-turbo", temperature=0.0),
    "Claude-3": ClaudeClient(model="claude-3-opus", temperature=0.0)  # If implemented
}
```

### 4. Analyze Results

Use the generated Stan data packages with the m_0 model:

```bash
# Using CmdStanPy
from cmdstanpy import CmdStanModel
import json

# Load Stan data for a specific model
with open("results/run_20251021_165854/stan_data_GPT-4.json", "r") as f:
    stan_data = json.load(f)

# Fit the m_0 model
model = CmdStanModel(stan_file="../../../models/m_0.stan")
fit = model.sample(data=stan_data)

# Extract rationality parameter
alpha = fit.stan_variable("alpha")
print(f"Rationality parameter (alpha): {alpha.mean()}")
```

### 5. Visualize Results

Create visualizations comparing different models:

```python
from visualization import plot_rationality_comparison

# Assuming model results are loaded in a dictionary
plot_rationality_comparison(results)
```

## Key Files

- **claim_design.py**: Creates decision problems from base descriptions
- **claim_embedding.py**: Generates embeddings for base claims
- **llm_client.py**: Interfaces with different LLM APIs
- **run_benchmark.py**: Main script for running benchmarks
- **visualization.py**: Creates plots and visualizations of results

## Output Files

Each benchmark run creates a timestamped folder with:

- **raw_choices.json**: Detailed record of all LLM choices
- **embeddings.npz**: Claim embeddings (numpy compressed format)
- **stan_data_[MODEL].json**: Data formatted for Stan model fitting
- **run_metadata.json**: Information about the benchmark run

## Extensions

### Adding New LLM Providers

Extend the `LLMClient` class in `llm_client.py`:

```python
class AnthropicClient(LLMClient):
    def __init__(self, model="claude-3-opus", temperature=0.0):
        super().__init__(model)
        self.temperature = temperature
        # Initialize Anthropic client
        
    def generate(self, prompt):
        # Implement generation using Anthropic's API
        pass
```

### Custom Decision Problems

Create new types of decision problems by:
1. Creating a new base claims file (e.g., `medical_diagnoses.json`)
2. Adapting the problem generation script
3. Running the benchmark with your new problems

## References

- [SEU Sensitivity Framework](https://github.com/jeffhelzner/seu-sensitivity)
- [m_0 Stan Model Documentation](../../models/m_0.stan)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)