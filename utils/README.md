# Utils Module for SEU Sensitivity Analysis

This directory contains utility modules for the SEU sensitivity project, including tools for experimental design, data processing, and analysis.

## Module Overview

| Module | Description |
|--------|-------------|
| `__init__.py` | Shared utilities: model detection, default parameters |
| `study_design.py` | Base study design for m_0 (uncertain choice only) |
| `study_design_m1.py` | Extended design for m_1 (risky + uncertain choice) |

## Shared Utilities (`__init__.py`)

The `__init__.py` module provides common utilities used across analysis scripts:

### Default Parameter Generation

```python
from utils import DEFAULT_PARAM_GENERATION

# Returns: {'alpha_mean': 0.0, 'alpha_sd': 1.0, 'beta_sd': 1.0}
print(DEFAULT_PARAM_GENERATION)
```

### Model Detection

```python
from utils import is_m1_model

# Check if a data dict is for m_1 model
data = {'M': 20, 'N': 20, 'K': 3, ...}
if is_m1_model(data):
    print("This is an m_1 model with risky choices")

# Also works with StudyDesign objects
from utils.study_design_m1 import StudyDesignM1
design = StudyDesignM1(M=20, N=20, K=3, D=2, R=10, S=8)
is_m1_model(design)  # Returns True
```

### Study Design Class Selection

```python
from utils import get_study_design_class

config = {
    'study_design_config': {
        'M': 20, 'N': 20, 'K': 3, 'D': 2, 'R': 10, 'S': 8
    }
}
DesignClass = get_study_design_class(config)  # Returns StudyDesignM1
design = DesignClass(**config['study_design_config'])
```

## Study Design (`study_design.py`)

The `study_design.py` module provides tools for generating and managing study designs for experiments and simulations in the SEU sensitivity framework.

### Key Features

- Generate feature vectors for alternatives
- Assign alternatives to decision problems
- Analyze design properties
- Visualize design characteristics
- Save/load designs from JSON

### Usage

#### Programmatic Use

```python
from utils.study_design import StudyDesign

# Create a design with default parameters
design = StudyDesign(M=20, K=3, D=2, R=10)
design.generate()
design.analyze()
design.save("results/designs/my_design.json")

# Create from configuration file
config_design = StudyDesign.from_config("configs/my_config.json")
config_design.analyze()
config_design.save("results/designs/config_design.json")

# Load existing design
loaded_design = StudyDesign.load("results/designs/existing_design.json")
loaded_design.analyze()
```

#### Command Line Use

```bash
# Create a study design using default parameters
python utils/study_design.py --output results/designs/default_design.json --seed 42

# Create a study design using a configuration file
python utils/study_design.py --config configs/study_design_config.json --output results/designs/custom_design.json --seed 42
```

## Study Design M1 (`study_design_m1.py`)

The `study_design_m1.py` module extends the base `StudyDesign` class for the m_1 model, which combines risky and uncertain choice problems.

### Key Features

- All base `StudyDesign` features for uncertain choice problems
- Additional risky choice problem generation (N problems, S alternatives)
- Flexible risky probability generation methods: `uniform`, `fixed`, `random`
- Combined data dictionary generation for m_1 Stan models

### Usage

```python
from utils.study_design_m1 import StudyDesignM1

# Create design with both uncertain (M) and risky (N) problems
design = StudyDesignM1(
    M=20,              # Number of uncertain choice problems
    N=20,              # Number of risky choice problems
    K=3,               # Number of consequences
    D=2,               # Feature dimensions
    R=10,              # Uncertain alternatives
    S=8,               # Risky alternatives
    risky_probs='fixed'  # Probability generation method
)
design.generate()
design.save("results/designs/m1_design.json")

# Get data dict for Stan
data = design.get_data_dict()
# Includes: M, N, K, D, R, S, w, I, x, J
```

### Risky Probability Methods

| Method | Description |
|--------|-------------|
| `uniform` | All consequences equally likely: [1/K, 1/K, ..., 1/K] |
| `fixed` | Predefined probability values for good coverage |
| `random` | Random simplexes from Dirichlet(1, 1, ..., 1) |

#### Command Line Use

```bash
# Using configuration file
python scripts/run_m1_study_design.py --config configs/m1_study_config.json

# Using command-line arguments
python scripts/run_m1_study_design.py \
    --M 30 --N 30 --K 3 --D 2 --R 15 --S 12 \
    --risky-probs fixed \
    --output my_m1_design.json
```

## Configuration Parameters

Study designs can be configured via JSON files. See `configs/` directory for examples:

- `study_config.json` - Base m_0 study design
- `m1_study_config.json` - m_1 study design with risky choices