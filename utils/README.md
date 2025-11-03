# Utils Module for Epistemic Agents

This directory contains utility modules for the Epistemic Agents project, including tools for experimental design, data processing, and analysis.

## Study Design

The `study_design.py` module provides tools for generating and managing study designs for experiments and simulations in the Epistemic Agents framework.

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

##### Configuration Parameters

This needs to be completed.