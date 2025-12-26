# Code Review and Cleanup Instructions for SEU-Sensitivity Project

## Overview

The m_1 model implementation is complete and tested. This review focuses on:
1. Removing outdated/redundant documentation
2. Consolidating duplicate code patterns
3. Cleaning up unused files
4. Ensuring consistent naming and documentation

**CONSTRAINT:** Do NOT modify any `.stan` files.

---

## Phase 1: Documentation Cleanup

### 1.1 Remove Outdated Status Document

**File to DELETE:** `M1_IMPLEMENTATION_STATUS.md`

**Rationale:** This was a temporary status tracker created during m_1 implementation. Its content is now outdated (references "compilation issue pending" which is resolved) and duplicates information in `models/README_m1.md`. The README_m1.md file should be the single source of truth for m_1 documentation.

### 1.2 Consolidate m_1 Documentation

**File to UPDATE:** `models/README_m1.md`

- Update the date from "December 6, 2025" to current
- Remove references to "compilation issues" since they're resolved
- Add a "Tested & Working" status badge/note at the top
- Ensure all command examples use the correct conda environment activation

### 1.3 Update Main README.md

**File to UPDATE:** `README.md`

- Add m_1 model to the "Development Status" section as completed
- Add m_1 to the "Project Structure" section
- Add a brief mention of m_1 in the "Quick Start" or create a new section for combined risky/uncertain choice
- Update the model listing to include m_1 models

---

## Phase 2: Configuration File Cleanup

### 2.1 Remove Duplicate/Test Config Files

**Files to REVIEW for deletion:**
- `configs/prior_analysis_config_1.json` - Appears to be a test/duplicate of `prior_analysis_config.json`
  - Check if it's referenced anywhere before deleting
  - If not referenced, DELETE

### 2.2 Verify Config File Completeness

Ensure each model (m_0, m_01, m_1) has a consistent set of configs. Currently:

**m_0 configs (base):**
- `study_config.json`
- `prior_analysis_config.json`
- `parameter_recovery_config.json`
- `sbc_config.json`
- `model_estimation_config.json`
- `sample_size_config.json`

**m_1 configs:**
- `m1_study_config.json` ✓
- `m1_prior_analysis_config.json` ✓
- `m1_parameter_recovery_config.json` ✓
- `m1_sbc_config.json` ✓
- (Missing: `m1_model_estimation_config.json` - verify if referenced in M1_IMPLEMENTATION_STATUS.md exists)

**Action:** Verify m1_model_estimation_config.json exists or remove references to it.

---

## Phase 3: Test/Utility File Cleanup

### 3.1 Remove Ad-hoc Test File

**File to DELETE or MOVE:** `test_compile.py` (root directory)

**Rationale:** This is a one-off test script. Either:
- DELETE it (functionality covered by `scripts/test_m1_model.py`)
- OR MOVE to `scripts/` directory and rename to `test_stan_compilation.py`

### 3.2 Review `__pycache__` Directories

**Action:** Add `**/__pycache__/` to `.gitignore` if not already present. These should not be tracked.

---

## Phase 4: Code DRY Improvements

### 4.1 Consolidate Study Design Import Logic

**Files affected:**
- `scripts/run_parameter_recovery.py`
- `scripts/run_sbc.py`
- `scripts/run_prior_predictive.py`
- `scripts/run_model_estimation.py`

**Issue:** Each script has similar logic for detecting m_0 vs m_1 models and importing appropriate StudyDesign class.

**Refactoring suggestion:** Create a utility function in `utils/__init__.py`:

```python
def get_study_design_class(config):
    """
    Determine appropriate StudyDesign class based on config.
    
    Returns StudyDesignM1 if config has both M and N parameters,
    otherwise returns base StudyDesign.
    """
    study_config = config.get('study_design_config', {})
    if 'N' in study_config and study_config.get('N', 0) > 0:
        from utils.study_design_m1 import StudyDesignM1
        return StudyDesignM1
    else:
        from utils.study_design import StudyDesign
        return StudyDesign
```

### 4.2 Consolidate Model Detection Logic

**Files affected:**
- `analysis/parameter_recovery.py`
- `analysis/sbc.py`

**Issue:** Both files have similar `_is_m1_model()` detection logic.

**Refactoring suggestion:** Move to a shared utility:

```python
# utils/__init__.py
def is_m1_model(study_design):
    """Check if study design is for m_1 model (has risky choice data)."""
    return hasattr(study_design, 'N') and study_design.N is not None and study_design.N > 0
```

### 4.3 Consolidate Parameter Generation Defaults

**Issue:** Default values for `alpha_mean`, `alpha_sd`, `beta_sd` are duplicated in:
- `analysis/parameter_recovery.py`
- `analysis/sbc.py`
- Various config files

**Refactoring suggestion:** Define defaults in one place:

```python
# utils/__init__.py
DEFAULT_PARAM_GENERATION = {
    'alpha_mean': 5.0,
    'alpha_sd': 2.0,
    'beta_sd': 1.0
}
```

---

## Phase 5: Unused/Obsolete File Review

### 5.1 Review m_01 Model Status

**Files to REVIEW:**
- `models/m_01.stan`
- `models/m_01_sim.stan`
- `models/m_01_sbc.stan`
- `scripts/copy_figures_for_report.py` (has m_01 specific functions)

**Questions to answer:**
1. Is m_01 an intermediate model that's still needed?
2. Is it documented anywhere?
3. Should it be kept, deprecated, or removed?

**Action:** If m_01 is a valid model variant, add documentation. If it's obsolete (superseded by m_1), consider deprecation notice or removal.

### 5.2 Review Generated Files in models/

**Files without extensions in `models/`:**
- `m_0`, `m_01`, `m_0_sbc`, `m_0_sim`, `m_01_sim`, `m_1`, `m_1_sbc`, `m_1_sim`

These are compiled Stan executables. 

**Action:** Add patterns to `.gitignore` to exclude compiled binaries but keep `.stan` source files:
```
# Compiled Stan models
models/m_0
models/m_01
models/m_0_sbc
models/m_0_sim
models/m_01_sim
models/m_1
models/m_1_sbc
models/m_1_sim
```

---

## Phase 6: Documentation Consistency

### 6.1 Standardize Header Comments

Ensure all Python files in `scripts/` and `analysis/` have consistent docstring format including:
- Purpose description
- Usage examples
- Parameter documentation

### 6.2 Update utils/README.md

**File to UPDATE:** `utils/README.md`

Add documentation for:
- `study_design_m1.py` - The m_1 study design generator
- Any new utility functions added during DRY refactoring

---

## Phase 7: Verification

After all changes, run:

```bash
# Activate environment
conda activate stan

# Run m_1 validation suite
python scripts/test_m1_model.py

# Run parameter recovery (quick test with few iterations)
python scripts/run_parameter_recovery.py --config configs/m1_parameter_recovery_config.json

# Run SBC (quick test)
python scripts/run_sbc.py --config configs/m1_sbc_config.json
```

All tests should pass without modification to any `.stan` files.

---

## Summary of Actions

| Priority | Action | Files |
|----------|--------|-------|
| HIGH | Delete outdated status doc | `M1_IMPLEMENTATION_STATUS.md` |
| HIGH | Delete/verify unused config | `configs/prior_analysis_config_1.json` |
| MEDIUM | Update .gitignore | `.gitignore` |
| MEDIUM | Move/delete test script | `test_compile.py` |
| MEDIUM | Update README_m1.md | `models/README_m1.md` |
| MEDIUM | Update main README | `README.md` |
| LOW | DRY refactoring | `utils/__init__.py`, analysis scripts |
| LOW | Review m_01 status | m_01 related files |
| LOW | Update utils README | `utils/README.md` |