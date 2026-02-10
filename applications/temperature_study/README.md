# Temperature Study

**How does LLM temperature affect estimated sensitivity (α) to expected utility maximisation?**

This module implements a controlled experiment that fits the `m_0` SEU sensitivity model separately at five temperature levels (0.0, 0.3, 0.7, 1.0, 1.5) and compares the resulting α posteriors. It addresses methodological issues discovered in the earlier prompt-framing pilot — most critically, silent default-to-position-0 for unparseable responses — by introducing position counterbalancing, transparent NA handling, and deliberative embeddings.

See [DESIGN.md](DESIGN.md) for the full experimental design and rationale.

---

## Quick start

```bash
# From the project root (seu-sensitivity/)
conda activate seu-sensitivity

# 1. Validate config and claim pool
python -m applications.temperature_study validate

# 2. Estimate API costs before committing
python -m applications.temperature_study estimate-cost

# 3. Run the full pipeline (collection → PCA → Stan data)
python -m applications.temperature_study run --skip-fitting

# 4. Fit Stan models on the prepared data
python -m applications.temperature_study fit

# 5. Re-run data preparation from saved collection data
python -m applications.temperature_study prepare
```

All subcommands accept `-c PATH` to override the default config at `configs/study_config.yaml`.

### CLI reference

| Command | Description |
|---------|-------------|
| `validate` | Check config validity and claim pool coverage |
| `estimate-cost` | Print estimated API call counts and costs |
| `run` | Run the full pipeline (phases 1–4) |
| `run --skip-collection` | Skip API calls; rebuild from saved data |
| `run --skip-fitting` | Stop after data preparation (no Stan) |
| `fit` | Fit Stan models on existing `stan_data_T*.json` files |
| `prepare` | Re-run phase 3 (PCA + Stan data) only |

Add `-v` for debug-level logging.

### Typical workflows

**Collect → inspect → fit** (recommended for a first run):

```bash
# Collect data and prepare Stan inputs, but don't fit yet
python -m applications.temperature_study run --skip-fitting

# Inspect data quality: check NA rates, position bias, consistency
# (use the analysis modules in a script or notebook — see §Analysis below)

# Once satisfied, fit the Stan models
python -m applications.temperature_study fit
```

This two-step approach lets you review NA rates (§6.4), position bias (§6.2), and choice consistency (§6.3) *before* committing to a potentially long Stan sampling run. If the data reveals problems (e.g., high NA concentration in certain claims), you can adjust the claim pool or config and re-collect without wasting fitting time.

**Re-prepare with different PCA settings:**

```bash
# Change target_dim in study_config.yaml (or pass a custom config)
python -m applications.temperature_study prepare -c my_config.yaml
python -m applications.temperature_study fit -c my_config.yaml
```

`prepare` re-runs only Phase 3 (pooled PCA + Stan data assembly) from the saved collection data — no API calls needed.

**Full end-to-end** (collection + fitting in one shot):

```bash
# Set fit_models: true in study_config.yaml, then:
python -m applications.temperature_study run
```

---

## Architecture

```
temperature_study/
├── config.py                  # StudyConfig dataclass & validation
├── llm_client.py              # LLM/embedding API clients, parse_choice()
├── problem_generator.py       # Problem generation with shuffled presentations
├── deliberation_collector.py  # Per-claim deliberation elicitation & embedding
├── choice_collector.py        # Choice collection with NA handling
├── data_preparation.py        # Pooled PCA, NA filtering, Stan data assembly
├── study_runner.py            # Pipeline orchestration (phases 1–4)
├── cli.py / __main__.py       # Command-line interface
│
├── position_analysis.py       # §6.2  Position bias analysis
├── consistency_analysis.py    # §6.3  Choice consistency across presentations
├── na_analysis.py             # §6.4  NA quality & sensitivity checks
├── visualization.py           # §6.1  Forest plots, density plots, bar charts
│
├── configs/
│   ├── study_config.yaml      # Default study parameters
│   └── prompts.yaml           # Deliberation & choice prompt templates
├── data/
│   └── claims.json            # 30-claim pool (C001–C030)
└── tests/                     # 122 tests
```

### Pipeline phases

| Phase | Module | Output |
|-------|--------|--------|
| 1. Problem generation | `problem_generator.py` | `problems.json` |
| 2a. Deliberation collection | `deliberation_collector.py` | `deliberations_T{t}.json`, `embeddings_T{t}.npz` |
| 2b. Choice collection | `choice_collector.py` | `choices_T{t}.json` |
| 3. Data preparation | `data_preparation.py` | `stan_data_T{t}.json`, `na_removal_log_T{t}.json` |
| 4. Model fitting (optional) | `study_runner.py` | CmdStanPy fit objects |

Phase 3 pools raw embeddings from **all** temperature conditions, fits PCA once on the pooled set, then projects each temperature's embeddings separately. This ensures a shared coordinate system without privileging any single temperature.

---

## Key design decisions

### NA handling (no silent defaults)

The pilot silently mapped unparseable LLM responses to position 0 (the first alternative), which introduced systematic position bias. This module instead:

1. `parse_choice()` returns `None` on any parse failure.
2. NA entries are recorded with `valid=false` and `position_chosen=null`.
3. Stan data is built from valid entries only.
4. NA rates, concentration, and effective sample sizes are reported per temperature via `na_analysis.py`.
5. A worst-case imputation function bounds the potential impact of excluded observations.

### Position counterbalancing

Each of the 100 problems is presented P = 3 times with claims shuffled to different positions. Consistency across presentations (same claim chosen despite different ordering) is evidence of content-driven choice rather than position bias.

### Per-temperature deliberation

Deliberations are collected at each temperature level. In `m_0`, the weight matrix `w[r]` represents how the decision-maker forms beliefs about consequences for alternative r. Temperature affects *how the LLM reasons*, which is part of the belief-formation mechanism that `w[r]` captures. See DESIGN.md §8 item 5 for the full rationale.

---

## Analysis modules

All analysis functions operate on the JSON data structures produced by the pipeline. They do not require a Stan fit — position, consistency, and NA analyses run on the choice data alone.

### Position bias (§6.2)

```python
from applications.temperature_study import position_analysis

# Per-temperature rates, chi-squared tests, Cramér's V
summary = position_analysis.position_bias_by_temperature(per_temp_choices)

# Per-claim position-1 vs. other comparison
claim_analysis = position_analysis.per_claim_position_analysis(choices)
```

### Choice consistency (§6.3)

```python
from applications.temperature_study import consistency_analysis

# Unanimity and modal-agreement rates by temperature
result = consistency_analysis.consistency_by_temperature(per_temp_choices)

# Shannon entropy of choice distributions
entropy = consistency_analysis.entropy_of_choices(choices)
```

### NA quality (§6.4)

```python
from applications.temperature_study import na_analysis

# Full §6.4 report
report = na_analysis.na_quality_report(per_temp_choices, nominal_M=300)

# Worst-case imputation for sensitivity check
augmented = na_analysis.worst_case_imputation(choices, imputed_position=1)
```

### Visualisation (§6.1)

```python
from applications.temperature_study import visualization

# Forest plot of α posteriors (requires posterior draws)
visualization.forest_plot(alpha_posteriors, save_path="forest.png")

# Position bias grouped bar chart
visualization.position_bias_bars(per_temp_rates, save_path="position_bias.png")

# Consistency line plot
visualization.consistency_line_plot(consistency_summary, save_path="consistency.png")

# Quantitative summaries (no plots)
prob = visualization.posterior_monotonicity_prob(alpha_posteriors)
slope = visualization.alpha_slope(alpha_posteriors)
table = visualization.alpha_summary_table(alpha_posteriors)
```

Matplotlib is optional — all plot functions return `None` gracefully when it is unavailable.

---

## Configuration

Edit `configs/study_config.yaml` or pass a custom YAML via `-c`:

```yaml
temperatures: [0.0, 0.3, 0.7, 1.0, 1.5]
num_problems: 100
num_presentations: 3
K: 3
target_dim: 32
llm_model: "gpt-4o"
embedding_model: "text-embedding-3-small"
provider: "openai"
seed: 42
```

Prompt templates live in `configs/prompts.yaml` and use `{claims_list}`, `{target_letter}`, and `{num_range}` placeholders.

---

## Testing

```bash
# Run all 122 tests
python -m pytest applications/temperature_study/tests/ -v

# Run a specific test file
python -m pytest applications/temperature_study/tests/test_position_analysis.py -v
```

Test coverage by module:

| Test file | Module under test | Tests |
|-----------|-------------------|-------|
| `test_problem_generator.py` | Problem generation & formatting | 23 |
| `test_deliberation_collector.py` | Deliberation collection | 7 |
| `test_choice_collector.py` | Choice collection & NA handling | 15 |
| `test_data_preparation.py` | PCA, filtering, Stan data | 21 |
| `test_study_runner.py` | End-to-end pipeline integration | 8 |
| `test_position_analysis.py` | Position bias statistics | 12 |
| `test_consistency_analysis.py` | Unanimity & agreement metrics | 12 |
| `test_na_analysis.py` | NA rates, concentration, imputation | 11 |
| `test_visualization.py` | Plots & posterior summaries | 13 |

All tests use mock LLM clients — no API keys required.

---

## Dependencies

Core (in `environment.yml`): `numpy`, `scipy`, `scikit-learn`, `pyyaml`, `openai`, `cmdstanpy`

Optional: `matplotlib` (for visualisation), `anthropic` (for Anthropic provider)

---

## Estimated costs

With default parameters (100 problems × 3 presentations × 5 temperatures):

| Component | API calls | Est. cost |
|-----------|-----------|-----------|
| Deliberations | ~1,500 | ~$3.75 |
| Choices | 1,500 | ~$1.25 |
| Embeddings | ~1,500 | ~$0.03 |
| **Total** | **~4,500** | **~$5–7** |

Run `python -m applications.temperature_study estimate-cost` for a breakdown based on your current config.
