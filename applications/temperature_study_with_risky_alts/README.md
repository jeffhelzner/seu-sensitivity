# Temperature Study with Risky Alternatives

Extends the temperature study by collecting **risky choice data** from an LLM at five temperature levels. Each risky alternative specifies an explicit probability distribution over K = 3 consequences, so the decision-maker does not need to form beliefs — only to evaluate lotteries. The collected risky data merges with the existing uncertain data (from `temperature_study`) to produce Stan data packages for models **m_1**, **m_2**, and **m_3**.

---

## Quick start

```bash
# 1. Validate configuration, alternatives, and uncertain data availability
python -m applications.temperature_study_with_risky_alts validate

# 2. Estimate API cost before committing
python -m applications.temperature_study_with_risky_alts estimate-cost

# 3. Run the full pipeline (generate problems → collect choices → merge)
python -m applications.temperature_study_with_risky_alts run

# 4. Prepare Stan data only (skip collection, use existing choices)
python -m applications.temperature_study_with_risky_alts prepare

# 5. Fit m_1, m_2, m_3 on previously prepared data
python -m applications.temperature_study_with_risky_alts fit
```

---

## CLI reference

| Command | Description | Key options |
|---------|-------------|-------------|
| `validate` | Check config, 30 simplex sums, uncertain data files | `-c`, `-v` |
| `estimate-cost` | Print API call count and estimated cost | `-c`, `-v` |
| `run` | Full pipeline: generate → collect → merge (→ fit) | `--skip-collection`, `--skip-fitting`, `-c`, `-v` |
| `prepare` | Merge existing risky + uncertain data into Stan JSON | `-c`, `-v` |
| `fit` | Fit m_1, m_2, m_3 on prepared Stan data | `-c`, `-v` |

Global flags: `-c / --config` (custom YAML path), `-v / --verbose` (debug logging).

---

## Typical workflows

### Collect → inspect → merge

```bash
python -m applications.temperature_study_with_risky_alts validate
python -m applications.temperature_study_with_risky_alts estimate-cost
python -m applications.temperature_study_with_risky_alts run --skip-fitting
# Inspect results/risky_choices_T*.json, then:
python -m applications.temperature_study_with_risky_alts fit
```

### Re-merge without re-collecting

If you already have risky choice data and want to rebuild the Stan data packages:

```bash
python -m applications.temperature_study_with_risky_alts prepare
```

### Full end-to-end

```bash
python -m applications.temperature_study_with_risky_alts run
```

This generates problems, collects choices at all five temperatures, merges with uncertain data, and fits all three models.

---

## Architecture

```
temperature_study_with_risky_alts/
├── cli.py                         # 5 subcommands
├── config.py                      # StudyConfig dataclass + YAML loader
├── risky_problem_generator.py     # Generate 100 problems from 30 alternatives
├── risky_choice_collector.py      # Collect LLM choices per temperature
├── data_preparation.py            # Build risky Stan block, merge with uncertain
├── study_runner.py                # Pipeline orchestration (4 phases)
├── __init__.py                    # Public API exports
├── __main__.py                    # Module entry point
│
├── configs/
│   ├── study_config.yaml          # Default study parameters
│   ├── prompts.yaml               # Risky choice prompt template
│   └── risky_alternatives.json    # 30 alternatives with probability simplexes
├── data/                          # Generated problems
└── results/                       # Collected choices + Stan data
```

### Pipeline phases

| Phase | Module | Output |
|-------|--------|--------|
| 1. Problem generation | `risky_problem_generator.py` | `data/risky_problems.json` |
| 2. Choice collection | `risky_choice_collector.py` | `results/risky_choices_T{t}.json` |
| 3. Data preparation | `data_preparation.py` | `results/augmented_stan_data_T{t}.json` |
| 4. Model fitting (optional) | `study_runner.py` | CmdStanPy fit objects |

Phase 3 loads the existing uncertain Stan data from `temperature_study/results/stan_data_T{t}.json` and augments it with the risky data block `{N, S, x, J, z}`. The output file contains both the uncertain block `{M, K, D, R, w, I, y}` and the risky block, ready for m_1, m_2, or m_3.

---

## Key design decisions

### Risky alternatives pool

The 30 alternatives in `configs/risky_alternatives.json` use hand-crafted K = 3 probability simplexes covering four structural groups:

| Group | IDs | Feature |
|-------|-----|---------|
| Near-vertex | R01–R06 | One outcome 80–90% likely |
| Edge-concentrated | R07–R12 | One outcome near 0% |
| Moderate asymmetry | R13–R18 | Mid-range skew |
| Interior / balanced | R19–R30 | Spread across all three outcomes |

Every simplex sums to 1.0. The `validate` command checks this at runtime.

### Problem construction

Each of the 100 risky problems samples 2–4 alternatives from the pool (without replacement within a problem). Problems are generated once, then reused across all temperatures and presentations. Each problem has P = 3 shuffled presentations for position counterbalancing.

### NA handling (no silent defaults)

Follows the same policy as `temperature_study`:

1. `parse_choice()` returns `None` on any parse failure.
2. NA entries are recorded with `valid = false` and `position_chosen = null`.
3. Stan data is built from valid entries only.

### Merging with uncertain data

The module does **not** re-collect uncertain data. It reads the existing `stan_data_T{t}.json` files from `temperature_study/results/` and appends the risky block. This ensures that uncertain data (M, K, D, R, w, I, y) is identical to stand-alone uncertain fits while adding the risky observations (N, S, x, J, z) that m_1, m_2, and m_3 require.

### Stan data dimensions

| Symbol | Meaning | Default value |
|--------|---------|--------------|
| M | Uncertain observations (problems × presentations) | 300 |
| K | Number of consequences | 3 |
| D | Embedding dimensions (PCA-reduced) | 32 |
| R | Number of uncertain alternatives (claims) | 30 |
| N | Risky observations (problems × presentations) | 300 |
| S | Number of risky alternatives | 30 |

---

## Configuration

Edit `configs/study_config.yaml` or pass a custom YAML via `-c`:

```yaml
temperatures: [0.0, 0.3, 0.7, 1.0, 1.5]
num_problems: 100
min_alternatives: 2
max_alternatives: 4
num_presentations: 3
K: 3
llm_model: "gpt-4o"
provider: "openai"
seed: 42
fit_models: false
```

Prompt templates live in `configs/prompts.yaml` and use `{alternatives_list}` and `{num_range}` placeholders.

---

## Estimated costs

With default parameters (100 problems × 3 presentations × 5 temperatures):

| Component | API calls | Est. cost |
|-----------|-----------|-----------|
| Risky choices | 1,500 | ~$2.00 |

Run `python -m applications.temperature_study_with_risky_alts estimate-cost` for a breakdown based on your current config.

---

## Dependencies

Core (in `environment.yml`): `numpy`, `pyyaml`, `openai`, `cmdstanpy`

Reuses `LLMClient`, `create_llm_client`, and `parse_choice` from `applications.temperature_study.llm_client` — the original temperature study module must be importable.
