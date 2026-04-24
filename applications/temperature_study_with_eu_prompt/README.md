# Temperature Study with EU Prompt

Extends the base temperature study by collecting choices under a prompt that explicitly instructs the LLM to maximise expected utility. This tests whether explicit EU-framing changes estimated sensitivity (α), and whether the effect interacts with temperature.

The module reuses problems, assessments, and embeddings from `temperature_study` — only new choices are collected, keeping API costs minimal.

---

## Quick start

```bash
# From the project root (seu-sensitivity/)
conda activate seu-sensitivity

# 1. Validate config and check base study data availability
python -m applications.temperature_study_with_eu_prompt validate

# 2. Estimate API costs (choices only — no assessments or embeddings needed)
python -m applications.temperature_study_with_eu_prompt estimate-cost

# 3. Run the pipeline (collect EU-prompt choices → prepare Stan data)
python -m applications.temperature_study_with_eu_prompt run --skip-fitting

# 4. Fit m_0 at each temperature
python -m applications.temperature_study_with_eu_prompt fit

# 5. Re-run data preparation from saved choices
python -m applications.temperature_study_with_eu_prompt prepare
```

All subcommands accept `-c PATH` to override the default config at `configs/study_config.yaml`.

---

## CLI reference

| Command | Description | Key options |
|---------|-------------|-------------|
| `validate` | Check config validity and base study data availability | `-c`, `-v` |
| `estimate-cost` | Print estimated API call count and cost | `-c`, `-v` |
| `run` | Run pipeline: collect choices → prepare Stan data (→ fit) | `--skip-collection`, `--skip-fitting`, `-c`, `-v` |
| `prepare` | Re-run data preparation from saved EU-prompt choices | `-c`, `-v` |
| `fit` | Fit Stan models on existing Stan data files | `-c`, `-v` |

Global flags: `-c / --config` (custom YAML path), `-v / --verbose` (debug logging).

---

## Architecture

```
temperature_study_with_eu_prompt/
├── cli.py                  # 5 subcommands
├── config.py               # StudyConfig dataclass + YAML loader
├── choice_collector.py     # Collect EU-prompt choices per temperature
├── data_preparation.py     # Build Stan data using base-study embeddings
├── study_runner.py         # Pipeline orchestration
├── __init__.py             # Public API exports
├── __main__.py             # Module entry point
│
├── configs/
│   ├── study_config.yaml   # Default study parameters
│   └── prompts.yaml        # EU-prompt choice template
├── data/                   # Generated data (if any)
└── results/                # Collected choices + Stan data
```

---

## Key design decisions

### Reuse from base temperature study

This module does **not** re-collect assessments, embeddings, or problem definitions. It reads them from `temperature_study/results/` and only collects new choice data under the modified EU-prompt. This ensures that the only difference between the base study and this study is the prompt wording.

### Stan model

Fits the Stan model specified by `stan_model` in the config (default: `m_01`, the calibrated variant of `m_0` with prior `alpha ~ lognormal(3.0, 0.75)`) at each temperature level, producing α posteriors that can be directly compared to the base temperature study's posteriors.

---

## Configuration

Edit `configs/study_config.yaml` or pass a custom YAML via `-c`:

```yaml
temperatures: [0.0, 0.3, 0.7, 1.0, 1.5]
K: 3
llm_model: "gpt-4o"
provider: "openai"
stan_model: "m_01"
base_study_results_dir: "../temperature_study/results"
```

---

## Dependencies

Core (in `environment.yml`): `numpy`, `pyyaml`, `openai`, `cmdstanpy`

Reuses `LLMClient`, `create_llm_client`, and `parse_choice` from `applications.temperature_study.llm_client` — the base temperature study module must be importable.
