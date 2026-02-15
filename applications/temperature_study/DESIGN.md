# Temperature Study: Experimental Design and Implementation Plan

## Executive Summary

This document proposes a new experimental application of the SEU sensitivity models that examines **how LLM temperature affects estimated sensitivity (α)**. The study addresses methodological issues discovered in the prompt framing pilot by implementing position counterbalancing and capturing per-claim "assessment embeddings" that reflect the AI's evaluation of each alternative.

---

## 1. Research Question

**Primary Question:** Does increasing the temperature parameter of an LLM decrease its estimated sensitivity (α) to expected utility maximization in a fraud detection task?

**Hypothesis:** Higher temperature settings introduce more noise into the LLM's decision-making process, which should manifest as lower α estimates in the SEU model (i.e., choices become less sensitive to expected utility differences).

**Rationale:** Temperature controls the "softmax temperature" in the LLM's token sampling, making high-probability tokens relatively less dominant. If the LLM's "internal" decision process is analogous to EU maximization, increasing external sampling noise should reduce the estimated sharpness of choice behavior.

---

## 2. Experimental Design

### 2.1 Independent Variable: Temperature

| Level | Temperature | Description |
|-------|-------------|-------------|
| 1 | 0.0 | Deterministic (greedy decoding) |
| 2 | 0.3 | Low variance |
| 3 | 0.7 | Moderate variance (typical default) |
| 4 | 1.0 | High variance |
| 5 | 1.5 | Very high variance |

**Note:** Temperature=0 provides a "ceiling" for sensitivity (no sampling noise). We expect a monotonic decrease in α as temperature increases.

### 2.2 Dependent Variable

Estimated sensitivity parameter **α** from the m_0 model, fit separately to choice data from each temperature condition.

### 2.3 Task Structure

- **Domain:** Insurance claims triage
- **Decision:** Select which claim to forward for investigation
- **Alternatives per problem:** 2 to 4 claims (randomly varied)
- **Consequences (K=3):**
  1. Both investigators agree the selection warrants investigation (best)
  2. One agrees, one doesn't (middle)
  3. Neither agrees (worst)
- **Prompt framing:** Fixed at "baseline" level (moderate framing, no explicit rationality emphasis) to isolate temperature effects

### 2.4 Addressing Position Bias: Repeated Presentation with Randomization

To address the position bias confound discovered in the pilot:

**Design:** Each decision problem is presented **P times** (e.g., P=3), with **claims randomly shuffled to different positions** in each presentation.

**Shuffling strategy:** For each presentation, we generate a random permutation of the claims. We do *not* attempt exhaustive coverage of all N[m]! possible orderings (which would be impractical for problems with 4 alternatives where 4! = 24). Instead, P random shuffles provide probabilistic coverage of different position assignments.

**Benefits:**
1. **Position counterbalancing:** Each claim appears in different positions across presentations (probabilistically)
2. **Multiple observations:** Each presentation becomes a separate observation in the model, increasing effective sample size
3. **Position bias estimation:** We can directly estimate and control for position effects

**Example:**
```
Problem P001 has claims [C003, C007, C012]

Presentation 1: [C003, C007, C012] → LLM chooses "Claim 2" → C007 selected
Presentation 2: [C007, C012, C003] → LLM chooses "Claim 1" → C007 selected  
Presentation 3: [C012, C003, C007] → LLM chooses "Claim 3" → C007 selected

Result: C007 chosen 3/3 times despite appearing in positions 2, 1, and 3
        → Strong evidence of content-driven choice (not position bias)
```

**Note on notation:** We use P for presentations to avoid confusion with R in m_0.stan, which denotes the number of distinct alternatives across all problems.

### 2.5 Assessment-Based Feature Descriptions

**Problem with context-dependent deliberation:** An earlier design asked the LLM to deliberate on each claim within the context of a specific decision problem (all other claims visible). This produced embeddings that varied by problem context, violating the m_0 model assumption that `w[r]` is a fixed property of alternative *r*.

**Current approach:** For each claim in the pool, ask the LLM to **assess the claim on its own merits**, independent of any specific decision problem. Then embed this assessment response.

**Benefits:**
- One embedding per claim, consistent with `w[r]` in m_0.stan
- No averaging hack needed to collapse context-dependent embeddings
- Dramatically fewer API calls (30 claims × 5 temps = 150 vs. ~1,500)
- More interpretable: assessments capture claim-level features, not problem-context effects

---

## 3. Prompt Specifications

### 3.1 Assessment Prompt

Used to elicit a per-claim evaluation that is independent of any specific decision problem. Called once per claim in the pool (not once per problem × claim).

```
SYSTEM PROMPT:
You are evaluating insurance claims to help determine which should be
forwarded to experienced fraud investigators for further review.

USER PROMPT:
Consider the following insurance claim:

{claim_description}

Provide a brief assessment (2-4 sentences) covering:
1. What indicators suggest this claim may or may not warrant investigation?
2. How likely is it that experienced fraud investigators would agree this
   claim deserves their attention?
3. What is the risk of this being a false positive vs. a true positive?
```

**Example instantiation:**

```
Consider the following insurance claim:

Auto collision claim for $12,000. Claimant reports rear-end collision
at low speed but vehicle shows extensive front-end damage. Police report
not filed. Three prior claims in past 2 years.

Provide a brief assessment (2-4 sentences) covering:
1. What indicators suggest this claim may or may not warrant investigation?
2. How likely is it that experienced fraud investigators would agree this
   claim deserves their attention?
3. What is the risk of this being a false positive vs. a true positive?
```

**Expected response (to be embedded):**

> This claim presents several red flags: the damage pattern is inconsistent
> with the reported accident mechanism (front damage from a rear-end
> collision), no police report was filed, and there is a pattern of
> prior claims. Experienced investigators would likely view this as a
> reasonable referral, as the physical evidence inconsistency alone warrants
> closer examination. The risk of false positive is relatively low given
> the multiple independent indicators.

### 3.2 Choice Prompt

Used after the assessment phase to collect the final choice. The choice agent sees the **assessment texts** (not raw claim descriptions) so that its decisions are based on the same information encoded in `w[r]`. The order of claims is shuffled for each presentation.

```
SYSTEM PROMPT:
You are evaluating insurance claims to help determine which should be
forwarded to experienced fraud investigators for further review.

USER PROMPT:
You are reviewing insurance claims and must select ONE to forward to a
team of two experienced fraud investigators for further review.

Your decision will be evaluated based on the investigators' assessments:
- Best outcome: Both investigators agree your selection warrants investigation
- Middle outcome: One investigator agrees, one does not
- Worst outcome: Neither investigator agrees with your selection

Here are the assessments for the claims under consideration:

{assessments_list}

Which claim do you select for investigation?

Respond with ONLY the claim number ({num_range}).
```

Where `{assessments_list}` is formatted as:
```
- Claim 1: [assessment text for claim at position 1]
- Claim 2: [assessment text for claim at position 2]
- Claim 3: [assessment text for claim at position 3]
```

**Note:** The assessment texts shown in the choice prompt come from the assessment phase at the *same temperature*. This ensures the choice agent's information basis is consistent with the temperature condition being tested.

### 3.3 Prompt Design Rationale

| Design Choice | Rationale |
|---------------|-----------|
| Per-claim assessments (no problem context) | Produces one embedding per claim, consistent with `w[r]` in m_0 |
| Assessments shown in choice prompt | Choice agent decides based on same info encoded in `w[r]` |
| Numeric labels (1, 2, 3) in choice | Standard choice format, enables position bias analysis |
| Separate assessment and choice phases | Assessments produce embeddings; choices produce `y[m]` |
| K=3 outcome structure explicit | Matches model assumptions; provides clear utility ordering |
| "ONLY the claim number" instruction | Reduces parsing errors; cleaner response extraction |
| No explicit EU language | Baseline framing; isolates temperature effects from rationality framing |
| Neutral system prompts (no "flagged" language) | Avoids priming the LLM toward suspicion |

---

## 4. Study Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Number of base problems | 100 | Sufficient for model fitting |
| Presentations per problem (P) | 3 | Balance counterbalancing vs. cost |
| Alternatives per problem | 2-4 (random) | Realistic variation; tests model with varying choice sets |
| Temperature conditions | 5 | Cover full practical range |
| Claim pool size | 30 | Larger than pilot (20) for better coverage |
| Embedding dimension | 32 | Matches pilot; adequate for claim diversity |
| LLM model | gpt-4o | Modern, capable model |
| Embedding model | text-embedding-3-small | Cost-effective, good quality |

### 4.1 Estimated API Calls

| Component | Calculation | API Calls |
|-----------|-------------|-----------|
| Assessments | 30 claims × 5 temps | 150 |
| Choices | 100 problems × P(=3) presentations × 5 temps | 1,500 |
| Embeddings | 150 assessment responses | 150 |
| **Total** | | **~1,800** |

### 4.2 Estimated Costs

| Component | Est. Tokens | Est. Cost (gpt-4o) |
|-----------|-------------|-------------------|
| Assessments (input) | ~45K | ~$0.11 |
| Assessments (output) | ~15K | ~$0.23 |
| Choices (input) | ~600K | ~$1.50 |
| Choices (output) | ~15K | ~$0.25 |
| Embeddings | ~30K | ~$0.003 |
| **Total** | **~705K** | **~$2-3** |

*Note: Costs are estimates based on current gpt-4o pricing. Significantly lower than the previous deliberation-based design (~$5-7) because assessments are per-claim, not per-problem×claim.*

---

## 5. Data Collection Protocol

### Phase 1: Problem Generation

1. Load claim pool (30 claims with descriptions)
2. Generate 100 decision problems:
   - For each problem, randomly select 2-4 claims
   - Record the base claim set (canonical ordering)
3. For each problem, generate P=3 randomly shuffled orderings (presentations)
4. Save problem definitions with all presentation orderings

**Output:** `problems.json`
```json
{
  "problems": [
    {
      "id": "P001",
      "claim_ids": ["C003", "C007", "C012"],
      "num_alternatives": 3,
      "presentations": [
        {"presentation_id": 1, "order": ["C003", "C007", "C012"]},
        {"presentation_id": 2, "order": ["C007", "C012", "C003"]},
        {"presentation_id": 3, "order": ["C012", "C003", "C007"]}
      ]
    }
  ]
}
```

### Phase 2: Assessment Collection (Per Temperature)

For each temperature T in [0.0, 0.3, 0.7, 1.0, 1.5]:

1. For each claim in the pool (30 claims):
   - Send assessment prompt with the claim's description (no other claims visible)
   - Record assessment response
   - Embed the assessment response
2. **After all temperatures are collected:** Pool the raw embeddings from all five temperature conditions and fit PCA on the pooled set to learn the projection matrix. Then apply that projection to each temperature's raw embeddings separately. This ensures a shared coordinate system derived from the full range of assessment variation, without arbitrarily privileging any single temperature.
3. Save assessments and reduced embeddings

**Why per-claim (not per-problem×claim):** In the m_0 model, `w[r]` is a fixed property of alternative *r*. Assessments collected per-claim directly produce one embedding per claim, consistent with this assumption. The earlier deliberation-based design produced context-dependent embeddings that required averaging across problem contexts—an unsatisfying workaround.

**Why per-temperature assessment:** Temperature affects *how the LLM reasons* about each claim. Collecting assessments at each temperature level captures a genuine aspect of the belief-formation process. Fixing assessments at a single temperature would artificially decouple belief formation from the temperature manipulation.

**Output per temperature:** `assessments_T{temp}.json`, `embeddings_raw_T{temp}.npz`

### Phase 3: Choice Collection (Per Temperature)

For each temperature T in [0.0, 0.3, 0.7, 1.0, 1.5]:

1. For each problem:
   - For each presentation (shuffled ordering):
     - Send choice prompt with **assessment texts** (from the same temperature) in presentation order
     - Parse response to extract position choice (1-indexed)
     - If parsing fails (no valid integer, out of range, or ambiguous response): mark as **NA** and log the raw response. Do **not** default to any position—this was a source of systematic position bias in the prompt framing pilot.
     - If parsing succeeds: map position to claim ID
2. Log summary of NA counts per temperature (total NAs, NA rate)
3. Save all choices with position metadata, including NA entries

**Output per temperature:** `choices_T{temp}.json`
```json
{
  "temperature": 0.7,
  "total_choices": 300,
  "valid_choices": 294,
  "na_choices": 6,
  "na_rate": 0.02,
  "choices": [
    {
      "problem_id": "P001",
      "presentation_id": 1,
      "claim_order": ["C003", "C007", "C012"],
      "position_chosen": 2,
      "claim_chosen": "C007",
      "valid": true,
      "raw_response": "2"
    },
    {
      "problem_id": "P042",
      "presentation_id": 2,
      "claim_order": ["C015", "C003", "C021"],
      "position_chosen": null,
      "claim_chosen": null,
      "valid": false,
      "raw_response": "I would recommend Claim 2, but Claim 1 also has merit..."
    }
  ]
}
```

### Phase 4: Data Preparation

For each temperature:

1. **Remove NA observations:** Filter out any choice entries where `valid == false`. Log the number and identity of removed observations. Stan does not accept missing values, so these must be excluded before constructing the data structure.

2. Build Stan data structure from **valid observations only**:
   - `M`: number of valid observations (≤ 100 × P = 300; exact count depends on NA removals)
   - `K`: number of consequences (3)
   - `D`: embedding dimension (32)
   - `R`: number of distinct claims across all problems (from the claim pool)
   - `w[R, D]`: assessment embeddings for each distinct claim
   - `I[M, R]`: indicator matrix (which claims in which problem-presentation)
   - `y[M]`: chosen claim for each valid observation

3. **Save NA removal log** alongside Stan data for audit and reporting:
   - Per-temperature: number of observations removed, which problem-presentations were affected
   - Cross-temperature: whether NA rates differ systematically by temperature (this itself is informative—higher temperatures may produce more unparseable responses)

4. **Data structure approach:** Each valid presentation is treated as a separate observation in the model. This means:
   - M ≤ 300 observations per temperature condition (after NA removal)
   - The same base problem may appear up to P times with potentially different choices
   - This naturally captures choice variability and increases statistical power
   - No aggregation or majority voting required—m_0.stan handles this directly

**Output:** `stan_data_T{temp}.json`, `na_removal_log_T{temp}.json`

### Phase 5: Model Fitting

1. Fit m_0 model to each temperature condition
2. Run posterior predictive checks
3. Extract α posteriors
4. Compare across temperatures

---

## 6. Analysis Plan

### 6.1 Primary Analysis: Temperature Effect on Sensitivity

**Visualization:** Forest plot of α posterior distributions by temperature

**Quantitative tests:**
- Posterior probability that α(T=0) > α(T=0.3) > ... > α(T=1.5)
- Estimated slope: Δα per unit increase in temperature
- 90% credible intervals for each α

### 6.2 Position Bias Analysis

For each temperature:
1. Calculate position choice rates (% choosing position 1, 2, 3, 4)
2. Test for deviation from uniform (expected if no bias)
3. Estimate position effect size
4. Test whether position bias varies with temperature

**Key question:** Does higher temperature increase or decrease position bias?

### 6.3 Choice Consistency Analysis

For each problem and temperature:
1. Count how many presentations yielded the same choice
2. Calculate consistency rate: % of problems with unanimous choice across presentations
3. Test whether consistency decreases with temperature

**Expected:** Consistency should decrease as temperature increases (more random choices)

### 6.4 Data Quality: NA Rates and Excluded Observations

Because failed choice parses are treated as NA (not silently defaulted to a position), the analysis must report:

1. **NA rate per temperature:** Total NAs and percentage of observations excluded. If NA rates increase with temperature, this is itself evidence that higher temperatures degrade response quality.
2. **NA distribution:** Whether NAs concentrate in specific problems or are spread uniformly. Concentrated NAs could indicate problematic claim descriptions.
3. **Effective sample size:** Report actual M per temperature condition (after NA removal) alongside the nominal M = 300.
4. **Sensitivity check:** Re-run primary analysis under a conservative scenario where all NAs are imputed as "worst-case" choices (e.g., position 1) to bound the potential impact of exclusions.

**Rationale:** The prompt framing pilot silently defaulted unparseable responses to position 0 (the first alternative), which likely contributed to the observed position bias. Transparent NA handling avoids this artifact.

### 6.5 Robustness Checks

- Vary embedding dimensionality (16, 32, 64)
- Alternative aggregation: all presentations as separate observations
- Subset analysis: problems with 2 vs. 3 vs. 4 alternatives

---

## 7. Implementation Plan

### 7.1 Directory Structure

```
applications/temperature_study/
├── __init__.py                    # Public API exports
├── README.md                      # User-facing documentation
├── DESIGN.md                      # This document
│
├── # Core modules
├── config.py                      # Configuration dataclasses & validation
├── llm_client.py                  # LLM API interface (adapted from prompt_framing_study)
├── problem_generator.py           # Problem generation with shuffled presentations
├── assessment_collector.py         # Per-claim assessment elicitation & embedding
├── choice_collector.py            # Choice collection with position tracking
├── study_runner.py                # Main pipeline orchestration
├── data_preparation.py            # Stan data packaging
│
├── # Analysis modules  
├── position_analysis.py           # Position bias analysis
├── consistency_analysis.py        # Choice consistency across presentations
├── visualization.py               # Result visualization
│
├── # CLI
├── cli.py                         # Command-line interface
├── __main__.py                    # Entry point for `python -m temperature_study`
│
├── # Configuration
├── configs/
│   ├── study_config.yaml          # Main study parameters
│   └── prompts.yaml               # Prompt templates
│
├── # Data
├── data/
│   └── claims.json                # Claim pool (30 claims)
│
├── # Results (gitignored)
├── results/
│
└── # Tests
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_problem_generator.py
    ├── test_assessment_collector.py
    ├── test_choice_collector.py
    ├── test_data_preparation.py
    ├── test_study_runner.py
    └── ...
```

### 7.2 Module Specifications

#### `config.py`
```python
@dataclass
class StudyConfig:
    # Temperature conditions
    temperatures: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.7, 1.0, 1.5])
    
    # Problem parameters
    num_problems: int = 100
    min_alternatives: int = 2
    max_alternatives: int = 4
    num_presentations: int = 3  # P
    
    # Model parameters
    K: int = 3  # consequences
    target_dim: int = 32  # embedding dimension
    
    # API settings
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    
    # Reproducibility
    seed: int = 42
```

#### `problem_generator.py`
- `ProblemGenerator.generate_problems()` - create base problems with 2-4 claims
- `ProblemGenerator.generate_presentations()` - create P randomly shuffled orderings per problem

#### `assessment_collector.py`
- `AssessmentCollector.collect_temperature(temperature)` 
- Iterates over claims (not problems × claims); one assessment per claim per temperature
- Embeds each assessment response
- `get_assessment_texts()` extracts {claim_id → text} for the choice phase

#### `choice_collector.py`
- `ChoiceCollector.collect_choices(problems, temperature)`
- Tracks position chosen and maps to claim ID
- Failed parses → `valid=false` with `position_chosen=null`; never defaults to a position
- Logs all NA responses with raw LLM output for post-hoc inspection

#### `study_runner.py`
- `TemperatureStudyRunner.run()` - full pipeline
- Checkpoint/resume support
- Progress logging and cost tracking

### 7.3 Implementation Phases

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. Foundation | 1-2 days | Directory structure, config.py, llm_client.py |
| 2. Data Collection | 2-3 days | problem_generator.py, deliberation_collector.py, choice_collector.py |
| 3. Integration | 1-2 days | study_runner.py, data_preparation.py, CLI |
| 4. Analysis | 1-2 days | position_analysis.py, visualization.py |
| 5. Testing & Docs | 1 day | Unit tests, README, example notebooks |

**Total estimated time:** 6-10 days

**Import convention:** All cross-module imports (e.g., from the top-level `analysis` package) must use proper relative imports. Do not use `sys.path` manipulation as was done in the prompt framing pilot—this is fragile and breaks under different working directories.

---

## 8. Resolved Design Decisions

1. **PCA fitting strategy:** Assessments are collected at every temperature level (see #5). To reduce embedding dimensionality, raw embeddings from all five temperature conditions are *pooled* and PCA is fit on the pooled set to learn a projection matrix. That projection is then applied to each temperature's embeddings separately. This derives the reduced coordinate system from the full range of assessment variation rather than arbitrarily anchoring on a single temperature, while still ensuring a shared basis for cross-condition comparability of `beta` estimates.

2. **Data structure:** Each presentation is treated as a separate observation in the model (M ≤ 100 × P = 300 per temperature, after NA removal). No aggregation or majority voting—the model naturally handles repeated observations of the same base problem.
   
3. **Claim pool:** Expand to 30 claims (from pilot's 20) for better coverage.

4. **Cross-model comparison:** Deferred to future work pending results of this initial study.

5. **Per-temperature assessment:** Assessments are collected at each temperature level (not fixed at a single temperature). In the m_0 model, `w[r]` forms the basis for the decision maker's subjective probabilities via `softmax(beta * w[r])`. Temperature affects the LLM's reasoning process, which is part of the belief-formation mechanism that `w[r]` is meant to capture. Fixing assessments at one temperature would artificially decouple belief formation from the manipulation.

6. **Assessment-based design (replaces deliberation-based):** Assessments are collected per-claim (not per-problem×claim), producing one embedding per claim per temperature. This is consistent with m_0's `w[r]` being a fixed property of alternative *r*. The choice agent sees assessment texts rather than raw claim descriptions, ensuring decisions are based on the same information encoded in `w`. The earlier deliberation-based design produced context-dependent embeddings that required averaging—this is eliminated.

7. **NA handling for failed choice parses:** Unparseable LLM responses are recorded as NA and excluded before passing data to Stan (which does not accept missing values). This replaces the prompt framing pilot's approach of silently defaulting to position 0 (the first alternative), which likely contributed to the observed position bias. NA rates and excluded observations are reported transparently in the analysis (see §6.4).

---

## 9. Success Criteria

1. **Technical completeness:** All modules implemented, tested, documented
2. **Data quality:** >95% valid choice responses; <5% NA rate per temperature. All NAs logged, excluded from Stan data, and reported in analysis. No silent defaults to any position.
3. **Statistical power:** Posterior distributions sufficiently narrow to detect meaningful α differences
4. **Position bias control:** Demonstrated reduction in position confound vs. pilot
5. **Interpretability:** Assessment embeddings provide meaningful, examinable features

---

## 10. Approval Checklist

- [ ] Research question and hypothesis approved
- [ ] Temperature levels confirmed
- [ ] Prompt templates approved
- [ ] 2-4 alternatives per problem (not fixed at 3) confirmed
- [ ] Cost estimate acceptable
- [ ] Implementation timeline acceptable
- [ ] Open questions resolved

---

**Please review and provide feedback. Implementation will begin upon approval.**
