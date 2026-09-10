# SEU Sensitivity Study Preregistration

Frozen: 2026-09-06

This document freezes the production design and confirmatory decision rules.
The executable design is
`applications/seu_sensitivity_study/configs/preregistered.yaml`.

## Scope and design

- Estimation pools: venture and hiring. Insurance is excluded because its
  embedding-based predictive-validity R3 was 0.052, below the frozen 0.30
  threshold, so alpha is not identified there.
- No third pool will be authored for this study.
- Each pool has 100 primary-family and 40 matched-family menus.
- Menu sizes are balanced over `{2,4,6,8}` and each menu is presented twice,
  with the second order reversed.
- The frozen variant-D recipe uses two contenders in every difficulty stratum.
- Six model arms and three prompt conditions form 18 cells per pool.
- The design contains 5,040 choice calls per pool and 10,080 total.

The embedding eta-gap axis remains primary. The belief-based axis may be
reported as a diagnostic but cannot replace the frozen gate or define a new
confirmatory analysis.

## Primary model

The primary model is `models/h_m01_size_pinned.stan` at
`max_treedepth=12`. For the three-outcome design, consequence utilities are
fixed to `(0, 0.5, 1)`. This is an identifying normalization, not a claim that
the middle consequence is known to have utility 0.5.

The same data will be refit with the middle utility fixed at 0.35 and 0.65.
Together with the primary 0.50 fit, these form the complete fixed utility-scale
sensitivity grid. Conclusions that change sign, interval decision, or
substantive interpretation across this grid will be labeled utility-scale
sensitive. The optional `Dirichlet(10,10)` model is not part of the frozen fit
campaign.

## Confirmatory decision rules

A log-alpha contrast is detected only when its credible interval excludes zero
and its magnitude exceeds `log(1.25)`, corresponding to a 25% multiplicative
change in alpha.

For RQ6, the menu-size slope has a separate ROPE:
`abs(gamma_size) > log(1.05)` per added alternative. Across menu sizes 2 to 8,
the boundary corresponds to an alpha ratio of approximately 1.34. A positive,
negative, or ROPE-indistinguishable result is reportable.

RQ4 is operationalized by the cross-cell/cross-pool variance component
`sigma_cell`. Cross-pool ordering agreement is descriptive and has no
confirmatory `(k, p0)` threshold.

NA rates are evaluated within cell. A rate above 10% is flagged. A rate above
30% triggers exclusion with an explicit caveat. Finer missingness attributions
remain post hoc.

## Gates and diagnostics

The frozen embedding-axis gate values are in
`applications/seu_sensitivity_study/configs/gate_thresholds.json`. All retained
estimation pools must pass the gate before production choice collection.

Final fits use four chains and must have R-hat at most 1.01, adequate bulk and
tail ESS, zero divergences, and no saturation at treedepth 12. Each launch has a
12-hour wall-clock stop rule. Posterior predictive checks are reported per pool
and cell.

The original free-utility model received marginal SBC and showed mild rank
drift in `gamma0`, `gamma_size`, and `sigma_cell`. No pinned-model SBC was run.
The fixed model instead relies on same-data convergence, J=18 recovery, and
existing null-calibration evidence; the earlier SBC is not claimed as proof of
calibration for the fixed model.

All Phase D power runs used treedepth 10, which truncated the J=18 geometry.
Their between-arm comparisons were affected symmetrically and are treated as
qualitative evidence, not precise final power estimates.

## Model-arm interpretation

The Anthropic flagship and reasoning arms use the same Sonnet 4.5 endpoint,
without and with extended thinking. The OpenAI flagship/reasoning comparison
uses different endpoints. These vendor-specific reasoning contrasts are
different estimands and will not be pooled.

## Collection budget

Production choice collection will use provider Batch APIs only after an E4
provider-specific dry run passes. The approved Batch ceiling is $31. The
synchronous fallback ceiling is $61 and requires separate approval before use.
API usage records, including input, output, cached, and thinking tokens, must be
persisted separately from the phase-local `run_summary.json`.

The expected choice cost is $24.17 at Batch rates, with the approved ceiling
providing a 25% contingency. Stan compute is scheduled as six serial fits for
two pools across the 0.35/0.50/0.65 utility grid. The observed conservative
envelope is 50.0 serial hours, subject to the 12-hour stop on every fit.

Before each provider Batch submission, the runner must durably reserve that
attempt's request count at `$31 / 10,080` per request in an append-only ledger.
Reservations are keyed by the durable submission intent and are never silently
released after failure, ambiguity, or completion. The planned 10,080-request
campaign therefore reserves the full approved $31 ceiling. Any retry,
replacement batch, or additional request requires separate authorization and a
corresponding ceiling amendment. A malformed or truncated reservation ledger
blocks submission rather than being ignored.

Every Batch submission must also belong to a named wave whose configuration
explicitly allowlists the cell ID. The checked-in production configuration has
no wave name and an empty allowlist, so it cannot submit merely by running the
choices phase. Enabling a wave is a separate authorization action and may cover
only the cells approved for that launch. A replacement attempt requires a new
authorization and budget amendment; it is never inferred from unused estimated
cost or a failed prior batch.

Before reservation, the authorized wave must pass the offline `preflight`
command. Preflight refreshes the configured validation gates, renders the exact
provider-ready request bodies, and creates a wave-specific read-only stage. Its
manifest binds the configuration, prerequisite source artifacts, prompts, fresh
gate reports, authorized cell IDs, and per-cell request hashes under an
aggregate hash. The reservation callback re-verifies the manifest, staged and
source artifacts, prompts, configuration, and the current cell request hash.
Any mismatch blocks before the budget ledger or provider is touched. A wave ID
is single-use and must be a path-safe identifier.

## Reporting restrictions

- Smoke estimates are feasibility evidence, not confirmatory results.
- The balanced position-stable subset is a sensitivity analysis, not the sole
  primary sample.
- Ordering agreement remains descriptive.
- Insurance is not an alpha estimation pool and will not be restored without a
  new preregistration.
- No production API collection begins until Phase E4 passes and the user gives
  a final explicit spending authorization.

## Amendment 1: assessment-anchored expected utilities

Amended: 2026-09-07, before production choice collection.

An independent frontier-model review identified a structural identification
failure in the frozen primary model at the production dimensions. With 60 items
and 32 embedding dimensions, the cell-specific belief map can rescale the
item-level expected-utility contrasts while an inverse rescaling of alpha leaves
the choice likelihood unchanged. Fixing consequence utilities removes the
utility-scale invariance but does not remove this belief-map/alpha invariance.
The earlier convergence and recovery evidence therefore does not establish
likelihood identification of alpha for the production design.

This amendment supersedes the **Primary model** section above. The primary
estimand is now assessment-anchored SEU sensitivity. For model arm `a`, pool
`p`, and item `r`, the neutral assessment collected before choice supplies the
stated consequence-probability vector `q[a,p,r]`. Expected utility is fixed as

`eta[a,p,r] = q[a,p,r]' * (0, 0.5, 1)`.

The same fixed item values are used in all three prompt conditions for a given
model arm and pool. The choice likelihood estimates how strongly choices track
the SEU ranking implied by the model's own previously stated probabilities. It
does not infer a latent belief map from those choices. This is an
assessment-anchored application of the SEU-sensitivity framework, not an
unchanged deployment of the paper's latent-belief `m_0` implementation.

The utility-grid fits at middle utilities 0.35 and 0.65 remain required
sensitivity analyses. For each grid value `u`, expected utility is recomputed as
`q[a,p,r]' * (0, u, 1)` before fitting. The 0.50 model remains primary.

This amendment also supersedes the **RQ4** rule above. With only the two
deliberately selected pools, cross-domain robustness is descriptive. The report
will present posterior differences in model-effect contrasts between venture
and hiring, direction changes, uncertainty intervals, and ordering agreement.
Neither within-pool `sigma_cell` nor a two-pool between-domain variance
component is a confirmatory RQ4 estimand.

All other confirmatory rules remain provisional until the amended model,
simulation/recovery design, RQ5 contract, contrast family, multiplicity policy,
NA exclusions, and diagnostics are implemented and frozen in a further dated
amendment. Phase E4 is reopened. No production choice requests may be submitted
until the amended pipeline passes its validation gates and receives a new
explicit spending authorization.

## Amendment 2: anchored-model size-slope prior

Amended: 2026-09-07, before production choice collection.

The assessment-anchored model uses `gamma_size ~ normal(0, 0.2)`, replacing the
legacy `normal(0, 0.5)` prior inherited from the latent-belief model. On the
frozen size range 2 to 8, the legacy prior assigned approximately 44% probability
to alpha changing by more than a factor of 10 in either direction and had an
approximately `[0.007, 146]` central 90% interval for the size-8/size-2 alpha
ratio. It also generated repeated non-finite-logit proposal rejections in both
anchored smoke fits.

Using the persisted neutral assessments and exact frozen menus, an SD of 0.2
gives an approximately `[0.14, 7.35]` central 90% interval for that ratio and
about 5% total probability beyond a factor of 10. An SD of 0.1 was rejected as
too restrictive because the earlier venture smoke estimated `gamma_size = 0.309
[0.166, 0.452]`; the structural alpha/eta rescaling counterexample leaves
`gamma_size` unchanged, so that evidence remains relevant to prior calibration.
The RQ6 practical-effect threshold remains `abs(gamma_size) > log(1.05)` per
added alternative.

## Amendment 3: executable confirmatory analysis contract

Amended: 2026-09-09, before production choice collection.

Confirmatory decisions use central 90% credible intervals (posterior quantiles
0.05 and 0.95), matching the calibration and recovery convention. A contrast is
detected only when this interval excludes zero and the posterior median exceeds
the applicable ROPE in absolute magnitude. Direction follows the median's sign.
The primary log-alpha ROPE remains `log(1.25)`; the RQ6 slope ROPE remains
`log(1.05)` per added alternative. Final fits require both bulk and tail ESS of
at least 400 for every structural parameter, in addition to the other frozen
sampler gates.

For each pool's additive primary fit, RQ1 contains seven confirmatory model
contrasts: all five non-reference models versus GPT-4o, plus the OpenAI and
Anthropic within-vendor flagship-minus-small contrasts. RQ2 contains two:
SEU-maximizing minus neutral and deliberative minus neutral. Thus the additive
primary family has nine decisions per pool. The OpenAI within-vendor contrast is
the sign reversal of the GPT-4o-mini-versus-GPT-4o coefficient; both are retained
because the approved family includes both the model-coded and directly stated
within-vendor hypotheses.

There is no multiplicity adjustment. Hierarchical shrinkage and the ROPE are
the predeclared mitigation. Every family member, interval, decision, and the
total decision count will be reported; selecting a detected contrast for
isolated presentation is prohibited.

RQ3 remains secondary and descriptive. In the complete 6-model by 3-prompt
factorial, the 10 treatment-coded interaction columns have residualized rank 10,
exactly equal to the 10 cell dimensions left after the rank-8 additive design.
The full interaction and the additive model's cell residuals therefore span the
same likelihood space. A separate saturated interaction fit would change only
the prior parameterization, not add information. RQ3 will instead report
`sigma_cell`, the `sigma_cell * z_alpha` cell residuals, and derived prompt
difference-in-differences from the existing anchored fit, without a confirmatory
interaction decision. The generated analysis contract records and verifies the
rank identity against the production design matrix.

RQ4 remains descriptive. The confirmatory RQ5 estimand is the within-model
hiring-minus-procurement contrast from a dedicated 36-cell matched-item fit.
The earlier joint-PCA requirement is superseded by the assessment-anchored
model: this fit consumes fixed assessment-derived expected utilities and has no
embedding matrix or latent belief map. It joins the separately labeled hiring
and procurement assessments over the 24 matched merit keys and 40 exactly
paired menus per task. Its rank-14 design contains additive model and prompt
effects, a hiring-task indicator, and five model-by-task terms; these identify
six within-model hiring-minus-procurement contrasts. Cross-pool differences
between primary venture and hiring fits remain supporting evidence, not a
substitute. Its production-dimensional validation and 40-dataset recovery
campaign have passed the prespecified sampler gates.

The pipeline writes this policy as `analysis_contract.json` beside each pool's
Stan data. After whole-cell NA exclusion, every anchored primary and utility-grid
payload must retain full rank in its intercept-plus-design matrix. Failure stops
analysis rather than silently changing the confirmatory family.

## Amendment 4: presentation dependence and calibration

Amended: 2026-09-10, before production choice collection.

The primary analysis retains both frozen presentations of every menu. To assess
dependence induced by presenting the same alternatives twice, each primary
`u=0.50` fit will be repeated on two deterministic subsets: presentation 1 only
and presentation 2 only. Each subset contains at most one observation per menu
and removes within-menu duplication without conditioning on whether the two
observed choices agree. This sensitivity applies to both pool-specific fits and
the dedicated matched-item RQ5 fit. The utility-scale grid remains a full-data
sensitivity and is not crossed with the presentation sensitivity.

Every primary contrast and `gamma_size` will be compared across the full-data,
presentation-1-only, and presentation-2-only fits. Any change in sign, central
90% interval decision, or substantive interpretation will be reported. The
previously specified position-stable subset remains a separate robustness
analysis and is not treated as a dependence correction because it conditions on
observed agreement.

No new formal SBC campaign will be run for the assessment-anchored model. The
decision rests on direct production-geometry calibration: separate 40-dataset
recovery campaigns for venture, hiring, and matched RQ5 passed all sampler gates
after deterministic longer reruns, and their central 90% coverages were
consistent with nominal coverage at the available Monte Carlo resolution. This
decision does not transfer the original latent-belief model's SBC evidence to
the anchored model; it prioritizes direct recovery of the actual anchored
estimands and geometry.