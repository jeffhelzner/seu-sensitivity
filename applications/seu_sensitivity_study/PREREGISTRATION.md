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