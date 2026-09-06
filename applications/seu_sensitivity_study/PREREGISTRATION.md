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