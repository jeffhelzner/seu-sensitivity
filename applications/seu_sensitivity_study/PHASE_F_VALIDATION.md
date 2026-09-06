# Phase F: Utility-Scale Validation

Date: 2026-09-05

## Decision

The primary J=18 inference model fixes the shared utility increments at equal
spacing. For K=3 this means `delta = (0.5, 0.5)` and
`upsilon = (0, 0.5, 1)`. The original model remains unchanged; the primary
variant is `models/h_m01_size_pinned.stan`.

This is a substantive normalization, not a claim that the middle consequence is
known to have utility 0.5. It removes a weakly learned utility-curvature
dimension that produced a chain-dependent ridge between the scale of expected
utility contrasts and alpha at J=18.

## Same-Data Diagnostic

Both models were fit to the exact persisted seed-24681 dataset: J=18, K=3,
P=7, 30 menus per cell, two presentations, `M_total=1080`, `rho_copy=0.9`,
true `sigma_cell=0.10`, four chains, 1,000 warmup draws, and
`max_treedepth=12`. The original and first pinned fits used 200 sampling draws;
the pinned confirmation used 800 draws.

Between/within entries are the maximum across the named parameter group. Values
below 0.3 indicate chain agreement.

| Diagnostic | Original, 200 draws | Pinned, 200 draws | Pinned, 800 draws |
|---|---:|---:|---:|
| Identified beta disagreement | 3.607 | 0.131 | 0.080 |
| Eta disagreement | 0.694 | 0.186 | 0.094 |
| Alpha-cell disagreement | 0.327 | 0.175 | 0.058 |
| Sigma-cell disagreement | 0.231 | 0.066 | 0.021 |
| Delta disagreement | 0.710 | fixed | fixed |
| Maximum R-hat, all monitored parameters | 1.5202 | 1.0178 | 1.0035 |
| Minimum bulk ESS | 7 | 189 | 682 |
| Mean treedepth | 10.25 | 9.69 | 9.68 |
| ESS per 1,000 seconds | 0.7 | 23.9 | 59.9 |
| Divergences | 0 | 0 | 0 |
| Wall clock, seconds | 10,904 | 7,901 | 11,386 |

The 800-draw confirmation reached maximum treedepth 10, so no draw saturated
the configured limit of 12. All four chain CSVs are retained in raw and gzip
form under `results/power/f4_pinned_hard_800/chains/`.

## Verdict

The equal-spacing variant passes all Phase F4 criteria on the same dataset that
produced the original pathology. Every sampled disagreement ratio is below 0.1,
maximum R-hat is 1.0035, minimum bulk ESS is 682, and sampling efficiency rises
from 0.7 to 59.9 bulk-ESS units per 1,000 seconds. Relative to the original fit,
the long pinned fit moves the posterior means of `gamma0` by -0.116 and
`gamma_size` by +0.072; their signs are unchanged, their 90% intervals overlap
strongly, and the shifts are approximately 0.45 and 0.76 pooled posterior SD.
The fix therefore removes the chain-dependent ridge without materially changing
the global estimands used for the main model and menu-size contrasts.

## Draft Preregistration Wording

The primary analysis fixes the K=3 consequence utilities to `(0, 0.5, 1)`.
This equal-spacing normalization is imposed because validation at the full
18-cell design found that estimating the middle utility jointly with the belief
maps produced a multimodal utility-scale/sensitivity ridge. The normalization
does not assert that equal spacing is known to be true. After data collection,
we will refit the model at fixed middle-utility values 0.35, 0.50, and 0.65,
holding the data and all other modeling choices fixed. We will report whether
the signs, posterior intervals, and substantive decisions for the preregistered
gamma and alpha contrasts change across this grid. Conclusions that depend on a
single spacing value will be labeled utility-scale-sensitive.

As an additional regularization sensitivity, we may fit
`delta ~ Dirichlet(10, 10)`. For K=3 this is equivalent to
`delta[1] ~ Beta(10, 10)`, centered at 0.5 while retaining uncertainty. Existing
matched-prior recovery and SBC results support concentration 10 as a calibrated
prior intervention, but do not establish robustness to prior misspecification
or show improved likelihood-based identification. This model is therefore a
secondary, prior-driven sensitivity analysis unless it separately passes the
same J=18 hard-dataset convergence criteria; it is not a substitute for the
fixed-grid analysis.

## Evidence

- Original calibration: `results/power/f3_pathology_search_24681/calibration.json`
- Original disagreement: `results/power/f3_pathology_search_24681_diagnostic/beta_identification.json`
- Pinned confirmation: `results/power/f4_pinned_hard_800/calibration.json`
- Pinned disagreement: `results/power/f4_pinned_hard_800_diagnostic/beta_identification.json`
- Concentrated-prior study: `reports/foundations/13_concentrated_delta_prior.qmd`