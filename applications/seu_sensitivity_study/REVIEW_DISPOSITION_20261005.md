# October Independent Review Disposition

Date: 2026-10-05; updated 2026-10-06

Review: `local/seu_sensitivity_precollection_final_review.md`, reviewing
`db145fe4a7b2340558b26e009fac55b27ea82d24`.

Status: **NOT READY**. The author approved realized-cell primary estimands with
additive companions and a descriptive assessment-scale analysis. This document
records that decision and the remaining conditions, not a blanket acceptance
of the review's quantitative claims or recommended implementations.

## Approved Estimand Change

[Amendment 5](PREREGISTRATION.md#amendment-5-realized-cell-primary-estimands)
specifies equally weighted, draw-wise realized log-sensitivity contrasts for
RQ1/RQ2/RQ5. RQ4 uses the corresponding realized model comparisons. Additive
gamma contrasts remain descriptive. Thresholds, likelihood, priors, 15 fit
variants, and 26 named decisions are unchanged. Missing required cells make
affected contrasts unavailable, without reweighting or gamma substitution.

Implementation includes explicit cell-ID weights, permutation and missing-cell
tests, separate companion reporting, and output schema 2. The historical
September evidence bundle is preserved, not regenerated to masquerade as
validation of the amended contract.

## Finding Dispositions

| Finding | Disposition and remaining work |
|---|---|
| A1: additive versus realized estimands | Scientific choice accepted and implemented under Amendment 5. The current report had already distinguished additive components from realized differences, so a literal prose-code contradiction was overstated. Fixed-design questions motivate the change, not greater detection rates. Verify saved-draw operating characteristics, including realized RQ5, before claiming validation. |
| A2: assessment scale | Approved and implemented in Amendment 6: transform primary posterior draws using finite-set SDs over all 60 primary-pool items or each matched task's own 24. Report fixed-reference gap/tie geometry, original/standardized intervals, offsets and descriptive sign/zero-exclusion changes without ROPE decisions or refits. Hash-bound assembly references are recomputed and checked against retained eta; full menu counts and size/family allocation are enforced. Exact affine rescaling preserves choice probabilities; this does not show that different arms actually differ only by rescaling or that their stated beliefs are artifacts. |
| A3: ceilings and prior sensitivity | Approved and implemented under Amendment 7: retained-choice geometry, conditional likelihood slices and exact limiting classifications, plus L/H/S prior checks on all three full-data datasets. Offline prior predictions and 841 application tests pass. Nine additional fits are planned, not executed or resource-authorized. No maximizer-rate gate or replacement primary prior is adopted. Posterior robustness and runtime/mixing remain unmeasured; recovery under fitting priors does not guarantee real-data adequacy. |
| A4: predictive adequacy for size effects | Accept the gap. Add size/stratum, filler, item-share and paired-presentation checks, with explicit statistic definitions and interpretation before production. Authored filler labels must be distinguished from assessed value ranks. A failure to reproduce relevant behavior must qualify RQ6; a numerical gate has not yet been chosen. Optional anchored E1 pilot fit requires separate authorization. |
| B1: dependence | Retain response-independent presentation fits. Report effect and interval changes, not just a binary switch in detection. Do not automatically demand exclusion of zero in both half-sized datasets as proof of robustness. Item and paired-presentation diagnostics remain open; a menu random-effect model is not silently added. |
| B2: utility grid | Accept limited scope of the existing grid. Few argmax changes do not imply that full softmax probabilities are unchanged. A wider-grid or heuristic comparison needs a specified purpose and method before adding fits. Historical normalization wording is clarified by the preregistration banner. |
| B3: treatment construction | Accept clearer interpretation as wording under fixed answer-first/output constraints, and explicit disclosure that probabilities are visible. Do not infer a necessary downward sensitivity bias from temperature alone. Add Sonnet-thinking minus Sonnet as a descriptive comparison in follow-up; do not add a primary hypothesis without a new decision. |
| B4: matched comparison | Accept disclosure of consequence wording and coding-induced prior differences; read together with the descriptive scale check. Review-reported additive detection rates are not realized-RQ5 rates. Verify before incorporation. |
| B5: recovery evidence | Verify and report shared truth vectors, independent simulated choices, and the generating priors. Preserve the distinction among 120 fits, three design geometries, and independent truth draws. Do not copy operating-characteristic rates into the evidence bundle before checking truth mapping, chain selection, counts and amended RQ5 contrasts. |
| B6: missingness | Accept size/stratum breakdowns. Specify how differential missingness qualifies RQ6 and whether an additional sensitivity analysis is needed; no arbitrary trigger adopted yet. |
| C1: duplicate decisions | Correct count is 26 named decisions and 24 distinct comparisons up to sign, not 25: the OpenAI duplicate is present in each pool. Updated contract and report. |
| C2: superseded text | Added a banner to the preregistration identifying controlling amendments; preserved historical record. |
| C3: size-prior interval | Analytic interval is authoritative; historical Monte Carlo approximations must stay labeled with their sources rather than rewritten as new results. Further documentary reconciliation remains open. |
| C4: display rounding and ties | Preserve raw numerical evidence. Display formatting and tolerance-consistent tie reporting remain open; do not change raw probabilities to hide floating-point representation. |
| C5: encoded position summary | Accept stratification by size or removal. Resolve alongside A4, while retaining the distinction from displayed positions. |
| C6: self-review | Report explicitly labels the author's nine affirmative responses as self-review, not independent acceptance. |

## Validation and Next Steps

The implementation passed 166 focused contrast/reporting tests and 718 tests
under `applications`, without fits or provider calls. Regression tests hold
gamma fixed while varying residual averages, cover equally weighted matched
contrasts, reordered cells and columns, missing cells, and RQ4 independence.
These are computational checks of estimand calculation, not new recovery,
power, or behavioral model-adequacy evidence.

Amendment 6 validation: 776 application tests passed, including posterior
probability invariance, RQ2/RQ6 cancellation, matched-task support, missing
cells, zero SD, reference tampering, fixed reference sets under exclusions,
and rejection of truncated/deduplicated menu references. No additional fits
or primary decisions were introduced. These tests do not verify production
behavior or close the A3/A4 conditions.

Amendment 7 validation: the application suite passed 841 tests after an
independent audit prompted full observation-universe reconciliation, frozen
presentation-order binding, canonical eta arithmetic in prior predictions,
and empty-prior-directory handling. The hash-bound offline prior check uses
10,000 draws per group/prior on the frozen design and reproduces exactly.
This closes A3 specification and offline implementation, not the unrun posterior
sensitivity assessment. A3 manifests require rebuilt version-2 observation
evidence; historical report compatibility does not establish A3 completeness.

Next: settle and implement A4 diagnostic policies, the descriptive Sonnet
comparison, and associated reporting/missingness details; verify the review's
saved-draw calculations; update the current review evidence without overwriting
the September snapshot; test and review the amended contract as a whole.
Only then obtain any required fit authorization and final-revision preflight
and collection authorization. The clean-revision guard remains unchanged.