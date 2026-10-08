# October Independent Review Disposition

Date: 2026-10-05; updated 2026-10-07

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
| A1: additive versus realized estimands | Implemented under Amendment 5 on fixed-design scientific grounds, not greater detection rates. October 7 saved-draw verification now covers realized RQ1/RQ2/RQ5, with the historical choice-input provenance limitation recorded separately. The earlier report already distinguished additive components, so a literal prose-code contradiction was overstated. |
| A2: assessment scale | Approved and implemented in Amendment 6: transform primary posterior draws using finite-set SDs over all 60 primary-pool items or each matched task's own 24. Report fixed-reference gap/tie geometry, original/standardized intervals, offsets and descriptive sign/zero-exclusion changes without ROPE decisions or refits. Hash-bound assembly references are recomputed and checked against retained eta; full menu counts and size/family allocation are enforced. Exact affine rescaling preserves choice probabilities; this does not show that different arms actually differ only by rescaling or that their stated beliefs are artifacts. |
| A3: ceilings and prior sensitivity | Approved and implemented under Amendment 7: retained-choice geometry, conditional likelihood slices and exact limiting classifications, plus L/H/S prior checks on all three full-data datasets. Offline prior predictions and 841 application tests pass. Nine additional fits are planned, not executed or resource-authorized. No maximizer-rate gate or replacement primary prior is adopted. Posterior robustness and runtime/mixing remain unmeasured; recovery under fitting priors does not guarantee real-data adequacy. |
| A4: predictive adequacy for size effects | Approved and implemented under Amendment 8: size/stratum, authored-filler, exposure-aware item, displayed-position and paired-presentation checks with explicit descriptive flags and RQ6 qualifications. No new fits or predictive gate. Offline tests pass; no production adequacy result exists. Optional anchored E1 pilot fit still requires separate authorization. |
| B1: dependence | Retain response-independent presentation fits and report effect/interval changes, not just detection switches. Amendment 8 implements item and paired-presentation diagnostics; excess repetition qualifies independence but does not identify its cause. No requirement to exclude zero in both half-sized datasets and no menu random-effect model are added. |
| B2: utility grid | Accept limited scope of the existing grid. Few argmax changes do not imply that full softmax probabilities are unchanged. A wider-grid or heuristic comparison needs a specified purpose and method before adding fits. Historical normalization wording is clarified by the preregistration banner. |
| B3: treatment construction | Implemented under Amendment 9: separate descriptive Sonnet-thinking minus Sonnet realized-cell comparison within venture and hiring, with fixed equal prompt weights, intervals, sign probabilities and geometric-mean sensitivity ratios for all supplied valid variants. Configured arms use their own neutral assessments and differing request settings on the same endpoint; this is not a pure causal reasoning effect. No new primary hypothesis or matched contrast. Retain answer-first/output constraints and visible-probability disclosures; do not infer a necessary downward sensitivity bias from temperature alone. |
| B4: matched comparison | Consequence wording and coding-induced prior differences remain disclosed. Verified realized RQ5 coverage is 218/240 contrast-dataset cases, mean interval width 0.5418. Review additive rates reproduce but concern different truths/bin memberships; they are not realized-RQ5 rates or a paired power comparison. |
| B5: recovery evidence | October 7 audit verifies current truth mapping, 480 chain selections, generating priors and saved-draw scores for 120 iterations. Venture/hiring share full truths; matched residuals overlap a shifted RNG stream: 40 coupled seed clusters, not 120 independent truths. Original temporary choice inputs are missing, so exact choice replay and replacement-input equality remain unverified. Final review must explicitly dispose of this provenance limit. Historical evidence is untouched. |
| B6: missingness | Amendment 8 implements size/stratum/presentation counts and unresolved-rate ranges with reconciled denominators, plus retention qualifications alongside RQ6. No imputation trigger or extra sensitivity fit is adopted. These diagnostics do not establish robustness to missing-not-at-random choices. |
| C1: duplicate decisions | Correct count is 26 named decisions and 24 distinct comparisons up to sign, not 25: the OpenAI duplicate is present in each pool. Updated contract and report. |
| C2: superseded text | Added a banner to the preregistration identifying controlling amendments; preserved historical record. |
| C3: size-prior interval | Reconciled October 7: analytic central 90% interval is [0.1389, 7.1982] for exp(6 gamma_size), using SD 0.2. Report labels and links E4's [0.135, 7.03] and Amendment 2's [0.14, 7.35] as historical finite-draw approximations, not alternative priors or fresh results. |
| C4: display rounding and ties | Reconciled October 7: a report-local table filter displays neutral probabilities to six decimals without altering raw YAML or verbatim requests. Separate frozen-input appendix reports exact ties and additional near ties for all six arms in each primary pool and matched subset, using canonical normalized eta and absolute tolerance 1e-12, relative tolerance zero. Historical evidence and predictive exact-maximizer rules remain unchanged. |
| C5: encoded position summary | Resolved under Amendment 8 with actual displayed-position checks stratified by size and presentation. Any retained sorted-index output is explicitly a legacy encoding diagnostic, not display-position evidence. |
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

Amendment 8 implementation and the full application suite passed after repair
of an independent audit finding: differing floating-point aggregation paths
could flag identical replicas. Regression tests now preserve exact equality
and row-permutation results. The production-shaped offline benchmark completed
without fits; expanded output is about 5.2 MB per representative fit. Current
preparation requires additional canonical role evidence; historical readability
does not establish A4 completeness. Production adequacy remains unassessed.

Amendment 9 implements B3 with policy `B3_descriptive_postreview_2026-10-07`.
Focused checks cover direct realized draws, zero gamma with nonzero residuals,
cell/column permutations, missing required cells, count-independent prompt
weights, sign probabilities and valid prior-variant report integration. The
26 named / 24 distinct primary decisions and 24-fit plan remain unchanged.
No new matched contrast, assessment-scale grid, or numeric production result
is added. Implementation checks do not establish behavioral or causal effects.

Validation on October 7: 198 distinct tests passed across the focused
contrast, realized-estimand, reporting and prior-contract modules, including
the separately run all-24-fit synthetic integration test. The QMD rendered
with execution disabled. No full application-suite rerun or posterior fitting
was performed for this narrow addition.

October 7 saved-draw verification is recorded in
[the recovery note](REALIZED_RECOVERY_VERIFICATION_20261007.md) and a separate
hash-bound artifact. All 120 current iterations verify and are sampler-eligible;
all five reviewer detection rows reproduce. The verifier's 32 focused tests and
exact full-artifact reproduction pass. Aggregate coverage denominators contain
dependent contrasts and shared truths; they are not independent trial counts.
No production adequacy, new SBC, null false-positive rate, or alternative-prior
validation follows from this reanalysis.

October 7 C3/C4 reconciliation passes 14 focused tests, including exact agreement
with Amendment 6's geometry, the near-tie boundary, analytic prior quantiles,
and table-only formatting with raw request preservation. The separate display
supplement reproduces exactly from the frozen bundle and records source hashes.
No assessment, menu, prior, likelihood, decision rule, or historical snapshot
was changed.

October 7 machine-contract reconciliation replaces obsolete pending-recovery
metadata with verified saved-draw calculations and explicit unresolved
choice-input provenance. It records 120 iterations, 480 selected chains and
40 coupled seed clusters without asserting independent contrast cases,
exact-null calibration, alternative-prior recovery or production adequacy.
The no-new-SBC decision is unchanged. A read-only contract review also found
an unqualified 15-fit count in current outputs; this now states 24 total,
15 base and 9 additional prior fits, all planned rather than authorized.
The Amendment 5 estimand version and 26 named / 24 distinct decisions remain
unchanged. Pool and matched contracts are tested against the actual fit plan.

The full application suite passed 935 tests in 6345.11 seconds after the
recovery-status edit, before the subsequent fit-count metadata correction.
After that correction, 53 focused contract/realized-estimand tests and all
116 confirmatory-reporting tests passed (the latter in 795.23 seconds).
Both October prior/recovery artifacts were regenerated and reproduced exactly
against the final code. Structured comparisons with the committed artifacts
found only source-hash changes: confirmatory_analysis in both, and a previously
stale data_preparation hash in the prior artifact. All 6,410 prior and 18,098
recovery numerical leaves were unchanged. No posterior fit or provider call
was performed, and the September snapshot was not refreshed.

Next: explicitly resolve acceptance or remediation of missing historical
choice-input provenance and obtain independent scientific acceptance of the
amended contract. The implementation review and tests do not constitute that
acceptance or final-revision preflight.
Only then obtain any required fit authorization and final-revision preflight
and collection authorization. The clean-revision guard remains unchanged.