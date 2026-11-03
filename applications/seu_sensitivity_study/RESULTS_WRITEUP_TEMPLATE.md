# SEU Sensitivity Study: Results Write-Up (All 24 Planned Fits)

Interim write-up, 2026-11-03. Generated from the complete confirmatory report by
`analysis/study_results_writeup.py`; every number below is filled from that report or from the
fit result files, not typed by hand. This is a working results memo for the author. It is not
the planned Quarto results report, has not been independently reviewed, and does not amend the
preregistration. Where wording here differs from `PREREGISTRATION.md` and its amendments, the
preregistration controls.

## 1. Status and scope

- Production collection (campaign `production_20261010`) is complete; no further collection is planned.
- All 24 planned posterior fits (venture, hiring and matched RQ5 × primary, two presentation-only
  subsets, two utility-grid variants and three prior variants) passed the frozen sampler gates
  (Section 3). The complete confirmatory report was built by the canonical
  `confirmatory_reporting.build_report_from_manifest` over all 24 fits.
- All {{VALUE:primary_decision_count}} named primary decisions are available
  ({{VALUE:distinct_up_to_sign}} distinct up to sign); {{VALUE:total_detected}} are detections
  (venture {{VALUE:venture_detected}}, hiring {{VALUE:hiring_detected}}, matched RQ5
  {{VALUE:matched_detected}}). Prior sensitivity is complete ({{VALUE:prior_complete}} fits).
- Sensitivity analyses, RQ3, RQ4, the Amendment 6 assessment scale, the Amendment 9 Sonnet-arm
  comparison, predictive checks and ceiling diagnostics are descriptive. They annotate the primary
  decisions; they never add, cancel or replace them.

## 2. How to read the estimates

The primary estimand is an equally weighted contrast of realized cell log sensitivities (Amendment 5).
"Sensitivity" is the SEU-sensitivity parameter alpha of the softmax choice model, with each item's
expected utility fixed from the model arm's own neutral probability assessment and imposed utilities
(0, 0.5, 1). A log contrast of d corresponds to a sensitivity ratio exp(d): choices track the
assessment-implied SEU ranking more sharply (ratio above 1) or less sharply (below 1).

A contrast is **detected** only when its central 90% interval excludes zero *and* its posterior
median lies outside the region of practical equivalence: |log 1.25| for RQ1/RQ2/RQ5 and |log 1.05| per
added alternative for the RQ6 size slope. The interval need not lie entirely outside the region.
Non-detection is not evidence of no difference. There is no multiplicity adjustment; hierarchical
shrinkage and the practical-equivalence rule are the prespecified mitigation, and the full family is
reported. The OpenAI flagship-minus-small contrast is the sign reversal of GPT-4o-mini minus GPT-4o;
both are retained as named decisions.

These are **assessment-conditioned conformity measures**, not objective decision quality,
correctness or deployment suitability. SEU is the analyst's benchmark: a model may coherently reject
the imposed utilities, and high sensitivity relative to its own assessments says nothing about
whether those assessments are well grounded.

## 3. Fits and sampler diagnostics

All fits used four chains, 1,000 warmup + 4,000 sampling iterations per chain (16,000 posterior
draws), adapt_delta 0.99, maximum tree depth 12 and seed 20261010. Gates: maximum R̂ < 1.01, bulk and
tail ESS ≥ 400, E-BFMI ≥ 0.3, zero divergences and zero depth-12 transitions. Six fits ran on the
original x86_64 Mac; eighteen ran on the arm64 Mac with rebuilt executables. Draws are not bitwise
comparable across machines. Sampling minutes exclude system sleep.

{{TABLE:sampler}}

Sampler acceptance establishes computational adequacy for these gates only; it does not establish
model adequacy (Section 8).

## 4. Primary decisions by pool (RQ1, RQ2, RQ6)

Ratios are exp(log contrast). RQ1 rows compare each arm with GPT-4o, plus the two within-vendor
flagship-minus-small contrasts; RQ2 rows compare prompt conditions; RQ6 is the per-added-alternative
menu-size slope.

### Venture

{{TABLE:venture_primary}}

### Hiring

{{TABLE:hiring_primary}}

### Reading the pool results

- **Within-vendor ordering (RQ1).** In both pools the flagship arm is more sensitive than the small
  arm: OpenAI {{RATIO:venture:rq1_openai_flagship_minus_small}} in venture and
  {{RATIO:hiring:rq1_openai_flagship_minus_small}} in hiring; Anthropic
  {{RATIO:venture:rq1_anthropic_flagship_minus_small}} and
  {{RATIO:hiring:rq1_anthropic_flagship_minus_small}}.
- **Comparisons with GPT-4o (RQ1).** GPT-4o-mini and Claude Haiku are less sensitive than GPT-4o in
  both pools. The remaining arms depend on the pool: in venture, Sonnet and Sonnet-thinking are less
  sensitive than GPT-4o and o3-mini is not detected; in hiring, o3-mini and Sonnet-thinking are more
  sensitive and Sonnet is not detected. The data therefore do not support a single cross-domain
  sophistication ordering of the reasoning-capable arms.
- **Prompt conditions (RQ2).** No prompt contrast is detected in either pool. In hiring, the
  SEU-maximizing-minus-neutral interval excludes zero, but its ratio,
  {{RATIO:hiring:rq2_seu_maximizing_minus_neutral}}, has a median inside the practical-equivalence
  region; under the frozen rule this is not a detection.
- **Menu size (RQ6).** Sensitivity increases with menu size in both pools:
  {{RATIO:venture:rq6_gamma_size}} (venture) and {{RATIO:hiring:rq6_gamma_size}} (hiring) per added
  alternative. exp(gamma_size) is a sensitivity multiplier per added alternative, not an accuracy gain
  in probability points, and it is estimated under the independent-choice likelihood (Sections 8 and 10).

## 5. Matched task context (RQ5)

The dedicated 36-cell matched fit pairs 24 procurement and 24 hiring items over 40 exactly paired menus
per task; each contrast is the within-model hiring-minus-procurement difference.

{{TABLE:matched_primary}}

- Hiring is more sensitive than procurement for GPT-4o ({{RATIO:matched_rq5:rq5_gpt_4o_hiring_minus_procurement}}),
  GPT-4o-mini ({{RATIO:matched_rq5:rq5_gpt_4o_mini_hiring_minus_procurement}}) and Claude Haiku
  ({{RATIO:matched_rq5:rq5_claude_haiku_4_5_hiring_minus_procurement}}). The o3-mini, Sonnet and
  Sonnet-thinking contrasts are not detected.
- The matched size slope is a **sensitivity estimate only**, not an additional RQ6 primary decision.
- **Ceiling qualification.** Five matched hiring-task cells, including all three GPT-4o hiring cells,
  have a conditional likelihood whose supremum is at infinite sensitivity (Section 9). Their finite
  posterior upper quantiles reflect prior regularization and hierarchical pooling, not likelihood-only
  bounds. The GPT-4o RQ5 contrast draws on these cells and should be read with that qualification; it
  is also not detected in the presentation-1-only subset (Section 7).

## 6. Descriptive analyses

### RQ3: model-by-prompt heterogeneity

{{TABLE:rq3}}

Posterior medians of σ_cell range from {{VALUE:sigma_min}} to {{VALUE:sigma_max}}, and no
model-by-prompt difference-in-differences 90% interval excludes zero. These descriptive results do not
establish absence of interactions, and positive-support σ_cell intervals excluding zero would not by
themselves establish interactions. RQ3 has no confirmatory decision.

### RQ4: cross-domain comparison (descriptive)

Independent pool posteriors are combined by resampling; draws are never index-paired across fits.

{{TABLE:rq4}}

Every RQ1 contrast differs between hiring and venture with a 90% interval excluding zero, and the
median signs disagree for o3-mini, Sonnet and Sonnet-thinking relative to GPT-4o. Across the 15
pairwise model orderings, median signs agree in {{VALUE:rq4_orderings_agree}}. With only two
deliberately selected pools, this is descriptive evidence about these two domains, not a
between-domain variance estimate or a confirmatory test.

### Amendment 9: Sonnet-thinking minus Sonnet

{{TABLE:sonnet}}

Descriptive only, with no detection, ROPE or equivalence classification (Amendment 9): the venture
median is negative and its 90% interval includes zero ({{VALUE:sonnet_venture}}); the hiring contrast is
positive with a 90% interval above zero ({{VALUE:sonnet_hiring}}). These compare configured arms under
their own fixed neutral assessments and differing request settings (including thinking budget and
temperature), not a pure causal effect of reasoning or thinking. All fit variants are shown in
Section 7.

### Amendment 6: assessment-scale transformation

{{TABLE:assessment_scale}}

The standardization creates no new decision and does not alter the primary decisions.
{{VALUE:amendment6_invariance}} The table flags descriptive sign reversals and changes in interval
zero-exclusion only; they qualify cross-arm comparisons of scale, not the primary estimates.

## 7. Sensitivity analyses

For the primary RQ1/RQ2/RQ5 estimands and menu-size slopes, the tables list rows with a sign,
decision-rule annotation or substantive-interpretation change; the remaining rows of those families
are unchanged. The Amendment 9 descriptive contrast is reported separately for every applicable
variant at the end of this section. Loss or gain of detection in a half-data presentation fit is not
by itself evidence of a between-presentation effect, and independent fits are never paired draw by draw.

### Presentation dependence (Amendment 4)

Venture:

{{TABLE:venture_presentation}}

Hiring:

{{TABLE:hiring_presentation}}

Matched RQ5:

{{TABLE:matched_presentation}}

### Utility scale (middle utility 0.35 / 0.50 / 0.65)

Venture:

{{TABLE:venture_utility}}

Hiring:

{{TABLE:hiring_utility}}

Matched RQ5:

{{TABLE:matched_utility}}

### Prior sensitivity (Amendment 7)

Priors L (wider common level), H (relaxed between-cell shrinkage) and S (wider size slope) were run on
exactly the primary data for each group. A changed decision is labeled prior-sensitive under these
specified checks; unchanged decisions do not imply unchanged magnitudes, and the checks are not an
exhaustive robustness envelope.

{{TABLE:prior_overview}}

{{TABLE:prior_flagged}}

Both prior-sensitive changes occur under prior H and concern estimands that are not detected under the
primary prior: the venture o3-mini-minus-GPT-4o contrast and the matched size slope (itself only a
sensitivity estimate). Under H the matched-RQ5 interval widths grow by up to the ratio shown above,
consistent with relaxed shrinkage.

### Amendment 9 contrast across all variants (descriptive)

Sonnet-thinking minus Sonnet for every primary-pool fit variant. No decision or equivalence
classification applies; the interval column only describes the 90% interval's position relative to 0.

{{TABLE:sonnet_variants}}

The venture contrast is not stable across variants: its median changes sign in the presentation-2-only
and utility-0.35 fits, and its interval lies below zero at utility 0.65. In hiring, the interval
includes zero in the presentation-1-only fit and lies above zero in the other variants. These
variations are descriptive and carry the same configured-arm qualification as Section 6.

## 8. Posterior predictive checks (Amendment 8, A4)

A4 checks use the saved replicated choices and joint draws of each fit on that fit's retained
observations; they are in-sample, descriptive central-90% review flags, not p-values, discoveries or
a gate. The A4 summaries were computed with the archived batched calculator; for each fit, a seven-draw
probe of that calculator matched the canonical calculator exactly ({{VALUE:a4_probes}} fits).

**Same-item repetition** across the two presentations of a menu (primary fits, pool or task level):

{{TABLE:repetition}}

Observed repetition exceeds the predictive range in venture, hiring and the matched procurement task.
Under the independent-choice likelihood this suggests that posterior uncertainty for contrasts drawing
on repeated menus may be understated; the presentation-only fits (Section 7) are the prespecified
response. Same-item repetition is not position invariance or a causal repeated-exposure effect.

**Qualification flags for every fit** (applies, with the number of flagged rows):

{{TABLE:qualifications}}

Choice-pattern, display-position and size-trend discrepancies are flagged in every fit; repetition is
flagged wherever both presentations are retained. Presentation-only fits cannot evaluate repetition by
design ("Unavailable"). Flags qualify interpretation without cancelling decisions; their absence would
not establish model adequacy.

## 9. Ceiling diagnostics (Amendment 7, A3)

Prior-free, full-menu conditional likelihood slices in cell log sensitivity, for the three full-data
midpoint-0.5 datasets:

{{TABLE:ceiling}}

Eventual upper-tail decay alone does not establish useful precision. For cells with a supremum at
infinite sensitivity (choices always among the assessed maximizers on menus with unequal utilities),
posterior upper quantiles reflect the prior and pooling. These are not profile likelihoods or
confidence sets.

## 10. Limitations

- **Conformity, not quality.** Results measure conformity to SEU relative to each arm's own elicited
  probabilities under imposed utilities; they are not objective performance, fairness or deployment
  evidence. The decision materials are synthetic.
- **Independence.** The likelihood treats the two presentations of each menu as independent; the
  excess-repetition flag (Section 8) raises concern that this assumption may understate uncertainty.
  The presentation-only fits provide the prespecified sensitivity assessment.
- **Retention.** Analyses condition on retained observations (unresolved responses excluded per the
  frozen rules); missingness is not modeled.
- **Two domains.** Cross-domain statements (RQ4) are descriptive for two selected pools.
- **Two machines.** Archived x86_64 fits and new arm64 fits are not bitwise comparable; re-deriving
  the archived primary reports on arm64 matched to within {{VALUE:cross_venture}} (venture) and
  {{VALUE:cross_hiring}} (hiring) relative difference with identical decisions.
- **Report construction.** On the 16 GiB arm64 machine, the canonical report builder was run with a
  projection-backed fit loader (structural columns copied verbatim into hash-verified diagnostic
  projections of each chain; per-observation arrays read from the full chains) and the archived batched
  A4 calculator, verified per fit by a seven-draw exact probe against the canonical calculator.
  Both substitutions are recorded in the report.

## Provenance

- Complete report: `applications/seu_sensitivity_study/results/complete_report_20261102_v1/complete_report.json`
  (SHA-256 `{{VALUE:complete_report_sha256}}`); summary `{{VALUE:summary_sha256}}`; fit manifest
  `{{VALUE:fit_manifest_sha256}}`.
- Generator `analysis/study_results_writeup.py` (SHA-256 `{{VALUE:generator_sha256}}`) and template
  `applications/seu_sensitivity_study/RESULTS_WRITEUP_TEMPLATE.md` (SHA-256 `{{VALUE:template_sha256}}`).
  Regenerate with `python -m analysis.study_results_writeup` from the repository root; the large result
  artifacts are in the private archive releases, not in Git.
- Off-machine copies: private releases `venture-utility-20261102-v1` and `remaining-fits-20261102-v1`
  in `jeffhelzner/seu-sensitivity-archive`, together with the earlier collection and analysis releases.
