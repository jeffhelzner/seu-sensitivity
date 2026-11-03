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
- All 26 named primary decisions are available
  (24 distinct up to sign); 17 are detections
  (venture 7 of 10, hiring 7 of 10, matched RQ5
  3 of 6). Prior sensitivity is complete (9 of 9 fits).
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

| Group | Variant | N | Machine | Max R̂ | Min ESS bulk / tail | Min E-BFMI | Divergences | TD12 share | Sampling min |
|---|---|---|---|---|---|---|---|---|---|
| Venture | primary | 5028 | x86_64 | 1.0038 | 1943 / 2933 | 0.69 | 0 | 0.0 | 42.3 |
| Venture | presentation_1_only | 2512 | x86_64 | 1.0026 | 2836 / 2850 | 0.87 | 0 | 0.0 | 14.7 |
| Venture | presentation_2_only | 2516 | x86_64 | 1.0020 | 2965 / 4119 | 0.68 | 0 | 0.0 | 18.3 |
| Venture | utility_035 | 5028 | arm64 | 1.0019 | 2015 / 3944 | 0.70 | 0 | 0.0 | 8.9 |
| Venture | utility_065 | 5028 | arm64 | 1.0028 | 2330 / 3286 | 0.73 | 0 | 0.0 | 11.6 |
| Venture | prior_L | 5028 | arm64 | 1.0030 | 2657 / 3517 | 0.77 | 0 | 0.0 | 15.2 |
| Venture | prior_H | 5028 | arm64 | 1.0013 | 3412 / 4761 | 0.80 | 0 | 0.0 | 15.8 |
| Venture | prior_S | 5028 | arm64 | 1.0013 | 2196 / 3222 | 0.70 | 0 | 0.0 | 16.1 |
| Hiring | primary | 5032 | x86_64 | 1.0013 | 3599 / 6107 | 0.82 | 0 | 0.0 | 35.2 |
| Hiring | presentation_1_only | 2517 | x86_64 | 1.0006 | 4428 / 6368 | 0.86 | 0 | 0.0 | 10.0 |
| Hiring | presentation_2_only | 2515 | x86_64 | 1.0006 | 4232 / 5540 | 0.89 | 0 | 0.0 | 9.9 |
| Hiring | utility_035 | 5032 | arm64 | 1.0009 | 3542 / 6062 | 0.81 | 0 | 0.0 | 6.9 |
| Hiring | utility_065 | 5032 | arm64 | 1.0008 | 4101 / 5175 | 0.87 | 0 | 0.0 | 8.9 |
| Hiring | prior_L | 5032 | arm64 | 1.0009 | 3771 / 5622 | 0.83 | 0 | 0.0 | 12.4 |
| Hiring | prior_H | 5032 | arm64 | 1.0010 | 2945 / 5649 | 0.80 | 0 | 0.0 | 10.5 |
| Hiring | prior_S | 5032 | arm64 | 1.0008 | 2957 / 6077 | 0.81 | 0 | 0.0 | 13.0 |
| Matched RQ5 | primary | 2871 | arm64 | 1.0013 | 6619 / 8032 | 0.93 | 0 | 0.0 | 4.4 |
| Matched RQ5 | presentation_1_only | 1433 | arm64 | 1.0018 | 2122 / 4489 | 0.78 | 0 | 0.0 | 2.7 |
| Matched RQ5 | presentation_2_only | 1438 | arm64 | 1.0010 | 6909 / 8981 | 0.86 | 0 | 0.0 | 1.8 |
| Matched RQ5 | utility_035 | 2871 | arm64 | 1.0021 | 6070 / 8224 | 0.92 | 0 | 0.0 | 5.6 |
| Matched RQ5 | utility_065 | 2871 | arm64 | 1.0011 | 4796 / 7820 | 0.95 | 0 | 0.0 | 4.2 |
| Matched RQ5 | prior_L | 2871 | arm64 | 1.0011 | 4721 / 6544 | 0.89 | 0 | 0.0 | 5.9 |
| Matched RQ5 | prior_H | 2871 | arm64 | 1.0017 | 4689 / 6462 | 0.93 | 0 | 0.0 | 8.4 |
| Matched RQ5 | prior_S | 2871 | arm64 | 1.0014 | 4762 / 5924 | 0.97 | 0 | 0.0 | 5.6 |

Sampler acceptance establishes computational adequacy for these gates only; it does not establish
model adequacy (Section 8).

## 4. Primary decisions by pool (RQ1, RQ2, RQ6)

Ratios are exp(log contrast). RQ1 rows compare each arm with GPT-4o, plus the two within-vendor
flagship-minus-small contrasts; RQ2 rows compare prompt conditions; RQ6 is the per-added-alternative
menu-size slope.

### Venture

| RQ | Contrast | Median (log) | 90% interval | Ratio | Ratio 90% | Decision |
|---|---|---|---|---|---|---|
| RQ1 | gpt-4o-mini minus gpt-4o | −1.196 | [−1.385, −1.016] | 0.30 | [0.25, 0.36] | detected − |
| RQ1 | o3-mini minus gpt-4o | −0.153 | [−0.364, 0.053] | 0.86 | [0.69, 1.05] | not detected |
| RQ1 | claude-sonnet-4-5 minus gpt-4o | −0.327 | [−0.528, −0.130] | 0.72 | [0.59, 0.88] | detected − |
| RQ1 | claude-haiku-4-5 minus gpt-4o | −1.524 | [−1.703, −1.350] | 0.22 | [0.18, 0.26] | detected − |
| RQ1 | claude-sonnet-4-5-thinking minus gpt-4o | −0.410 | [−0.606, −0.220] | 0.66 | [0.55, 0.80] | detected − |
| RQ1 | openai flagship minus small | 1.196 | [1.016, 1.385] | 3.31 | [2.76, 3.99] | detected + |
| RQ1 | anthropic flagship minus small | 1.197 | [1.045, 1.349] | 3.31 | [2.84, 3.85] | detected + |
| RQ2 | seu_maximizing minus neutral | 0.023 | [−0.092, 0.142] | 1.02 | [0.91, 1.15] | not detected |
| RQ2 | deliberative minus neutral | 0.064 | [−0.052, 0.179] | 1.07 | [0.95, 1.20] | not detected |
| RQ6 | menu-size slope (per added alternative) | 0.075 | [0.053, 0.097] | 1.08 | [1.05, 1.10] | detected + |

### Hiring

| RQ | Contrast | Median (log) | 90% interval | Ratio | Ratio 90% | Decision |
|---|---|---|---|---|---|---|
| RQ1 | gpt-4o-mini minus gpt-4o | −0.671 | [−0.817, −0.527] | 0.51 | [0.44, 0.59] | detected − |
| RQ1 | o3-mini minus gpt-4o | 0.477 | [0.287, 0.669] | 1.61 | [1.33, 1.95] | detected + |
| RQ1 | claude-sonnet-4-5 minus gpt-4o | 0.158 | [−0.017, 0.337] | 1.17 | [0.98, 1.40] | not detected |
| RQ1 | claude-haiku-4-5 minus gpt-4o | −0.736 | [−0.879, −0.595] | 0.48 | [0.42, 0.55] | detected − |
| RQ1 | claude-sonnet-4-5-thinking minus gpt-4o | 0.582 | [0.379, 0.784] | 1.79 | [1.46, 2.19] | detected + |
| RQ1 | openai flagship minus small | 0.671 | [0.527, 0.817] | 1.96 | [1.69, 2.26] | detected + |
| RQ1 | anthropic flagship minus small | 0.897 | [0.740, 1.052] | 2.45 | [2.10, 2.86] | detected + |
| RQ2 | seu_maximizing minus neutral | 0.116 | [0.001, 0.230] | 1.12 | [1.00, 1.26] | not detected |
| RQ2 | deliberative minus neutral | 0.011 | [−0.100, 0.123] | 1.01 | [0.90, 1.13] | not detected |
| RQ6 | menu-size slope (per added alternative) | 0.065 | [0.044, 0.087] | 1.07 | [1.05, 1.09] | detected + |

### Reading the pool results

- **Within-vendor ordering (RQ1).** In both pools the flagship arm is more sensitive than the small
  arm: OpenAI 3.31 [2.76, 3.99] in venture and
  1.96 [1.69, 2.26] in hiring; Anthropic
  3.31 [2.84, 3.85] and
  2.45 [2.10, 2.86].
- **Comparisons with GPT-4o (RQ1).** GPT-4o-mini and Claude Haiku are less sensitive than GPT-4o in
  both pools. The remaining arms depend on the pool: in venture, Sonnet and Sonnet-thinking are less
  sensitive than GPT-4o and o3-mini is not detected; in hiring, o3-mini and Sonnet-thinking are more
  sensitive and Sonnet is not detected. The data therefore do not support a single cross-domain
  sophistication ordering of the reasoning-capable arms.
- **Prompt conditions (RQ2).** No prompt contrast is detected in either pool. In hiring, the
  SEU-maximizing-minus-neutral interval excludes zero, but its ratio,
  1.12 [1.00, 1.26], has a median inside the practical-equivalence
  region; under the frozen rule this is not a detection.
- **Menu size (RQ6).** Sensitivity increases with menu size in both pools:
  1.08 [1.05, 1.10] (venture) and 1.07 [1.05, 1.09] (hiring) per added
  alternative. exp(gamma_size) is a sensitivity multiplier per added alternative, not an accuracy gain
  in probability points, and it is estimated under the independent-choice likelihood (Sections 8 and 10).

## 5. Matched task context (RQ5)

The dedicated 36-cell matched fit pairs 24 procurement and 24 hiring items over 40 exactly paired menus
per task; each contrast is the within-model hiring-minus-procurement difference.

| RQ | Contrast | Median (log) | 90% interval | Ratio | Ratio 90% | Decision |
|---|---|---|---|---|---|---|
| RQ5 | gpt-4o: hiring minus procurement | 0.531 | [0.211, 0.848] | 1.70 | [1.24, 2.34] | detected + |
| RQ5 | gpt-4o-mini: hiring minus procurement | 0.550 | [0.357, 0.750] | 1.73 | [1.43, 2.12] | detected + |
| RQ5 | o3-mini: hiring minus procurement | 0.106 | [−0.263, 0.492] | 1.11 | [0.77, 1.64] | not detected |
| RQ5 | claude-sonnet-4-5: hiring minus procurement | −0.057 | [−0.364, 0.250] | 0.94 | [0.70, 1.28] | not detected |
| RQ5 | claude-haiku-4-5: hiring minus procurement | 1.308 | [1.085, 1.542] | 3.70 | [2.96, 4.67] | detected + |
| RQ5 | claude-sonnet-4-5-thinking: hiring minus procurement | 0.264 | [−0.115, 0.645] | 1.30 | [0.89, 1.91] | not detected |
| RQ6 | menu-size slope (per added alternative) | 0.048 | [0.019, 0.077] | 1.05 | [1.02, 1.08] | not detected |

- Hiring is more sensitive than procurement for GPT-4o (1.70 [1.24, 2.34]),
  GPT-4o-mini (1.73 [1.43, 2.12]) and Claude Haiku
  (3.70 [2.96, 4.67]). The o3-mini, Sonnet and
  Sonnet-thinking contrasts are not detected.
- The matched size slope is a **sensitivity estimate only**, not an additional RQ6 primary decision.
- **Ceiling qualification.** Five matched hiring-task cells, including all three GPT-4o hiring cells,
  have a conditional likelihood whose supremum is at infinite sensitivity (Section 9). Their finite
  posterior upper quantiles reflect prior regularization and hierarchical pooling, not likelihood-only
  bounds. The GPT-4o RQ5 contrast draws on these cells and should be read with that qualification; it
  is also not detected in the presentation-1-only subset (Section 7).

## 6. Descriptive analyses

### RQ3: model-by-prompt heterogeneity

| Group | σ_cell median | σ_cell 90% | Model×prompt DiDs | DiD intervals excluding 0 | Max |DiD median| |
|---|---|---|---|---|---|
| Venture | 0.102 | [0.012, 0.286] | 10 | 0 | 0.184 |
| Hiring | 0.066 | [0.006, 0.193] | 10 | 0 | 0.092 |
| Matched RQ5 | 0.051 | [0.005, 0.166] | 20 | 0 | 0.012 |

Posterior medians of σ_cell range from 0.051 to 0.102, and no
model-by-prompt difference-in-differences 90% interval excludes zero. These descriptive results do not
establish absence of interactions, and positive-support σ_cell intervals excluding zero would not by
themselves establish interactions. RQ3 has no confirmatory decision.

### RQ4: cross-domain comparison (descriptive)

Independent pool posteriors are combined by resampling; draws are never index-paired across fits.

| Contrast | Venture median | Hiring median | Hiring − venture [90%] | P(same sign) | Median signs agree |
|---|---|---|---|---|---|
| gpt 4o mini minus gpt 4o | −1.196 | −0.671 | 0.526 [0.291, 0.763] | 1.000 | yes |
| o3 mini minus gpt 4o | −0.153 | 0.477 | 0.630 [0.351, 0.914] | 0.109 | no |
| claude sonnet 4 5 minus gpt 4o | −0.327 | 0.158 | 0.486 [0.220, 0.756] | 0.073 | no |
| claude haiku 4 5 minus gpt 4o | −1.524 | −0.736 | 0.787 [0.560, 1.016] | 1.000 | yes |
| claude sonnet 4 5 thinking minus gpt 4o | −0.410 | 0.582 | 0.992 [0.713, 1.277] | 0.000 | no |
| openai flagship minus small | 1.196 | 0.671 | −0.526 [−0.763, −0.291] | 1.000 | yes |
| anthropic flagship minus small | 1.197 | 0.897 | −0.300 [−0.518, −0.082] | 1.000 | yes |
| seu maximizing minus neutral | 0.023 | 0.116 | 0.092 [−0.072, 0.255] | 0.616 | yes |
| deliberative minus neutral | 0.064 | 0.011 | −0.054 [−0.211, 0.107] | 0.543 | yes |

Every RQ1 contrast differs between hiring and venture with a 90% interval excluding zero, and the
median signs disagree for o3-mini, Sonnet and Sonnet-thinking relative to GPT-4o. Across the 15
pairwise model orderings, median signs agree in 10 of 15. With only two
deliberately selected pools, this is descriptive evidence about these two domains, not a
between-domain variance estimate or a confirmatory test.

### Amendment 9: Sonnet-thinking minus Sonnet

| Pool | Median (log) | 90% interval | Geometric-mean ratio | Ratio 90% | P(negative) |
|---|---|---|---|---|---|
| Venture | −0.084 | [−0.256, 0.091] | 0.92 | [0.77, 1.09] | 0.786 |
| Hiring | 0.424 | [0.206, 0.637] | 1.53 | [1.23, 1.89] | 0.001 |

Descriptive only, with no detection, ROPE or equivalence classification (Amendment 9): the venture
median is negative and its 90% interval includes zero (−0.084 [−0.256, 0.091]); the hiring contrast is
positive with a 90% interval above zero (0.424 [0.206, 0.637]). These compare configured arms under
their own fixed neutral assessments and differing request settings (including thinking budget and
temperature), not a pure causal effect of reasoning or thinking. All fit variants are shown in
Section 7.

### Amendment 6: assessment-scale transformation

| Group | Rows | Sign reversal | Zero-exclusion changed | Unavailable |
|---|---|---|---|---|
| Venture | 9 | none | none | none |
| Hiring | 9 | none | rq1_claude_sonnet_4_5_minus_gpt_4o | none |
| Matched RQ5 | 6 | rq5_o3_mini_hiring_minus_procurement | rq5_gpt_4o_mini_hiring_minus_procurement | none |

The standardization creates no new decision and does not alter the primary decisions.
All reference SDs are positive; within-model RQ2 offsets cancel exactly (offset 0) and the RQ6 menu-size slope is unchanged in every group. The table flags descriptive sign reversals and changes in interval
zero-exclusion only; they qualify cross-arm comparisons of scale, not the primary estimates.

## 7. Sensitivity analyses

For the primary RQ1/RQ2/RQ5 estimands and menu-size slopes, the tables list rows with a sign,
decision-rule annotation or substantive-interpretation change; the remaining rows of those families
are unchanged. The Amendment 9 descriptive contrast is reported separately for every applicable
variant at the end of this section. Loss or gain of detection in a half-data presentation fit is not
by itself evidence of a between-presentation effect, and independent fits are never paired draw by draw.

### Presentation dependence (Amendment 4)

Venture:

| Contrast | primary | presentation_1_only | presentation_2_only | Changed |
|---|---|---|---|---|
| claude-sonnet-4-5 minus gpt-4o | −0.327 [−0.528, −0.130] detected − | −0.293 [−0.583, −0.017] detected − | −0.188 [−0.451, 0.065] not detected | decision, interpretation |
| claude-sonnet-4-5-thinking minus gpt-4o | −0.410 [−0.606, −0.220] detected − | −0.508 [−0.790, −0.245] detected − | −0.081 [−0.340, 0.176] not detected | decision, interpretation |
| o3-mini minus gpt-4o | −0.153 [−0.364, 0.053] not detected | 0.048 [−0.264, 0.342] not detected | −0.134 [−0.400, 0.124] not detected | sign |
| seu_maximizing minus neutral | 0.023 [−0.092, 0.142] not detected | 0.078 [−0.080, 0.237] not detected | −0.053 [−0.212, 0.105] not detected | sign |

Hiring:

| Contrast | primary | presentation_1_only | presentation_2_only | Changed |
|---|---|---|---|---|
| claude-sonnet-4-5 minus gpt-4o | 0.158 [−0.017, 0.337] not detected | 0.313 [0.070, 0.555] detected + | 0.033 [−0.209, 0.270] not detected | decision, interpretation |

Matched RQ5:

| Contrast | primary | presentation_1_only | presentation_2_only | Changed |
|---|---|---|---|---|
| claude-sonnet-4-5: hiring minus procurement | −0.057 [−0.364, 0.250] not detected | −0.096 [−0.496, 0.290] not detected | 0.073 [−0.375, 0.518] not detected | sign |
| claude-sonnet-4-5-thinking: hiring minus procurement | 0.264 [−0.115, 0.645] not detected | 0.182 [−0.375, 0.732] not detected | 0.455 [−0.028, 0.952] not detected | interpretation |
| gpt-4o: hiring minus procurement | 0.531 [0.211, 0.848] detected + | 0.381 [−0.078, 0.791] not detected | 0.587 [0.231, 0.945] detected + | decision |
| o3-mini: hiring minus procurement | 0.106 [−0.263, 0.492] not detected | −0.036 [−0.604, 0.537] not detected | 0.371 [−0.084, 0.846] not detected | sign, interpretation |

### Utility scale (middle utility 0.35 / 0.50 / 0.65)

Venture:

| Contrast | utility_035 | primary | utility_065 | Changed |
|---|---|---|---|---|
| claude-sonnet-4-5-thinking minus gpt-4o | −0.204 [−0.401, −0.014] not detected | −0.410 [−0.606, −0.220] detected − | −0.590 [−0.789, −0.402] detected − | decision, interpretation |

Hiring:

_No sign, decision or interpretation changes._

Matched RQ5:

| Contrast | utility_035 | primary | utility_065 | Changed |
|---|---|---|---|---|
| claude-sonnet-4-5: hiring minus procurement | 0.057 [−0.247, 0.352] not detected | −0.057 [−0.364, 0.250] not detected | −0.141 [−0.450, 0.165] not detected | sign |
| o3-mini: hiring minus procurement | 0.182 [−0.169, 0.527] not detected | 0.106 [−0.263, 0.492] not detected | 0.235 [−0.135, 0.615] not detected | interpretation |
| menu-size slope (per added alternative) | 0.037 [0.010, 0.065] not detected | 0.048 [0.019, 0.077] not detected | 0.059 [0.029, 0.090] detected + | decision, interpretation |

### Prior sensitivity (Amendment 7)

Priors L (wider common level), H (relaxed between-cell shrinkage) and S (wider size slope) were run on
exactly the primary data for each group. A changed decision is labeled prior-sensitive under these
specified checks; unchanged decisions do not imply unchanged magnitudes, and the checks are not an
exhaustive robustness envelope.

| Group | Prior | Estimands | Max |median shift| | Width ratio range | Decision changed |
|---|---|---|---|---|---|
| Venture | prior_L | 10 | 0.022 | 0.98–1.00 | none |
| Venture | prior_H | 10 | 0.087 | 0.98–1.03 | rq1_o3_mini_minus_gpt_4o |
| Venture | prior_S | 10 | 0.001 | 0.98–1.00 | none |
| Hiring | prior_L | 10 | 0.016 | 0.99–1.01 | none |
| Hiring | prior_H | 10 | 0.018 | 0.97–1.05 | none |
| Hiring | prior_S | 10 | 0.004 | 0.98–1.02 | none |
| Matched RQ5 | prior_L | 7 | 0.023 | 1.00–1.01 | none |
| Matched RQ5 | prior_H | 7 | 0.139 | 1.02–1.62 | rq6_gamma_size |
| Matched RQ5 | prior_S | 7 | 0.003 | 1.00–1.01 | none |

| Group | Contrast | Prior | Primary prior | Alternative prior |
|---|---|---|---|---|
| Venture | o3-mini minus gpt-4o | prior_H | −0.153 [−0.364, 0.053] not detected | −0.232 [−0.445, −0.024] detected − |
| Matched RQ5 | rq6_gamma_size | prior_H | 0.048 [0.019, 0.077] not detected | 0.053 [0.024, 0.084] detected + |

Both prior-sensitive changes occur under prior H and concern estimands that are not detected under the
primary prior: the venture o3-mini-minus-GPT-4o contrast and the matched size slope (itself only a
sensitivity estimate). Under H the matched-RQ5 interval widths grow by up to the ratio shown above,
consistent with relaxed shrinkage.

### Amendment 9 contrast across all variants (descriptive)

Sonnet-thinking minus Sonnet for every primary-pool fit variant. No decision or equivalence
classification applies; the interval column only describes the 90% interval's position relative to 0.

| Pool | Variant | Median (log) | 90% interval | Interval vs 0 | Geometric-mean ratio | Ratio 90% |
|---|---|---|---|---|---|---|
| Venture | primary | −0.084 | [−0.256, 0.091] | includes 0 | 0.92 | [0.77, 1.09] |
| Venture | presentation_1_only | −0.217 | [−0.460, 0.022] | includes 0 | 0.80 | [0.63, 1.02] |
| Venture | presentation_2_only | 0.108 | [−0.139, 0.356] | includes 0 | 1.11 | [0.87, 1.43] |
| Venture | utility_035 | 0.144 | [−0.025, 0.315] | includes 0 | 1.16 | [0.98, 1.37] |
| Venture | utility_065 | −0.269 | [−0.434, −0.102] | below 0 | 0.76 | [0.65, 0.90] |
| Venture | prior_L | −0.082 | [−0.255, 0.089] | includes 0 | 0.92 | [0.77, 1.09] |
| Venture | prior_H | −0.086 | [−0.259, 0.086] | includes 0 | 0.92 | [0.77, 1.09] |
| Venture | prior_S | −0.082 | [−0.251, 0.087] | includes 0 | 0.92 | [0.78, 1.09] |
| Hiring | primary | 0.424 | [0.206, 0.637] | above 0 | 1.53 | [1.23, 1.89] |
| Hiring | presentation_1_only | 0.158 | [−0.137, 0.451] | includes 0 | 1.17 | [0.87, 1.57] |
| Hiring | presentation_2_only | 0.720 | [0.405, 1.037] | above 0 | 2.06 | [1.50, 2.82] |
| Hiring | utility_035 | 0.487 | [0.271, 0.704] | above 0 | 1.63 | [1.31, 2.02] |
| Hiring | utility_065 | 0.389 | [0.182, 0.598] | above 0 | 1.48 | [1.20, 1.82] |
| Hiring | prior_L | 0.423 | [0.210, 0.637] | above 0 | 1.53 | [1.23, 1.89] |
| Hiring | prior_H | 0.443 | [0.224, 0.657] | above 0 | 1.56 | [1.25, 1.93] |
| Hiring | prior_S | 0.423 | [0.209, 0.637] | above 0 | 1.53 | [1.23, 1.89] |

The venture contrast is not stable across variants: its median changes sign in the presentation-2-only
and utility-0.35 fits, and its interval lies below zero at utility 0.65. In hiring, the interval
includes zero in the presentation-1-only fit and lies above zero in the other variants. These
variations are descriptive and carry the same configured-arm qualification as Section 6.

## 8. Posterior predictive checks (Amendment 8, A4)

A4 checks use the saved replicated choices and joint draws of each fit on that fit's retained
observations; they are in-sample, descriptive central-90% review flags, not p-values, discoveries or
a gate. The A4 summaries were computed with the archived batched calculator; for each fit, a seven-draw
probe of that calculator matched the canonical calculator exactly (24 of 24 fits).

**Same-item repetition** across the two presentations of a menu (primary fits, pool or task level):

| Fit group | Pool/task | Complete pairs | Observed same-item | Predictive 90% | Review flag |
|---|---|---|---|---|---|
| Venture | venture | 2509 | 0.8629 | [0.7788, 0.8051] | yes |
| Hiring | hiring | 2512 | 0.8965 | [0.8336, 0.8579] | yes |
| Matched RQ5 | hiring | 719 | 0.9124 | [0.8734, 0.9124] | no |
| Matched RQ5 | venture | 713 | 0.8626 | [0.7461, 0.7966] | yes |

Observed repetition exceeds the predictive range in venture, hiring and the matched procurement task.
Under the independent-choice likelihood this suggests that posterior uncertainty for contrasts drawing
on repeated menus may be understated; the presentation-only fits (Section 7) are the prespecified
response. Same-item repetition is not position invariance or a causal repeated-exposure effect.

**Qualification flags for every fit** (applies, with the number of flagged rows):

| Group | Variant | Trend | Choice pattern | Repetition | Position | Unavailable | Retention |
|---|---|---|---|---|---|---|---|
| Venture | primary | yes (86) | yes (455) | yes (40) | yes (61) | no | yes |
| Venture | presentation_1_only | yes (57) | yes (235) | no | yes (34) | yes | yes |
| Venture | presentation_2_only | yes (59) | yes (251) | no | yes (16) | yes | yes |
| Venture | utility_035 | yes (86) | yes (466) | yes (44) | yes (61) | no | yes |
| Venture | utility_065 | yes (89) | yes (432) | yes (42) | yes (53) | no | yes |
| Venture | prior_L | yes (89) | yes (454) | yes (40) | yes (61) | no | yes |
| Venture | prior_H | yes (87) | yes (453) | yes (40) | yes (61) | no | yes |
| Venture | prior_S | yes (90) | yes (453) | yes (40) | yes (61) | no | yes |
| Hiring | primary | yes (41) | yes (355) | yes (32) | yes (48) | no | yes |
| Hiring | presentation_1_only | yes (32) | yes (175) | no | yes (26) | yes | yes |
| Hiring | presentation_2_only | yes (28) | yes (199) | no | yes (22) | yes | yes |
| Hiring | utility_035 | yes (47) | yes (445) | yes (39) | yes (49) | no | yes |
| Hiring | utility_065 | yes (46) | yes (326) | yes (30) | yes (45) | no | yes |
| Hiring | prior_L | yes (41) | yes (358) | yes (32) | yes (49) | no | yes |
| Hiring | prior_H | yes (42) | yes (361) | yes (34) | yes (48) | no | yes |
| Hiring | prior_S | yes (41) | yes (359) | yes (33) | yes (49) | no | yes |
| Matched RQ5 | primary | yes (39) | yes (259) | yes (21) | yes (40) | no | yes |
| Matched RQ5 | presentation_1_only | yes (33) | yes (177) | no | yes (12) | yes | yes |
| Matched RQ5 | presentation_2_only | yes (28) | yes (133) | no | yes (30) | yes | yes |
| Matched RQ5 | utility_035 | yes (37) | yes (276) | yes (22) | yes (38) | no | yes |
| Matched RQ5 | utility_065 | yes (46) | yes (268) | yes (20) | yes (42) | no | yes |
| Matched RQ5 | prior_L | yes (41) | yes (258) | yes (21) | yes (41) | no | yes |
| Matched RQ5 | prior_H | yes (38) | yes (262) | yes (21) | yes (40) | no | yes |
| Matched RQ5 | prior_S | yes (40) | yes (253) | yes (21) | yes (39) | no | yes |

Choice-pattern, display-position and size-trend discrepancies are flagged in every fit; repetition is
flagged wherever both presentations are retained. Presentation-only fits cannot evaluate repetition by
design ("Unavailable"). Flags qualify interpretation without cancelling decisions; their absence would
not establish model adequacy.

## 9. Ceiling diagnostics (Amendment 7, A3)

Prior-free, full-menu conditional likelihood slices in cell log sensitivity, for the three full-data
midpoint-0.5 datasets:

| Group | Cells | Classifications | Supremum at infinite sensitivity |
|---|---|---|---|
| Venture | 18 | eventual_upper_tail_decay: 18 | none |
| Hiring | 18 | eventual_upper_tail_decay: 18 | none |
| Matched RQ5 | 36 | eventual_upper_tail_decay: 31, finite_supremum_at_infinity: 5 | claude_sonnet_4_5_thinking_deliberative_hiring, gpt_4o_deliberative_hiring, gpt_4o_neutral_hiring, gpt_4o_seu_maximizing_hiring, o3_mini_seu_maximizing_hiring |

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
  the archived primary reports on arm64 matched to within 6.6e-16 (venture) and
  1.7e-14 (hiring) relative difference with identical decisions.
- **Report construction.** On the 16 GiB arm64 machine, the canonical report builder was run with a
  projection-backed fit loader (structural columns copied verbatim into hash-verified diagnostic
  projections of each chain; per-observation arrays read from the full chains) and the archived batched
  A4 calculator, verified per fit by a seven-draw exact probe against the canonical calculator.
  Both substitutions are recorded in the report.

## Provenance

- Complete report: `applications/seu_sensitivity_study/results/complete_report_20261102_v1/complete_report.json`
  (SHA-256 `08350dba614e83c0e42b14d2c3d0753e57666fb672b5e8dc54d49cef7e992eb8`); summary `c5593d0bd3aa6f942c218189b2cfb75fce0038266b8984d66ba0045a589cc99a`; fit manifest
  `46038ef0ec3d6301161ad8c624af9925f5552f61a68bbbefabe50704536508c2`.
- Generator `analysis/study_results_writeup.py` (SHA-256 `de6bc713e68c1a9f9f6962276e2231a1c7efe9fc6f2508396cfaecd1c45c938f`) and template
  `applications/seu_sensitivity_study/RESULTS_WRITEUP_TEMPLATE.md` (SHA-256 `bbecc7c9833df0cbe696f372a68b99e4c6b38362f801a60ea9f6c9d02b057ed9`).
  Regenerate with `python -m analysis.study_results_writeup` from the repository root; the large result
  artifacts are in the private archive releases, not in Git.
- Off-machine copies: private releases `venture-utility-20261102-v1` and `remaining-fits-20261102-v1`
  in `jeffhelzner/seu-sensitivity-archive`, together with the earlier collection and analysis releases.
