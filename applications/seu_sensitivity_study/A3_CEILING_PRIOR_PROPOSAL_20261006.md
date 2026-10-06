# A3 Proposal: Ceiling Diagnostics and Prior Sensitivity

Date: 2026-10-06

Status: **APPROVED by the author, 2026-10-06**, adopted through Amendment 7
of [the preregistration](PREREGISTRATION.md#amendment-7-ceiling-diagnostics-and-prior-sensitivity).
The proposal below was based on revision `30332e2`; its filename is retained
for continuity. Implementation and offline validation are complete. Posterior
fits remain unrun and require separate resource authorization. A4 and the other
outstanding review conditions remain separate; no launch permission is granted.

## Purpose

Distinguish strong consistency with assessed expected utility from precise
measurement of sensitivity. Nearly all choices can maximize assessed expected
utility while leaving large sensitivity values hard to distinguish. Conversely,
a high maximizer fraction can coexist with informative non-maximizing choices.
Neither a 95% maximizer rate nor a well-behaved sampler establishes a ceiling.

Keep the current primary prior and report all primary results under it. Add
likelihood diagnostics and a small, prespecified set of alternative-prior fits
to qualify interpretation, not to select a more favorable primary analysis.

## Observed Choice Diagnostics

For each retained cell in each of the three full-data, midpoint-0.5 datasets
(venture, hiring, matched RQ5), record:

- Resolved observation count, distinct menu-ID count, and exclusions, with
  counts by menu size and presentation. Repeated presentations are observations,
  not additional distinct menus; repeated compositions retain their menu IDs.
- Choices of exact expected-utility maximizers, choices below the maximum,
  number of tied maximizers per menu, and completely equal-utility menus.
- Per-observation regret, defined as maximum menu eta minus chosen eta, and
  top-two eta gap. Save individual values and minimum, 5th percentile, median,
  95th percentile, and maximum by size. Empty groups have no summary, not zero.
- Repeat maximizer/tie counts using absolute eta tolerance 1e-12 and relative
  tolerance zero, explicitly labeled as numerical near-tie summaries. Do not
  round or replace eta in the likelihood. The exact and near-tie counts can
  differ and must not be silently interchanged.

An entirely tied menu contributes no information about sensitivity. A tie among
the best alternatives does not make the whole menu uninformative if other
alternatives have lower eta. Do not use the top-two gap alone as an information
measure. No count or rate above is a pass/fail gate or a refit trigger.

Use the retained Stan input and its bound observation/item mappings, not the
unfiltered reference menus, for likelihood diagnostics. Amendment 6's fixed
reference geometry remains a different, complementary report.

## Likelihood Information

Let `t` denote a cell's log sensitivity at the fitted size center, and `b` the
shared size slope. For observation `m` in the cell, define

$$
a_m=\exp(t+b s_m),\qquad
\ell_j(t;b)=\sum_{m\in j}\left[
a_m\eta_{m,y_m}-\operatorname{logsumexp}_{r\in A_m}(a_m\eta_{m,r})
\right].
$$

Evaluate this likelihood without any prior or hierarchical penalty. Subtract
the menu maximum before multiplying eta by sensitivity for numerical stability.
Keep every alternative, not just the top two. The other cells' levels do not
enter this cell likelihood; the shared slope still does.

Report conditional curves at the primary posterior's 5th percentile, median,
and 95th percentile of `b`. These are **conditional likelihood slices**, not
profile likelihoods, confidence intervals, or a test of joint identification.
They do not explore every possible slope and cannot prove that nuisance
parameter uncertainty is harmless.

For each slice use 401 equally spaced `t` values on
`[min(-5, q05(t)-2), max(10, q99(t)+2)]`, and also evaluate at the posterior
5th, 50th, 95th, and 99th percentiles of `t` and at `q95(t)+log(2)`.
Record the grid endpoints, the grid maximum, endpoint-max flags, and log
likelihood differences from that grid maximum. If an added evaluation exceeds
the grid maximum, refine the grid to include that location and report the
refinement. The upper grid endpoint is not an estimated upper bound. Nonfinite
evaluations make the diagnostic unresolved; they are not evidence of a ceiling.

Provide the exact limiting classification for each nonempty cell at any fixed,
finite `b`:

- If all menus have equal eta across alternatives, the likelihood is constant
  in `t`.
- If every chosen item is an exact maximizer and at least one menu has unequal
  eta, the likelihood increases toward a finite supremum as `t` tends to
  infinity. Its limit is minus the sum of log counts of exact maximizers.
- If at least one chosen item is strictly below its menu maximum, the log
  likelihood tends to minus infinity as `t` tends to infinity. This establishes
  eventual upper-tail decay at fixed `b`, not useful finite-sample precision.

For the second case, report the difference between the analytic supremum and
the likelihood at each posterior percentile. For all cases, report the signed
change from `q95(t)` to `q95(t)+log(2)` on each slice. These are continuous
descriptions; do not invent a calibrated "flat enough" cutoff or label the
slice as a likelihood-ratio confidence set. Tiny nonzero regrets can produce
eventual decay far beyond the plotted range.

All statements condition on the saved eta and the working independent-choice
likelihood. They neither assess uncertainty in elicited probabilities nor
correct repeated-presentation dependence. Existing presentation-specific fits
remain relevant to that separate limitation.

## Proposed Prior Variants

Normal distribution parameters below are mean and standard deviation. HN is
the half-normal distribution on a positive scale parameter. All priors retain
the current coding, independence assumptions, and noncentered parameterization.

| Variant | gamma0 | Each gamma coefficient | sigma_cell | gamma_size |
|---|---|---|---|---|
| Primary, unchanged | N(2.5, 0.5) | N(0, 0.5) | HN(0.3) | N(0, 0.2) |
| L: wider common level | N(2.5, 1.0) | N(0, 0.5) | HN(0.3) | N(0, 0.2) |
| H: less heterogeneity shrinkage | N(2.5, 0.5) | N(0, 1.0) | HN(0.6) | N(0, 0.2) |
| S: wider size slope | N(2.5, 0.5) | N(0, 0.5) | HN(0.3) | N(0, 0.4) |

Keep every `z_alpha` prior N(0,1). L tests the regularization of a high common
level; the median of `exp(gamma0)` stays about 12.2 while its central 90% prior
interval widens from about [5.35, 27.7] to [2.35, 63.1]. These are intervals for
the intercept, not marginal intervals for cell sensitivities. H jointly relaxes
the two sources of between-cell shrinkage relevant to realized-cell contrasts;
it cannot attribute a change uniquely to gamma or sigma_cell. S tests direct
regularization of RQ6 and its coupling to cell levels.

The twofold SD changes are transparent, bounded stress choices, not uniquely
correct alternative beliefs or an exhaustive robustness envelope. They retain
coding-induced prior asymmetries, including those in matched RQ5. Stability
under them does not establish robustness to recoding, heavy-tailed priors, or
joint changes to all prior components.

As an adoption prerequisite, validate the induced distributions of cell log sensitivities,
realized contrasts, and choice probabilities using prior draws on the frozen
design, including matched coding and sizes 2/4/6/8. This is an offline
prior-predictive check, not a posterior fit or new provider collection. Any
revision of the proposed values returns for approval before production choices
are inspected; do not tune priors to obtain a desired result.

Under the approved policy, run L, H, and S for **all three full-data midpoint-0.5 datasets**,
not just cells with high maximizer rates. That is nine additional posterior
fits, increasing the planned total from 15 to 24. Do not cross these variants
with utility or presentation variants. Use exactly the corresponding primary
data, eta, cell order, design matrix, size centering, and exclusion decisions.

These fits require separate resource authorization. Benchmark one authorized
fit before scheduling the remainder; do not infer a runtime budget from fit
count alone. Retain the existing sampler acceptance checks and durable chain
artifacts. Failed or unrun variants are reported as incomplete sensitivity
assessment, never silently omitted or interpreted as stability.

## Comparison and Interpretation

For each prior variant, report cell log sensitivity and sensitivity quantiles
(5%, 50%, 95%, 99%), the shared size slope, and the Amendment 5 realized
contrasts. Use the same fixed cell weights and missing-cell policy. Show each
contrast's median, central 90% interval, sign probability, median shift from
primary, interval-endpoint shifts, and interval-width ratio. Do not pair draws
across distinct fits to manufacture a posterior for their difference.

For RQ1/RQ2/RQ5/RQ6, apply the existing interval-plus-practical-effect rule to
the alternative-prior summaries solely as a sensitivity annotation of each
named primary result. Record median sign changes, zero-exclusion changes, and
decision-rule changes separately. This adds no primary hypotheses, discoveries,
or votes; the primary family remains 26 named decisions, 24 distinct up to sign.
RQ3 residual-scale summaries and RQ4 independent cross-pool contrasts/orderings
remain descriptive, with RQ4 combining pools within the same prior variant.
Amendment 6's standardized outputs remain primary-prior only; there is no new
prior-by-assessment-scale reporting grid in this proposal.

Use the following interpretation rules:

- A primary decision that changes under any valid alternative is labeled
  **prior-sensitive under the specified checks**. Keep the primary result
  visible but do not describe its conclusion as robust to the tested priors.
- An unchanged decision with changing magnitude or upper tail still requires
  quantitative disclosure. "Same decision" does not mean "same estimate."
- A cell with a likelihood supremum only at infinite sensitivity has no finite
  likelihood maximizer at fixed slope. Its finite posterior upper quantile
  reflects prior regularization and hierarchical pooling, not a likelihood-only
  upper bound. Report the full interval and this qualification; do not replace
  it with an uncalibrated one-sided bound.
- Do not automatically discard a contrast because one contributing cell has
  this property. Assess the contrast's actual joint posterior and sensitivity;
  differences can be better or worse constrained than individual levels.
- If choice predictions barely move while sensitivity estimates move, describe
  predictive stability alongside parameter sensitivity. Neither resolves A4
  model adequacy. Sampler failure is computational uncertainty, not evidence
  for or against a substantive ceiling.

No automatic primary-result suppression, replacement prior, numerical ceiling
gate, or launch clearance follows from these diagnostics.

## Implementation Acceptance Checks

Before adoption is considered complete, require tests for exact and near ties;
all-equal menus; all-maximizer monotone likelihoods and tied-maximizer limits;
one non-maximizer causing eventual decay; tiny positive regrets; correct y/item
mapping; exclusion and presentation counts; cell reordering; size-slope use;
stable extreme-logit evaluation; and unresolved numerical cases.

Validate the likelihood against direct softmax calculations on small examples.
Verify primary settings reproduce the current likelihood and priors, while
each alternative changes only its declared prior components. Test unchanged
primary decisions, separately labeled sensitivity annotations, missing fits,
invalid sampler diagnostics, and hash-bound data/prior provenance. Prior draws
must verify the realized, not merely additive, contrast distributions.

Approval recorded: the author approved this diagnostic/reporting policy and
the L/H/S prior specification on 2026-10-06. The offline prior-predictive check
and implementation tests are recorded in Amendment 7. Approval does not
authorize posterior fits or production calls.