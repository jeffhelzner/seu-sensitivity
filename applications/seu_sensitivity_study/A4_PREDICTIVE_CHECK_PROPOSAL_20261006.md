# A4 Proposal: Predictive Adequacy and Size-Effect Interpretation

Date: 2026-10-06

Status: **APPROVED by the author, 2026-10-06**, adopted through
[Amendment 8](PREREGISTRATION.md#amendment-8-expanded-predictive-checks).
The specification was based on revision `3bb8be1`; the filename is retained
for continuity. Implementation and offline checks are complete. This addresses
A4, the diagnostic parts of B1 and B6, and the encoded-position issue C5.
No fitting or collection is authorized; other review conditions remain open.

## Recommendation and Scope

Extend the existing posterior predictive report using saved `y_pred` and
parameter draws, fixed eta, and the exact retained observations of each fit.
No new model or posterior fit is required. The planned total remains 24 and
the primary family remains 26 named decisions, 24 distinct up to sign.

Run the expanded checks for every supplied, sampler-valid variant. Emphasize
the three full-data primary-prior fits in the main report. Utility variants
use their own eta; presentation variants use their own retained observations.
Prior variants use exactly the corresponding primary observations. Missing or
invalid variants remain explicitly incomplete under Amendment 7, and comparisons
across variants must disclose different observation sets. Do not choose a
preferred fit by which checks look most favorable.

Replicated choices are independent conditional on each joint parameter draw,
as in the fitted likelihood. Preserve the same draw across observations and
statistics, including both presentations of a menu. Do not pair different
posterior iterations or fit groups to construct a replicated dataset. These
are in-sample checks under the working likelihood, not held-out forecasts,
calibrated hypothesis tests, or evidence that elicited beliefs are correct.

## Evidence and Grouping

Use Amendment 7's complete retained/excluded observation universe and frozen
presentation-order evidence. Add hash-bound canonical item quality labels,
menu family and difficulty stratum, and the frozen composition recipe needed
to establish authored roles. Validate these against the fixed pool/menu
artifacts. Missing or contradictory role evidence is an input error, not a
reason to infer roles from assessed eta or displayed positions.

Within each fit, report the following fixed partitions:

- Each cell overall, by size, and by family x authored stratum x size.
- Each pool/task overall, by size, and by family x authored stratum x size.
  The matched fit keeps procurement and hiring separate rather than combining
  them into a single task average.
- Display-position summaries additionally separate presentation 1 and 2.
- Item summaries use cell x item and pool/task x item, as defined below.
- Paired summaries use cell and pool/task, overall and by family x stratum x
  size, restricted to pairs with both presentations retained.

Pool/task aggregates weight retained observations equally, not cells equally.
Show observation, distinct menu, and contributing-cell counts, and never
interpret such aggregates as Amendment 5's equally weighted cell estimands.
Use the identical retained mask for observed and replicated calculations.
Emit empty groups as unavailable with count zero, not a fabricated zero rate.
Do not suppress small groups; their counts and discrete predictive intervals
show the limitation. Repeated compositions retain their distinct menu IDs.

## Statistics

For each basic observation grouping, retain the current exact-maximizer
selection fraction, mean selected-choice probability, and mean log score, and
add mean regret and authored-filler selection fraction. For regret,
`max(eta in menu) - eta of selected item` is evaluated identically for observed
and replicated choices. Scores and selected-choice probabilities depend on
the parameter draw; regret, roles, and exact maxima do not.

### Authored Fillers

Under the frozen variant-D recipes, two contenders have strong or ambiguous
authored labels, and the other `size - 2` alternatives have weak labels. Verify
the full recipe, including contender label multiplicities, before defining
the filler set. A filler can be an assessed maximizer: report its selection as
filler selection nonetheless. Do not relabel it a contender using eta ranks.

The denominator is every retained observation in the group, not the number
of fillers. Record the number of observations containing fillers as well.
Size-2 menus have no fillers, so their rate is structurally zero in both
observed and replicated data, explicitly labeled as such. Their apparent
agreement supplies no evidence for reproducing filler avoidance. The model
probability of selecting a filler is the sum of probabilities over that set.

### Item Shares

For every item in the applicable fixed pool or matched task, record its
exposure count, observed selection count, and replicated selection-count
distribution. Exposure means a retained observation in which the item was
available, including separate presentations and cells where applicable.

Report selection count divided by exposure count as an exposure-conditional
selection rate. Also report selection count divided by all retained choices
in that pool/task or cell as the unconditional choice share. The former rates
need not sum to one; the latter shares do. Zero-exposure items have unavailable
conditional rates, but zero unconditional share if the group has observations.
Do not treat these as preference estimates controlling for competitors: the
model's own replicated distribution conditions on the actual menus.

### Display Position

Map each observed and replicated sorted-active-set choice index to item ID,
then to its position in the bound frozen presentation. Report position counts
and fractions within menu size and presentation, by cell and pool/task. Do not
pool raw positions across different sizes. The original sorted-index summary
may remain in machine-readable output as a legacy encoding diagnostic but
must not be presented as evidence about display-position bias.

### Paired Presentations

Match only `(cell_id, problem_id)` with presentations 1 and 2 both retained.
Never pair across cells, tasks, or distinct menus with identical composition.
Give counts of complete pairs, single retained presentations, and pairs with
neither retained, relative to the eligible frozen design. A presentation-only
fit has paired checks unavailable by design, not a zero agreement rate.

For complete pairs, calculate the fraction choosing the same item and the
fraction choosing the same displayed position. Derive each replicated pair
from its two saved `y_pred` entries within the same posterior draw. Under
reversal with even menu sizes, the two events are mutually exclusive. Do not
call same-item agreement position invariance or call its excess a causal
effect of repeated exposure: shared unmodeled item preferences and other
misspecification can also produce discrepancies.

For numerical cross-checking, the conditional same-item agreement probability
is the sum over items of the product of the two presentation probabilities.
Same-position probability instead sums products for the items occupying each
common displayed position. Average these conditional expectations over the
eligible pairs, and report their posterior summaries alongside the replicated
fractions. Conditional means are not substitutes for predictive intervals.

## Size-Pattern Checks

Per-size predictive distributions are necessary but may obscure a consistent
direction of discrepancy. For maximizer fraction, mean regret, and filler
fraction, add a fixed linear size-trend summary within each cell/family/stratum
and pool/task/family/stratum having observations at all four sizes.

For the four size-specific rates or means `T_n`, define

$$
B(T)=\frac{\sum_{n\in\{2,4,6,8\}}(n-5)T_n}
{\sum_{n\in\{2,4,6,8\}}(n-5)^2}.
$$

Sizes receive equal weight, and the identical calculation is applied to each
replicated dataset. Report individual size summaries too: this single linear
summary cannot detect every nonlinear discrepancy. If any size has no
retained observations, the four-size trend is unavailable without reweighting,
imputation, or dropping that size. The structural zero at size 2 is retained
in the filler trend and is modeled identically in replication.

This is a diagnostic trend in choice behavior, not an estimator of gamma_size
and not another confirmatory size-effect hypothesis. Expected changes in
choice rates with size already include menu composition, all eta gaps, and
the fitted slope. Do not compare observed choice rates to a flat line or
interpret any size trend as proof of a nonzero sensitivity slope.

## Summaries and Review Flags

For each scalar statistic save its observed value and replicated 5th, 50th,
and 95th percentiles, plus observed-minus-replicated discrepancies. For fixed
observed statistics, report the signed difference from the predictive median
and the central 90% interval of that discrepancy. For draw-dependent scores,
save observed-score and replicated-score summaries separately and summarize
their within-draw difference. Do not compare an observed score evaluated at
one parameter estimate with replicas evaluated at other draws.

Report the three empirical probabilities `T_rep < T_obs`, `T_rep == T_obs`,
and `T_rep > T_obs`, using the matched draw for draw-dependent statistics.
Keep equality separate because discrete ties are common. Use exact eta ties
for maximizer checks, retaining A3's separately labeled near-tie geometry;
do not silently switch the predictive maximizer definition.

A fixed observed statistic strictly outside its central 90% predictive
interval receives a **descriptive review flag**. For draw-dependent scores,
flag only when the central 90% within-draw discrepancy interval excludes zero.
These flags identify rows to discuss; they are not calibrated p-values,
additional discoveries, an omnibus test, or a multiplicity-controlled gate.
Report the full prespecified table, including unflagged rows and counts.
Many dependent comparisons can generate flags even under an adequate model;
do not count flags as independent failures or search for a threshold afterward.

## Missingness and RQ6 Interpretation

Before interpreting predictive agreement, report eligible, resolved, retained,
unresolved, and whole-cell-excluded counts by cell, family, authored stratum,
size, and presentation. Keep resolution failures separate from resolved rows
removed with their cell; exclusion totals must reconcile without double counting.
Record complete-pair availability. These are observed missingness descriptions,
not predictive checks: the choice model does not model failure to resolve.

For each cell and pool/task, show the four size-specific unresolved fractions
and their maximum-minus-minimum range, with original eligible denominators.
Missing denominators are errors in frozen-design evidence, not zero failure
rates. No numerical missingness-triggered imputation or extra fit is introduced.
Any unresolved observations make the predictive assessment conditional on
retention; whole-cell exclusions and absent size strata limit its scope further.

Preserve the primary RQ6 estimate and decision, but attach a structured
interpretation record with these nonexclusive qualifications:

- A flagged four-size behavioral trend requires explicit disclosure that the
  fitted model does not reproduce that trend at the stated descriptive band.
  Do not present gamma_size as a complete explanation of size-related choices.
- Other per-size, filler, or item discrepancies require reporting their pattern
  and counts. Do not infer a specific omitted mechanism from one flagged row.
- Excess same-item repetition requires qualification of the independent-choice
  assumption and a comparison with the existing presentation-specific fits.
  An interval including zero in a half-sized fit alone is not proof of failure;
  report median and interval changes as well.
- Display-position discrepancies qualify the position-insensitive model, with
  emphasis on their size and presentation patterns rather than a causal claim.
- Absent sizes or pairs leave the corresponding diagnostic unavailable.
  Differential missingness limits the retained-choice interpretation even if
  predictive checks look satisfactory. Do not equate missingness with equivalence
  or certify robustness to missing-not-at-random choices.
- Passing the sampler gates, lack of review flags, or predictive agreement on
  these summaries does not establish model truth, precise upper-tail sensitivity,
  or unrestricted behavioral interpretation. Read A3 and A4 jointly.

There is no automatic cancellation of a primary decision, replacement fit,
or predictive launch pass. Any later model revision is a separately documented
analysis, not a silent repair of the preregistered model. The main report must
present qualifications alongside RQ6 rather than only in a diagnostic appendix.

## Implementation Acceptance Checks

Before adoption is considered implemented, test hand-computable observed and
replicated statistics, exact ties and all-tied menus, a filler that maximizes
eta, size-2 structural zero, unequal exposures and zero-exposure items, matched
task support, cell/item permutations, frozen display-order mapping, and
incomplete pairs. Test same-item versus same-position behavior under reversal,
and conditional agreement probabilities against explicit enumeration.

Test equal-size trend weights, missing-size unavailability, nonzero modeled
size trends, and a constructed size-pattern mismatch hidden by pooled averages.
Test complete denominators, exclusion reconciliation, tampered role metadata,
discrete equality tails, zero-width predictive intervals, and draw-dependent
discrepancy summaries. Preserve primary decisions and the 24-fit plan exactly.
Reuse saved replicas without new RNG draws; record source bindings and report
unavailable checks explicitly. Validate serialization and report size/runtime
on a representative offline fixture without running posterior fits.

Approval recorded: the author approved these descriptive checks and
interpretation rules on 2026-10-06. Implementation and offline validation are
recorded in Amendment 8. No new fits, model changes, provider calls, or
production authorization are included.