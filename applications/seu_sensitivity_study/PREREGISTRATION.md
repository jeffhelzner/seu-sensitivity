# SEU Sensitivity Study Preregistration

Frozen: 2026-09-06

> Current interpretation (2026-10-08): read the dated clarification and amendments before using
> the historical body below. Amendment 1 replaces the latent-belief primary
> model and the RQ4 variance-component interpretation. Amendment 2 changes the
> size-slope prior. Amendments 3 and 4 specify the decision contract and required
> sensitivity fits. Amendment 5 makes realized-cell contrasts primary for
> RQ1/RQ2/RQ5 and retains additive coefficients as descriptive companions.
> Amendment 6 specifies descriptive assessment-scale transformations without
> refitting or adding confirmatory decisions.
> Amendment 7 adds ceiling diagnostics and nine prior-sensitivity fits, for
> 24 planned fits; the primary priors and 26 named decisions remain unchanged.
> Amendment 8 expands descriptive predictive checks and RQ6 qualifications
> using saved replicas, without adding fits or primary decisions.
> Amendment 9 adds a descriptive Sonnet-thinking minus Sonnet comparison
> within each primary pool, without adding fits or primary decisions.
> The fixed midpoint is a substantive utility assumption, not merely a choice
> of units. Independent review 2 and author approval establish **scientifically
> ready for final preparation and preflight**, not operational readiness or
> authorization to collect. The historical recovery-provenance limitation is
> accepted with disclosure, without remediation.
>
> Amendment 4's phrases "direct production-geometry calibration" and "consistent
> with nominal coverage" must be read in the narrower scope established by the
> October 7 verification: recovery under the generating model with 40 coupled
> seed clusters and dependent contrast cases, not a formal calibration claim.
> Review 2 accepted the disclosed historical choice-input provenance limitation
> on October 8; it did not establish exact replay or authorize collection.

## Approved Review 2 Clarification (2026-10-08)

The author approved all proposals in the [October 8 disposition](REVIEW_DISPOSITION_20261008.md)
and selected the existing arm-wide **STOP** rule alone, with no descriptive
available-cell companion. This dated clarification controls current interpretation;
the historical body and amendment sections below, the October 5 disposition,
and the September evidence snapshot remain unchanged. Historical **NOT READY**
statements describe their dated checkpoints, not a reopened scientific review.

**Missing-cell cascade.** Excluding GPT-4o's neutral cell makes eight of a
pool's ten named primary decisions unavailable: six GPT-4o-involving RQ1
decisions and both RQ2 decisions. Excluding its SEU-instruction or deliberative
cell makes seven unavailable: the same six RQ1 decisions and the corresponding
RQ2 decision. Provided the retained design stays full rank and other required
checks pass, the Anthropic flagship-minus-small comparison and RQ6 remain
available, as does the unaffected RQ2 comparison in the latter cases.
Excluding an entire arm makes the original design rank deficient and stops
that pool's analysis. No reweighting, additive fallback, reduced-model fit,
or available-cell companion is adopted.

**Fixed assessment-scale offsets.** The separately dated supplement is specified
at [offset evidence](../../reports/applications/seu_sensitivity_study/data/offset_evidence.json).
The supplement is published and its exact offline reproduction passed, as
recorded in the October 8 disposition. RQ1 adds
`log(S_a/S_b)` and RQ5 adds `log(S_hiring/S_procurement)`, where S is the
population SD of normalized assessment-derived expected utilities at midpoint
0.5 over all 60 primary-pool items or each matched task's own 24 items.
RQ2 offsets cancel algebraically because prompt siblings share the anchor.
Negative RQ5 offsets shift a contrast downward; they do not determine the sign
of either production contrast. These are descriptive changes of scale, not
new choice findings or decisions. Sonnet-thinking-minus-Sonnet offsets are
fixed-input context only, not a new standardized Amendment 9 posterior analysis.

**Dependence reporting.** An existing excess same-item repetition flag in
contributing cells also qualifies the affected RQ1/RQ2/RQ5 reporting rows and
is read alongside their existing presentation-only estimates. It raises concern
about the independent-observation uncertainty model without identifying a cause
or changing any interval, threshold, or primary detection decision. An
unavailable paired diagnostic is not evidence of no dependence. Use existing
flags and contributing-cell mappings; add no flag computation, aggregate test,
refit, gate, or requirement that both presentation-only intervals exclude zero.
Reporting code now attaches this conditional text and references to existing
diagnostics and presentation comparisons; no production diagnostic result is claimed.

**RQ2 interpretation.** Because the choice prompts display the assessed
probabilities, an SEU-instruction effect may reflect following an explicit
expected-utility calculation rather than greater coherence between beliefs and
actions beyond that displayed task. Prompts and the estimand are unchanged.

**Finding 5 prerequisite: NOT RUN; separately authorized.** Before L/H/S
posterior fits, verify primary/prior-sibling log-density equivalence up to a
constant at primary prior settings on shared unconstrained points and
production-shaped inputs. Align parameter order, transformations, Jacobian
settings and density-constant conventions; declare numerical tolerances.
Check deterministic generated quantities and implied choice probabilities;
random replicated choices require distributional consistency, not presumed
bit-for-bit equality. This is required verification before the existing prior
fits, not a scientific blocker before collection or an additional scientific
gate. No compilation, density evaluation, generated-quantity execution or fit
is authorized by this clarification.

**Provenance and current status.** Review 2 accepts the historical evidence as
saved-draw recovery under the working model at production geometry and an
indication of precision in those settings, not formal calibration or production
adequacy. The 120 iterations represent 40 coupled seed clusters with dependent
contrast cases. Original choice inputs remain missing; exact seed-to-choice
replay and replacement-input equality remain unverified. No remediation is
requested or planned for this accepted limitation. The artifact-bound machine
contract still requires provenance acceptance and its source is unchanged.
This dated clarification and the October 8 disposition externally supersede
the pending human-acceptance status only; they do not change verification
fields, hashes, source bindings, or executable checks.

The family remains 26 named decisions, 24 distinct up to sign; the fit plan
remains 15 base fits plus nine L/H/S fits, not a crossed grid. Scientific
readiness does not authorize a clean commit, fresh staging, final-revision
preflight, a runtime benchmark, posterior fits, provider calls, spending or
collection. Each remains separately authorized and outstanding for the final
revision; none was run or authorized in this documentation update. A benchmark
that fits a model requires fit authorization. Existing historical runs and
budget approvals remain historical, not authorization for these next steps.

## Historical Preregistration Body

This document freezes the production design and confirmatory decision rules.
The executable design is
`applications/seu_sensitivity_study/configs/preregistered.yaml`.

## Scope and design

- Estimation pools: venture and hiring. Insurance is excluded because its
  embedding-based predictive-validity R-squared was 0.052, below the frozen 0.30
  threshold. This is a failed screening criterion, not proof that alpha is
  structurally unidentified; the latent eta/alpha invariance is a separate
  identification issue addressed by Amendment 1.
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
without and with extended thinking. The reasoning arm uses a 4,096-token
thinking budget, a 4,160-token total output cap, and temperature 1; the
thinking-off baseline uses temperature 0. This contrast is a treatment bundle,
not a pure causal effect of thinking. The OpenAI flagship/reasoning comparison
uses different endpoints. These vendor-specific reasoning contrasts are
different estimands and will not be pooled.

## Collection budget

Production choice collection will use provider Batch APIs only after an E4
provider-specific dry run passes. The approved Batch ceiling is $122.78. The
synchronous fallback ceiling is $61 and requires separate approval before use.
API usage records, including input, output, cached, and thinking tokens, must be
persisted separately from the phase-local `run_summary.json`.

The historical expected choice cost was $24.17 at Batch rates, with a stated
25% contingency; this estimate is not a maximum-liability calculation. The
original six-fit schedule and 50.0-serial-hour envelope do not cover the amended
campaign. There are now 15 mandatory fits: venture, hiring, and matched RQ5,
each with primary, presentation-1-only, presentation-2-only, and full-data
0.35/0.65 utility variants. The position-stable subset remains a separate
preregistered robustness analysis. No updated total compute envelope is claimed;
the 12-hour stop applies to every fit.

Before each provider Batch submission, the runner must durably reserve that
attempt's conservative liability in an append-only ledger. For each rendered
request, the input allowance is its UTF-8 message-content byte count plus 1,024
per message plus 1,024 request overhead; the output allowance uses the exact
provider output cap, including reasoning where applicable. Input and output
allowances are priced at the pinned model rates with the 0.5 Batch multiplier.
The `$31 / 10,080` per-request amount is only a reservation floor, not the
liability estimate. This conservative allowance is a conditional bound under
the stated pricing and protocol assumptions, not an absolute provider billing
guarantee. The verified 2026-09-11 offline audit rendered all 10,080 requests:
the actual reservation total is $122.7729167015873 ($118.18225175 before the
floor), and the maximum output-only amount is $61.867008, not a forecast of
actual spend. Exact inputs, per-arm caps, pinned rates, and reservations are
recorded in [the E4 offline audit](PHASE_E4_VALIDATION.md#verified-offline-audit-2026-09-11).
Reservations are keyed by the durable submission intent and are never silently
released after failure, ambiguity, or completion. Both audited amounts exceeded
the historical $31 ceiling. The 2026-09-11 budget approval below resolves that
blocker under the current conditional liability assumptions without changing
the token limits or design; the audit itself authorized no spending. Any retry,
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

The ridge leave-one-out formula now includes the unpenalized intercept. The
2026-09-11 pure `item_validation.run_gate` recomputation used saved inputs and
fresh sibling summaries in memory: pooled LOO R-squared is 0.8232 (venture)
and 0.6986 (hiring), worst cross-size gaps are 0.0906 and 0.0484, and the
cross-pool gap is 0.0601. Both pass the frozen 0.30 R-squared and 0.25 gap
thresholds, with LDA fallback in both pools and zero assessment parse failures.
This is verified offline evidence, not production preflight. Saved historical
gate reports remain stale; fresh production staging, its manifest, and
production-wave preflight remain pending. No production wave is authorized;
status is **NOT READY**. These documentation corrections involved no provider
calls, environment creation, or production staging/preflight. The preceding
offline audit involved no new fits; the separately authorized venture
iteration-16 rerun completed on 2026-09-11 is recorded in Amendment 4 below.

### Budget approval (2026-09-11)

The user said: "ok, i give my approval on the budget. let's continue".
The current `configs/preregistered.yaml` sets `batch_choice_budget_usd` to
**122.78**, up from the historical 31: the calculated $122.7729167015873
reservation rounded up to cents for the frozen **10,080-request campaign**.
The budget blocker is resolved under the current conditional pricing/protocol
liability assumptions, not as a provider billing guarantee or authorization
for replacement attempts. The flat per-request floor remains
`0.0030753968253968253`; design and token limits are unchanged.
`batch_wave_id: null` and `batch_wave_cell_ids: []` remain unchanged.
No calls were launched for this approval update. The worktree is dirty and
the user has not authorized a Git commit. A clean committed revision, fresh
production staging and manifest, separately authorized preflight, and explicit
wave and production launch authorization remain pending. Status remains
**NOT READY**; budget approval alone does not authorize collection.

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

Including one RQ6 decision per pool and six matched RQ5 decisions, the total
primary decision count is `2 * (9 + 1) + 6 = 26`. Sensitivity results and
descriptive RQ3/RQ4 summaries do not create additional confirmatory hypotheses.

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
`sigma_cell`, the raw `sigma_cell * z_alpha` cell residuals, and derived
model-by-prompt difference-in-differences from the existing anchored fit, without a confirmatory
interaction decision. The generated analysis contract records and verifies the
rank identity against the production design matrix.

RQ4 remains descriptive: posterior differences use independent pool posteriors,
not index-paired chain draws. Reports include all 15 model-pair rankings and
same-sign probabilities across the two pools, without adding confirmatory
hypotheses or treating `sigma_cell` as cross-pool variance.

The confirmatory RQ5 estimand is the within-model
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
Stan data. After whole-cell NA exclusion, every anchored payload, including
presentation-only and matched RQ5 variants, must retain full rank in its
intercept-plus-design matrix. Failure stops analysis rather than silently
changing the confirmatory family. Each variant has its own assembly report
recording retained `cell_ids`, `design_columns`, `rank`, and `presentation_id`.
Reporting requires the strict schema-version-1, 15-entry fit manifest documented
in `PHASE_E4_VALIDATION.md`, all four chain hashes per fit, and hash-bound Stan
data, preparation report, and analysis contract. It checks finite structural
values, actual four-chain/treedepth-12 metadata, posterior consistency, and
posterior predictive summaries. These bindings declare the inputs associated
with a fit; they do not prove that CmdStan executed with those inputs.

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
recovery campaigns for venture, hiring, and matched RQ5 completed with
deterministic longer reruns. The historical before-rerun saved-summary audit
on 2026-09-11 found 39/40 venture, 40/40 hiring, and 40/40 matched RQ5:
venture iteration 16 had `sigma_cell` tail ESS 310.525, below 400. That
119/120 result was genuine, and the earlier all-pass claim was false at that
checkpoint. Runtime CSV headers sampled in each group verified the actual
`h_m01_size_assessment_anchored_model` name and maximum depth 12. All 120
fits then had zero divergences and no treedepth saturation. The central 90%
coverage results recorded in E4 are historical before-rerun results,
consistent with nominal coverage at the available Monte Carlo resolution;
they have not been recomputed for this update.

Update 2026-09-11: the separately authorized venture iteration-16 rerun
completed with four chains, 1,000 warmup and 1,000 retained sampling draws per
chain, and maximum treedepth 12 using the existing
`configs/h_m01_size_assessment_anchored_venture_recovery_rerun_config.json`.
Verified `_summarize_sampler_diagnostics` results now give **40/40 venture,
40/40 hiring, and 40/40 matched RQ5: 120/120 pass all frozen sampler gates**.
This closes the recovery shortfall. The rerun's minimum structural tail ESS
is 1,960.77; full diagnostics, timing, archived originals, and numerical-truth
and unchanged-artifact checks are recorded in the
[dated E4 completion record](PHASE_E4_VALIDATION.md#authorized-venture-iteration-16-rerun-completed-2026-09-11).
This adds no SBC or power claim and does not transfer the original
latent-belief model's SBC evidence to the anchored model; the decision
continues to prioritize direct recovery of the actual anchored estimands and
geometry. The 2026-09-11 approval of the $122.78 ceiling resolves the budget
blocker for the frozen campaign under the current conditional liability
assumptions. Status remains **NOT READY**: fresh staging and its manifest,
preflight from a clean committed revision, and explicit wave and production
launch authorization remain pending. The worktree is dirty and a Git commit
is not yet authorized; neither the recovery rerun nor budget approval
authorizes those remaining actions.

## Amendment 5: realized-cell primary estimands

Amended: 2026-10-05, before production choice collection, following the
independent review of report revision `db145fe` and the author's approval of
realized-cell primary contrasts with additive-coefficient companions.

This amendment changes the estimand, not the Stan likelihood, priors, menus,
assessed probabilities, utility grid, or collection plan. The reason is that
the scientific questions concern the particular arms, prompts, and tasks in
this fixed design, rather than only their hierarchical additive components.
Narrower intervals are not the scientific justification for this choice.

Let `ell[a,p,d]` be the realized log sensitivity for model arm `a`, prompt `p`,
and domain `d` at a common centered menu size. In each posterior draw it equals
`gamma0 + X[cell] gamma + sigma_cell z_alpha[cell]`. All averaging is of these
log sensitivities, within each draw, before summarizing the posterior:

- RQ1: `mean_p ell[a,p,d] - mean_p ell[b,p,d]`, with weight 1/3 on each
  of the three prompt conditions for each model. Retain the seven named model
  comparisons in each pool.
- RQ2: `mean_a (ell[a,p,d] - ell[a,neutral,d])`, with weight 1/6 on each
  of the six model arms. Retain the two non-neutral prompt comparisons.
- RQ5: `mean_p (ell[a,p,hiring] - ell[a,p,procurement])`, with weight
  1/3 on each prompt, using only the dedicated matched fit. Retain six models.
- RQ4: use the same realized model averages as RQ1 in independent-posterior
  domain comparisons and all 15 pairwise model orderings; remain descriptive.
- RQ3 and RQ6 retain their existing estimands and status.

Exponentiating these contrasts yields ratios of geometric mean sensitivities,
not ratios of arithmetic means or maximizer-selection rates. Equal weights do
not change with observation counts, missingness, or posterior precision. The
common intercept and common menu-size contribution cancel in these zero-sum
contrasts; the implementation may therefore reconstruct them without the
intercept from the validated design and residual draws.

The previous gamma-based contrasts remain descriptive companions, with
posterior summaries but no additional primary decisions. The central-90%
interval-plus-median rule, practical thresholds, and multiplicity policy are
unchanged. There are still 26 named primary decisions, representing 24 distinct
comparisons up to sign because the OpenAI duplicate occurs in each pool.
Presentation-only and utility-grid comparisons use the new primary estimands.

Cell weights are bound to explicit canonical IDs. A required cell excluded
from a fit makes the affected contrast unavailable, with the missing IDs
reported. Do not renormalize over remaining cells, impute a missing realized
cell, or substitute the additive coefficient. The existing full-rank check
still applies to each retained design. All 26 named slots remain in reporting,
with separate counts of available and unavailable decisions; unavailable is
not a non-detection.

The generated contract records version `amendment5_realized_log_sensitivity_v1`,
the primary cell weights, and descriptive companion status. Report output uses
schema version 2; the 15-entry artifact manifest retains input schema version
1. Historical contracts do not match the new contract and must not be silently
accepted by changing or weakening hash checks. This amendment requires no new
fits for its arithmetic, but reuse of saved fits still requires verified
design, cell-order, and artifact provenance.

The September 12 evidence bundle remains an unchanged historical snapshot of
the previous additive-primary contract. Its recovery summaries are not new
validation of all amended estimands. In particular, the review's saved-draw
script did not compute realized-cell RQ5 coverage. Verification of the review's
operating-characteristic calculations, including the shared simulation truths,
and evaluation of realized-cell RQ5 remain outstanding; no new coverage or
power claim is made by this amendment.

### Remaining independent-review conditions

Amendment 6 below resolves the descriptive assessment-scale specification. It
is a different measurement scale, not a correction proving which stated
probabilities are appropriate. Ceiling/prior-sensitivity diagnostics and
size/stratum/filler/item/presentation predictive checks also remain open.
Their implementation and interpretation rules must be settled before
production; this amendment does not invent numeric thresholds for them.

The dated [October review disposition](REVIEW_DISPOSITION_20261005.md) records
accepted findings, qualifications, and remaining work. No pilot fit, production
preflight, provider call, commit, or collection is authorized by this amendment.
Status remains **NOT READY** pending review remediation, a final committed
revision, fresh preflight, and explicit launch authorization.

## Amendment 6: descriptive assessment-scale transformation

Amended: 2026-10-05, before production choice collection, with author approval.

Use the primary full-data posterior at utility midpoint 0.5 to describe how
realized-cell comparisons change when sensitivity is expressed per unit of
within-reference-set variation in assessed expected utility. No additional
fit, change of prior, or rescaling of the primary likelihood is introduced.
The same 15 fits and 26 named primary decisions remain required.

For arm `a` and domain/task `d`, let `sd[a,d]` be the finite-set standard
deviation of expected utility, using divisor N (ddof=0), and let `mean[a,d]`
be the corresponding mean. Define

`eta_star = (eta - mean[a,d]) / sd[a,d]`

`alpha_star = alpha * sd[a,d]`

`log_alpha_star = log_alpha + log(sd[a,d])`.

For positive SD these transformations preserve every softmax choice
probability. Centering is conceptual: reporting adds the log-SD offset to
existing draws without modifying Stan inputs. This preserves the fitted
posterior, including its original prior assumptions; it is not a fit with
unchanged numerical priors on standardized sensitivities.

### Fixed reference sets and provenance

- Primary venture/hiring fits: all 60 frozen items in the arm's pool.
- Matched RQ5: the 24 procurement items and the 24 matched hiring items,
  separately for each arm and task. Do not include off-task padded eta columns
  or use a full-pool SD for a matched comparison.
- Every item has equal weight, regardless of menu appearances or observed
  choices. Build references before presentation filtering, parsing exclusions,
  or cell exclusions. Use midpoint 0.5 in every reference, including the copy
  accompanying another utility variant's assembly report.
- Geometry includes full item IDs, probabilities, eta values, means and SDs,
  plus menu IDs, individual top-two gaps, gap quantiles and tie prevalence.
  Count every frozen menu once, retaining repeated compositions with distinct
  IDs. Primary pools have 140 menus (100 primary-family and 40 matched-family),
  with 25 and 10 respectively at each size 2/4/6/8; matched tasks each have
  40 menus, 10 at each size.
- For these descriptive geometry summaries a top-two gap within absolute
  tolerance 1e-12 of zero is a tie (relative tolerance zero). This does not
  change the existing posterior-predictive exact-tie definition.

The runner writes `assessment_scale_reference` in the assembly report. The
existing preparation-report SHA256 binding therefore covers its complete
reference inputs. Reporting recomputes geometry, checks canonical item IDs and
families, menu counts and size allocation, validates retained eta against the
bound probabilities at the fit's utility variant, and requires identical
references across the five sibling variants. Counts establish structural
completeness, not independent proof of the exact frozen content: the supplied
artifact hashes and final preflight remain essential. No unbound historical
bundle fallback is allowed.

Missing or inconsistent provenance is an error. Zero reference SD makes the
affected standardized comparisons unavailable, without substituting epsilon;
it does not suppress the original comparison. Missing required cells retain
Amendment 5's unavailable policy. Reduced smoke preparation may explicitly
omit these references, but such outputs cannot pass Amendment 6 reporting.

### Descriptive reporting and interpretation

Apply Amendment 5's fixed cell weights to the transformed draws. Show original
and standardized medians and central 90% intervals with the deterministic
offset. For an RQ1 model contrast this offset is
`log(sd[a,d]) - log(sd[b,d])`; RQ5 uses hiring minus procurement SD offsets.
RQ4 compares the transformed model contrasts and all 15 model-pair orderings
using the existing independent-posterior combination, preserving whole-draw
dependence within each pool.

When SDs are positive, RQ2's within-model prompt offsets cancel exactly, and
the RQ6 size slope is unchanged because its scale factor is constant across
menu sizes. Report those invariances explicitly. With zero SD the standardized
cell parameter is undefined; do not claim a valid standardized contrast merely
because formal prompt offsets would cancel. The unchanged slope can still be
reported as the original fitted parameter.

Flag median sign reversals and changes in whether central intervals exclude
zero as scale-dependent descriptive conclusions. Do not apply the original
ROPE, emit new detection decisions, or treat these flags as confirmatory
failures. This comparison concerns a different unit, not a correction of
probability judgments, and it cannot remove ranking or distribution-shape
differences. An actual difference in stated beliefs can legitimately contribute
to a scale difference.

The policy is recorded as `amendment6_assessment_scale_v1`; report schema 2 and
input manifest schema 1 remain in use, with the new exact contract and required
reference metadata. Historical contracts/references are not silently promoted.
This amendment resolves the A2 implementation specification only. Prior and
ceiling diagnostics, expanded predictive checks, and verification of amended
recovery evidence remain open. Status remains **NOT READY**. No fits, provider
calls, preflight, commits, or production launch are authorized here.

## Amendment 7: ceiling diagnostics and prior sensitivity

Amended: 2026-10-06, before production choice collection, with author approval.
The complete approved specification is
[A3 ceiling and prior policy](A3_CEILING_PRIOR_PROPOSAL_20261006.md).
This amendment supersedes earlier statements that the A3 procedure is
unspecified or that 15 fits exhaust the required plan. It does not supersede
the primary likelihood, priors, estimands, or decision thresholds.

### Diagnostics and interpretation

For each retained cell in the three full-data midpoint-0.5 datasets, report
observation and distinct-menu counts, presentation/size breakdowns, exclusions,
exact maximizer and tie counts, separately labeled near-tie counts (absolute
eta tolerance 1e-12, relative tolerance zero), and individual choice regrets
and top-two gaps with size summaries. Use canonical production eta arithmetic.
Do not modify eta to manufacture ties or treat top-two gaps as sufficient
information about the whole menu.

Evaluate prior-free, full-menu conditional likelihood slices in cell log
sensitivity at the primary posterior size-slope 5th, 50th and 95th percentiles.
The approved specification fixes the 401-point grid, added posterior-quantile
locations, refinement, and numerical-failure reporting. These are not profile
likelihoods, calibrated confidence sets, or a joint identification test.
At any fixed finite slope, entirely equal-utility menus give constant
likelihood; all-maximizer choices on at least one unequal menu give a supremum
only at infinite sensitivity; any strictly nonmaximizing choice gives eventual
upper-tail decay. Eventual decay alone does not establish useful precision.
Report continuous likelihood changes and analytic limits, not an arbitrary
maximizer-rate or flatness gate.

### Prior variants and fit scope

Normal parameters are mean and SD; HN denotes a positive half-normal scale.
Every residual z retains its standard-normal prior and all coding is unchanged.

| Variant | gamma0 | Each gamma | sigma_cell | gamma_size |
|---|---|---|---|---|
| Primary | N(2.5, 0.5) | N(0, 0.5) | HN(0.3) | N(0, 0.2) |
| L | N(2.5, 1.0) | N(0, 0.5) | HN(0.3) | N(0, 0.2) |
| H | N(2.5, 0.5) | N(0, 1.0) | HN(0.6) | N(0, 0.2) |
| S | N(2.5, 0.5) | N(0, 0.5) | HN(0.3) | N(0, 0.4) |

L widens the common level, H relaxes both sources of between-cell shrinkage,
and S widens the size slope. These are specified stress checks, not an
exhaustive robustness envelope or a correction for coding asymmetries.
Run each for venture, hiring, and matched RQ5 on exactly the corresponding
primary data, adding nine fits to the existing 15. Do not select fits by
maximizer rate or cross these priors with utility/presentation variants.
The original primary Stan model is unchanged; a parameterized sibling supplies
the alternatives with validated prior fields and saved setting echoes.

Show cell log-sensitivity and sensitivity quantiles through the 99th percentile,
realized-contrast medians and central 90% intervals, sign probabilities,
median/endpoint shifts and width ratios. Apply the original decision rule to
alternative summaries only as sensitivity annotations. A changed decision is
labeled prior-sensitive under these checks; unchanged decisions do not imply
unchanged magnitudes. Preserve the 26 named primary decisions, 24 distinct up
to sign. RQ3 stays descriptive and RQ4 combines independent pool posteriors
within each prior variant, including all 15 model-pair orderings. Do not pair
draws across priors. Amendment 6 standardized outputs remain primary-prior only.

Finite posterior upper quantiles for cells with a supremum at infinity reflect
prior regularization and hierarchical pooling, not likelihood-only upper
bounds. Do not automatically suppress those cells' contrasts. Incomplete or
sampler-invalid checks remain explicitly incomplete, never evidence of
robustness. Prediction stability does not establish parameter precision or A4
adequacy. No replacement primary prior or automatic launch gate is introduced.

### Provenance and compatibility

Policy version is `amendment7_ceiling_prior_v1`. A3 fit manifests use schema 2
and reports use schema 3. Observation evidence uses version 2 and binds the
frozen menu/presentation order separately from Amendment 6 geometry. The full
eligible observation-key universe must equal the disjoint retained and excluded
sets, with canonical membership, chosen-item/order checks, complete NA audits,
and reconciled whole-cell exclusions. Missing collection records fail closed;
they are not fabricated NA outcomes. Reduced smoke preparation cannot establish
A3 completeness.

Prior inputs must equal the primary input apart from their exact declared prior
fields. Reports validate preparation identity, prior contracts, model-source
hashes, and saved prior-setting echoes. Hashes bind supplied artifacts, not
cryptographic proof of execution. Old preparation reports must be rebuilt for
A3 schema-2 loading. Historical schema-1 loading remains readable with A3
absence explicitly marked; it cannot establish completion of the amended plan.
Missing or empty optional-prior chain directories remain visible as missing;
malformed supplied provenance is an error.

The offline `fit-plan` and `fit-manifest` CLI commands prepare the plan and
bindings but do not execute fits. Resource authorization remains separate:
benchmark one authorized posterior fit before scheduling the remainder, keep
durable chains, and retain the existing sampler gates. A fit count is not a
runtime estimate.

### Offline validation and remaining conditions

The reproducible prior-predictive check uses 10,000 draws per group/prior,
seed 20261006, actual frozen neutral probabilities and menus, primary and
matched coding, and sizes 2/4/6/8. It reads no collected choices and runs no
Stan sampler. Source hashes and summaries are recorded in
[the A3 prior evidence](../../reports/applications/seu_sensitivity_study/data/a3_prior_predictive.json).
Regenerate and verify it with
`python -m applications.seu_sensitivity_study.a3_prior_predictive --check`
using the project interpreter and the recorded local frozen inputs.

All prior summaries are finite. H approximately doubles realized-contrast
90% interval widths; L broadens common levels without directly broadening
zero-sum contrasts; S broadens size effects. These are induced-prior checks,
not posterior robustness or recovery evidence. No approved prior values were
changed following inspection.

The application suite passed 841 tests, including likelihood limits, tiny
regrets, mappings, missing records, frozen presentation orders, exact production
eta arithmetic, prior identity, incomplete fits and unchanged primary results.
The sibling Stan model passed syntax validation without posterior sampling.
A3 specification and offline implementation validation are complete; posterior
checks and runtime/mixing remain unmeasured. A4, amended recovery verification,
and other review conditions remain open. Status is **NOT READY**. No posterior
fit, provider call, preflight, commit, push, or collection is authorized here.

## Amendment 8: expanded predictive checks

Amended: 2026-10-06, before production choice collection, with author approval.
The complete [approved A4 specification](A4_PREDICTIVE_CHECK_PROPOSAL_20261006.md)
controls the statistics, groupings, denominators, and interpretation rules.
This amendment resolves the previously unspecified A4 procedure, the diagnostic
parts of B1/B6, and C5. It adds no fits or primary decisions: the plan remains
24 fits and 26 named decisions, 24 distinct up to sign.

### Checks and interpretation

Use the saved replicated choices and joint parameter draws for every supplied,
sampler-valid variant, on exactly that fit's retained observations and eta.
Keep the same posterior draw across observations and statistics. These are
in-sample checks under the independent-choice likelihood, not held-out forecasts
or calibrated tests. Missing or invalid prior variants remain incomplete.

Report cell and pool/task summaries overall, by size, and by family x authored
stratum x size. Keep procurement and hiring separate in the matched fit.
Pool/task summaries weight retained observations equally and do not redefine
the equally weighted cell estimands. Empty groups remain unavailable with zero
counts, and all nonempty groups retain counts regardless of sample size.

- Compare exact-maximizer fractions, mean regret, authored-filler fractions,
  mean selected-choice probabilities and mean log scores. Exact eta maxima are
  unchanged; A3 near-tie geometry stays separately labeled.
- Validate the frozen variant-D recipe before classifying weak-labeled items
  as fillers. A filler may maximize assessed utility. Size-2 filler selection
  is a structural zero, not evidence that the model reproduces avoidance.
- Report each item's exposures and observed/replicated selection counts,
  exposure-conditional rates, and unconditional choice shares. Zero exposure
  makes a conditional rate unavailable, not an observed zero preference.
- Map choices from sorted-active index to item ID and then to frozen displayed
  position. Stratify position distributions by size and presentation. Legacy
  sorted-index summaries remain encoding diagnostics, not position-bias checks.
- Pair only the same cell/menu with both presentations retained. Report
  same-item and same-position fractions, conditional probability-product
  expectations, and complete/singleton/neither-retained counts. Use both saved
  replica entries from the same draw. Presentation-only paired checks are
  unavailable by design; repeated menu compositions do not become one pair.
- For maximizer, regret, and filler summaries, report equal-weight four-size
  linear trends with weights `(size - 5) / 20`. Missing any of 2/4/6/8 makes
  the trend unavailable. These behavioral trends are not gamma_size estimates.

Observed and replicated statistics use consistent stable summation, including
size-trend inputs, so identical contributions cannot receive discrepancy flags
solely from different floating-point reduction paths. This does not replace
exact eta or statistic comparisons with an arbitrary tolerance.

Report central 90% predictive intervals, signed discrepancies, and separate
less/equal/greater tail fractions. A fixed observed value strictly outside its
predictive interval receives a descriptive review flag. For draw-dependent
scores, use within-draw observed-minus-replicated differences and flag an
interval excluding zero. Report every prespecified row. Flags are not
multiplicity-controlled discoveries, independent failures, or an omnibus gate.

The RQ6 output carries nonexclusive qualifications for unreproduced size trends,
choice-pattern and display-position discrepancies, excess same-item repetition,
missing sizes/pairs, and conditioning on retained observations. Report eligible,
resolved, retained, unresolved and whole-cell-removed counts with reconciled
denominators, and size-specific unresolved rates and ranges. Do not double-count
unresolved rows also removed with their cell. Missingness is not modeled by the
choice likelihood; predictive agreement cannot establish robustness to it.
Preserve primary estimates and decisions alongside these qualifications, without
automatic cancellation, replacement fits, causal mechanism claims, or launch
clearance. Lack of flags is not proof of model adequacy or precise sensitivity.

### Evidence, migration, and validation

Policy ID is `amendment8_predictive_checks_v1`. Input manifest schema 2, output
report schema 3, and observation metadata version 2 are unchanged. Preparation
now additionally binds version-1 `predictive_reference` evidence containing
canonical item quality labels, recipe `variant_D`, and frozen menus. Validate
the complete role recipe and sibling evidence; do not infer roles from eta.
Historical schema-1 preparation without this evidence yields A4 unavailable.
Current schema-2 inputs require rebuilt preparation reports and refreshed
SHA256 bindings. Rebuilding this metadata does not itself require refitting;
any change to actual fit inputs must not be passed off as a metadata refresh.

Expanded results are under `posterior_predictive_checks[group][variant].a4`;
RQ6 carries `predictive_interpretation`. The full application suite and
production-shaped offline benchmark completed successfully after an independent
audit found and corrected aggregation roundoff that falsely flagged identical
replicas. Regressions cover identical replicas, row permutations, exact tails,
roles, item exposure, pairs, display mapping, trends, missingness, and unchanged
primary results. The benchmark uses 5,040 observations, 18 cells, 60 items and
500 saved synthetic replica draws; it is not a posterior fit or runtime forecast
for sampling. Its expanded output is approximately 5.2 MB for one fit, so
machine-readable tables should not all be inlined into the main report.

Implementation validation is not a production adequacy result. No production
posterior checks have run. Amended recovery verification and other review
conditions remain open; final-revision preflight and collection authorization
are still required. Status remains **NOT READY**. No fits, provider calls,
commits, pushes, or collection are authorized by this amendment.

## Amendment 9: descriptive Sonnet arm comparison

Amended: 2026-10-07, before production choice collection, implementing the
accepted B3 independent-review follow-up. Policy ID:
`B3_descriptive_postreview_2026-10-07`. This is a dated post-review descriptive
addition, not an original primary hypothesis or a production result.

Within each primary pool (venture and hiring), compute in every joint draw

$$
{\theta}_d = \frac{1}{3}\sum_{p=1}^{3}
\left(\ell_{\mathrm{thinking},p,d}-\ell_{\mathrm{base},p,d}\right),
\qquad \ell_{a,p,d}=\gamma_0+X_{a,p,d}\gamma+\sigma_{\mathrm{cell}}z_{a,p,d}.
$$

The canonical arms are `claude-sonnet-4-5-thinking` minus
`claude-sonnet-4-5`. Use fixed weights +1/3 and -1/3 across the three prompts,
not observation counts. The common intercept and menu-size term cancel at a
common menu size; cell residuals do not. Report the posterior median, central
90% interval, and positive/negative/exact-zero probabilities. Also summarize
`exp(theta_d)` draw by draw: the ratio of geometric-mean sensitivities across
prompts, not the arithmetic mean of prompt-specific sensitivity ratios.
If any of the six required cells is missing, report the contrast as unavailable
with missing IDs; do not reweight, impute, or fall back to gamma coefficients.

This compares configured arms under their own fixed neutral assessments. Both
use `claude-sonnet-4-5-20250929`, but request settings differ (including thinking
budget and temperature), and their assessments can differ. It is not a pure
causal reasoning effect, a shared-assessment comparison, or a claim that
temperature necessarily biases sensitivity downward.

The separate `sonnet_thinking_descriptive` section of `posterior_fit_report`
applies to every supplied sampler-valid primary-pool variant: primary, both
presentation subsets, both utility variants, and L/H/S priors. Prior variants
retain their existing complete/missing/failed status handling. No matched-RQ5
combined-task or new difference-in-differences contrast is added, and no
Amendment 6 standardized variant or crossed sensitivity grid is introduced.
There is no ROPE, detection decision, or classification for this contrast.
The primary family remains 26 named decisions, 24 distinct up to sign; the
fit plan remains 24. Existing gamma companions and decision rows are unchanged.

Offline synthetic-draw and report-integration checks are implementation
validation only. No numeric production results are stored. Saved-recovery
verification and independent review remain outstanding; status is **NOT READY**.
This amendment authorizes no fits, provider calls, preflight, or collection.