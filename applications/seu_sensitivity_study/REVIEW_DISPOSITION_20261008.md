# Independent Review 2: Approved Disposition and Implementation Status

Date: 2026-10-08

Review: [Independent pre-collection review 2](../../local/seu_sensitivity_precollection_review2.md),
reviewing commit `eabf974`.

Precedent: [October 5 disposition](REVIEW_DISPOSITION_20261005.md).

**Independent verdict recorded: ready for final preparation and preflight;
no scientific blockers.** The author approved all clarification proposals and
retained the existing arm-wide **STOP** rule alone, with no descriptive
available-cell companion. The historical recovery-provenance limitation is
accepted with disclosure, without remediation. The design report and a dated
preregistration banner/clarification now record that scientific status and the
approved reporting text. Historical bodies and verdicts remain unchanged.

The verdict and this documentation approval authorize no collection, spending, posterior fits,
provider calls, preflight, commits or pushes. Each requires separate explicit
authorization. No staging or runtime benchmark has been performed for this
disposition.

## Finding Dispositions

| Finding | Assessment and disposition |
|---|---|
| 1: missing-cell cascade | Approved and documented: loss of GPT-4o's neutral cell makes 8/10 pool decisions unavailable; loss of either instructed cell makes 7/10 unavailable, not 8/10. The other RQ2 comparison does not use that instructed cell. Author selected arm-wide STOP under the unchanged full-rank guard, with no available-cell companion. |
| 2: fixed assessment-scale offsets | Published a separate frozen-input supplement with 26 rows, 24 arm/reference combinations and 10 consumed-input/source hashes; exact reproduction passed. All 18 reviewer offsets match to three decimals using canonical normalized-eta/reference helpers and population SDs. Negative RQ5 offsets predict the scale shift, not the sign of an unknown production effect. |
| 3: dependence qualification | Approved text implemented in the report, dated preregistration clarification and reporting-code annotations for RQ1/RQ2/RQ5 rows, with references to existing paired diagnostics and presentation-only comparisons. No new flag computation, refit, gate, interval adjustment or detection rule was added. |
| 4: RQ2 visible calculation | Approved interpretation caveat implemented in both documents. It describes the existing intervention; prompts and estimand are unchanged. |
| 5: prior-sibling equivalence | Required before L/H/S posterior fits, not a scientific blocker before collection. NOT RUN; separately authorized. Scope is log-density equivalence up to a constant at primary prior settings on shared unconstrained points and production-shaped inputs, plus generated-quantity checks. The prerequisite is documented, not executed. |
| 6: historical calibration wording | Approved dated qualification implemented in the current preregistration banner and design report. Amendment 4's historical body is unchanged. |

The review's correction to the first review is accepted: **26 named decisions
represent 24 distinct comparisons up to sign**, not 25. Each primary pool has
its own exact OpenAI sign-reversed duplicate. No decision is added or removed.
The 24-fit plan remains 15 base fits plus 9 L/H/S fits, not a crossed grid.

Review 2 accepts A1, A3's specification, A4 and B3 as resolved; A2 is adequate
as a descriptive analysis. It accepts the disclosed dependence, narrow utility
grid, matched-task and missingness limitations. These are independent scientific
assessments, not claims that production adequacy or posterior sensitivity has
already been measured. Finding 6's historical-wording clarification is now
implemented in documentation.

## Historical Provenance: Accepted With Disclosure

The reviewer's Section 2 acceptance is recorded with author approval. The
historical recovery evidence may be used for its limited purpose: saved-draw
recovery under the working model at production geometry and an indication of
precision in those simulated settings. It is not a gate or a confirmatory result.
**No remediation is requested or planned for this accepted limitation.**

Keep the existing disclosures: 120 iterations across three geometries use
40 coupled seed clusters; contrast cases are dependent; original choice inputs
are missing; exact seed-to-choice replay and replacement-input equality remain
unverified. Acceptance does not turn these unverified claims into verified ones,
or establish formal calibration, exact-null false-positive rates, cross-geometry
independence, or production adequacy.

The review's phrase that the saved evidence "contradicts" mispairing and
selection risks is stronger than the available evidence warrants as proof of
absence. Posterior-truth agreement and replacement/original coverage comparisons
are reassuring diagnostics, but neither observes the missing inputs or the
selection process. This is an interpretive qualification, not a disagreement
with acceptance or a request to reopen remediation. The existing verification
artifact explicitly preserves those limits.

The optional venture-iteration-9 explanatory addition remains outside this
approved edit package. The reviewer reports an extreme-heterogeneity, high-sensitivity
case; the suggested shrinkage explanation should remain an interpretation, not
an established causal account. Its detailed numerical analysis was not rerun
for this documentation update.

## Approved Clarifications

These clarifications are approved. The wording is implemented in the design
report and the new October 8 preregistration clarification; supplement work is
tracked separately below. They add no fits, gates, primary decisions, or changes
to thresholds, priors, likelihood, menus or prompts.

### 1. Missing Cells

Approved report text and dated preregistration clarification:

> A single missing cell can make several fixed-design contrasts unavailable.
> For example, excluding GPT-4o's neutral cell removes eight of a pool's ten
> named primary decisions: six GPT-4o-involving RQ1 decisions and both RQ2
> decisions. Excluding its SEU-instruction or deliberative cell instead removes
> seven: the same six RQ1 decisions and the corresponding RQ2 decision. Provided
> the retained design remains full rank and other required checks pass, the
> Anthropic flagship-minus-small comparison and RQ6 remain available, as does
> the unaffected RQ2 comparison in the latter cases. Excluding an entire arm
> makes the original design rank deficient and stops that pool's analysis;
> unavailable contrasts are never reweighted or replaced by additive estimates.

**Author decision: retain arm-wide STOP alone, with no available-cell
companion.** No reduced-model posterior fit, reweighting or additive fallback
is adopted. The rank guard and fixed-design estimands are unchanged.

### 2. Frozen Offset Supplement

Approved caption/interpretation:

> These offsets are determined before choices are collected. For RQ1 the
> standardized-minus-original log-sensitivity contrast is log(S_a/S_b); for
> RQ5 it is log(S_hiring/S_procurement). S is the population SD of normalized
> assessment-derived expected utilities at midpoint 0.5, over all 60 items in
> each primary pool or each matched task's own 24 items. Negative RQ5 offsets
> mean standardization shifts the contrast downward; they do not determine the
> sign of the original or standardized production contrast. These are descriptive
> changes of scale, not new tests or findings about choices.

The separate dated JSON/table supplement is generated from the frozen bundle
through `assessment_scale.build_reference` and `contrast_offset`, using canonical
`assessment_expected_utilities`. It contains 14 named RQ1 rows, four algebraically
zero RQ2 rows, six RQ5 rows and two fixed-input context rows. The 24 arm/reference
combinations retain item IDs/counts, normalized eta values, population SDs,
directions, unrounded offsets and display rounding. Ten SHA256 hashes bind the
consumed frozen inputs and computation source, including canonical pool files
read by the reference helper. Zero SD remains unavailable, without an epsilon.
The September snapshot was not refreshed or edited.

The report includes the [offset table](../../reports/applications/seu_sensitivity_study/_offset_evidence.qmd)
and links the [JSON supplement](../../reports/applications/seu_sensitivity_study/data/offset_evidence.json).
Exact offline reproduction and full Quarto rendering passed. An independent
read-only implementation check recomputed all 24 references and 26 rows, with
maximum absolute numerical difference 4.44e-16; all ten hashes matched.

The review's two Sonnet-thinking-minus-Sonnet offsets can be shown as fixed-input
scale context, clearly separated from Amendment 6's primary-comparison table.
They do not add an Amendment 9 standardized posterior analysis or decision.

### 3. Dependence Qualification

Approved reporting text for affected RQ1/RQ2/RQ5 rows:

> An excess same-item repetition flag in contributing cells also qualifies this
> comparison and should be read with its existing presentation-only estimates.
> The flag raises concern about the independent-observation uncertainty model;
> it neither identifies the cause of repetition nor changes the interval,
> threshold or primary detection decision. An unavailable paired diagnostic
> is not evidence of no dependence.

Use existing flags and contributing-cell mappings; do not create a new flag,
aggregate test or requirement that both presentation-only intervals exclude zero.
The reporting implementation adds conditional text, contributing-cell IDs and
paths to the existing A4 record and presentation-only comparisons. The A4 record
remains addressable when historical evidence is unavailable and has no `pairs`
child. No new applicability flag is computed, no numerical or decision content
is changed, and no production diagnostic result is claimed.

### 4. RQ2 Interpretation

Approved sentence:

> Because the choice prompts display the assessed probabilities, an SEU-instruction
> effect may reflect following an explicit expected-utility calculation rather
> than greater coherence between beliefs and actions beyond that displayed task.

### 6. Historical Calibration Banner

Implemented dated banner addition, leaving Amendment 4 intact:

> Amendment 4's phrases "direct production-geometry calibration" and "consistent
> with nominal coverage" must be read in the narrower scope established by the
> October 7 verification: recovery under the generating model with 40 coupled
> seed clusters and dependent contrast cases, not a formal calibration claim.
> Review 2 accepted the disclosed historical choice-input provenance limitation
> on October 8; it did not establish exact replay or authorize collection.

## Finding 5: Required Later, Not Executed

Before L/H/S posterior fits, and only with explicit authorization, compare the
primary and prior-sibling Stan log densities at primary prior settings using
shared unconstrained parameter values on production-shaped data. Align parameter
order, transformations, Jacobian settings and density-constant conventions;
declare numerical tolerances and examine pointwise differences up to a constant.
Check deterministic generated quantities and implied choice probabilities as
well. Random replicated choices require distributional consistency, not automatic
bit-for-bit equality from identical seeds across rewritten sampling code.

This is a verification prerequisite for the existing prior comparisons, not a
new scientific gate or posterior fit. It is not a condition of the review's
precollection verdict. No Stan compilation, log-density evaluation,
generated-quantity execution or posterior sampling was performed for this
documentation update. Status: **NOT RUN; separate authorization required**.

## Preliminary Verification Record (Before Documentation Implementation)

The following records the checks performed while preparing the draft
disposition. These preliminary checks are distinct from the published-supplement
and reporting implementation validation recorded below.

The reviewer offset script was read and executed without modification. An
in-memory check independently built all three canonical references from the
frozen bundle, used `contrast_offset` for RQ1/RQ5, and compared all 18 displayed
values. All matched at three decimals; the maximum absolute difference from
a printed reviewer value was 0.0004869958893847365, within rounding.
No source probabilities were mutated and no published supplement was generated
in that preliminary check.

| Offset direction | Venture | Hiring |
|---|---:|---:|
| GPT-4o-mini minus GPT-4o | +0.063 | -0.181 |
| o3-mini minus GPT-4o | +0.037 | +0.026 |
| Sonnet minus GPT-4o | +0.019 | +0.059 |
| Haiku minus GPT-4o | -0.040 | -0.133 |
| Sonnet thinking minus GPT-4o | -0.054 | +0.166 |
| Sonnet thinking minus Sonnet (fixed-input context only) | -0.073 | +0.107 |

| RQ5 hiring minus procurement | Offset |
|---|---:|
| GPT-4o | -0.197 |
| GPT-4o-mini | -0.409 |
| o3-mini | -0.161 |
| Sonnet | -0.096 |
| Haiku | -0.101 |
| Sonnet thinking | -0.037 |

Reproduction input/helper hashes recorded for this preliminary check:

| File | SHA256 |
|---|---|
| reports/applications/seu_sensitivity_study/data/design_evidence.yml | 53b4299bbcfca78fc11a84a6de99bc1e45fb9439e8d986d8c25c553c8f228708 |
| applications/seu_sensitivity_study/assessment_scale.py | 378639cddf2f23e4e3a2e9642988fa082ca6a0217642b5b413696c39eb523923 |
| applications/seu_sensitivity_study/data_preparation.py | 1d23ac39d6ca833fb02cee9d547511d948f6e6681baa47e4a0e0984153414912 |
| local/tmp/seu_review2/offsets.py | d825dac69370403180bd426b7cb3d44bb5daf68bc0c50ff5a5737d4281d5731c |

These preliminary hashes are not a substitute for the complete consumed-input
manifest required for the approved published supplement.

The existing synthetic realized-estimand fixture was used in-memory to exclude
each GPT-4o prompt cell in each pool: unavailable counts were 8, 7 and 7 for
neutral, SEU and deliberative respectively. Removing the entire GPT-4o arm in
either pool raised `Retained design rank must be unchanged and full with intercept`.
These checks call the real posterior reporting helper on constructed draws;
they do not fit a model or establish production missingness rates.

Earlier focused validation on October 8 in the existing Python 3.10.19 environment:

```text
PYTHONDONTWRITEBYTECODE=1 /Users/jeffhelzner/miniforge3/envs/seu-sensitivity/bin/python -m pytest applications/seu_sensitivity_study/tests/test_assessment_scale.py applications/seu_sensitivity_study/tests/test_realized_estimands.py -q -p no:cacheprovider
95 passed in 8.75s

quarto pandoc applications/seu_sensitivity_study/REVIEW_DISPOSITION_20261008.md --from=gfm --to=html --output=/dev/null
Passed (exit 0).

git diff --check
Passed (exit 0).
```

No full suite, recovery regeneration, prior simulation, scientific report render,
preflight, Stan equivalence check, posterior fit or provider call was run for
that preliminary validation.

## Implementation and Remaining Procedure

All proposed clarifications are author-approved, including retention of arm-wide
STOP with no companion. Current status is:

> Scientifically ready for final preparation and preflight following independent
> review 2; operational preparation and authorizations remain outstanding.

Implemented documentation:

- [Design report](../../reports/applications/seu_sensitivity_study/01_study_design.qmd):
	missing-cell cascade, visible-calculation caveat, repetition qualification,
	calibration qualification, finding 5 prerequisite, current scientific status,
	and offset supplement include/link.
- [Preregistration](PREREGISTRATION.md): updated interpretation banner and a new
	dated October 8 clarification before the historical body. Amendment 4 and
	every other historical section remain unchanged.
- [Offset builder](../../analysis/study_offset_evidence.py): frozen-input table,
	JSON publication, complete consumed-input hashes and exact reproduction check.
- [Reporting implementation](confirmatory_reporting.py): conditional dependence
	qualification on RQ1/RQ2/RQ5 rows, preserving existing results and decisions.
- This disposition: approved decisions, implementation and deferred operations recorded.

The October 5 disposition and September evidence snapshot are unchanged.
Models, estimation rules and executable guards were not changed. The unrelated
user edit to `.gitignore` was left untouched. The existing artifact-bound machine contract still says that provenance
acceptance is required. Its source is intentionally **unchanged**: this dated
external disposition supersedes the pending human-acceptance status only.
Replay/equality verification fields remain false/unverified, and no hashes,
source bindings or executable guards are changed or waived. No claim is made
that the machine-contract source now encodes acceptance. Any later rebind or
regeneration must be separately recorded and checked, never manually weakened.

Initial Pandoc-to-HTML syntax checks passed for the preregistration, design QMD
and draft disposition. The preregistration's plain-HTML check emitted a
TeX-conversion warning on an existing Amendment 9 equation; the QMD check used
`--mathjax`. Subsequent implementation validation was:

```text
PYTHONDONTWRITEBYTECODE=1 /Users/jeffhelzner/miniforge3/envs/seu-sensitivity/bin/python -m pytest applications/seu_sensitivity_study/tests/test_review2_reporting.py -q -p no:cacheprovider
16 passed in 0.52s (after historical A4 path repair).

PYTHONDONTWRITEBYTECODE=1 /Users/jeffhelzner/miniforge3/envs/seu-sensitivity/bin/python -m pytest applications/seu_sensitivity_study/tests/test_confirmatory_reporting.py::test_build_report_from_saved_fit_manifest applications/seu_sensitivity_study/tests/test_review2_reporting.py applications/seu_sensitivity_study/tests/test_study_offset_evidence.py -q -p no:cacheprovider
57 passed in 15.11s.

quarto render reports/applications/seu_sensitivity_study/01_study_design.qmd
Passed; HTML rendered with the offset supplement.

PYTHONDONTWRITEBYTECODE=1 /Users/jeffhelzner/miniforge3/envs/seu-sensitivity/bin/python -m analysis.study_offset_evidence --check
Passed; 14 RQ1, 4 RQ2, 6 RQ5, 2 context rows and 24 references reproduced.

PYTHONDONTWRITEBYTECODE=1 /Users/jeffhelzner/miniforge3/envs/seu-sensitivity/bin/python reports/applications/seu_sensitivity_study/_build_evidence.py --check
Passed; historical evidence bundle validated offline without refresh.

PYTHONDONTWRITEBYTECODE=1 /Users/jeffhelzner/miniforge3/envs/seu-sensitivity/bin/python -m analysis.study_display_evidence --check
Passed; 24 arm/subset rows validated and historical bundle unchanged.

quarto pandoc applications/seu_sensitivity_study/REVIEW_DISPOSITION_20261008.md --from=gfm --to=html --output=/dev/null
Passed (exit 0).

git diff --check
Passed (exit 0).
```

The 57-test run includes 40 offset tests, 16 annotation tests and one assembled
saved-report integration test. The integration test resolves diagnostic paths
against historical unavailable A4 records and compares the report's original
content exactly after removing only the reporting annotations. These checks use
constructed draws and fake fit loading, not posterior sampling.

VS Code's error check reported no errors in the offset builder, reporting
implementation, two new focused test modules or updated saved-report test.
No full test suite was run for this implementation.

Finding 5 remains **NOT RUN**, required before L/H/S fits and separately
authorized, without becoming a scientific blocker before collection.

Keep distinct: author approval of edits; a separately authorized clean commit
(the unrelated `.gitignore` edit must be resolved by its owner); fresh staging
and separately authorized preflight against that commit; explicit collection
and spending authorization; the separately authorized runtime benchmark and
posterior fits; and the authorized finding 5 check before L/H/S fits. A benchmark
that fits a model itself requires fit authorization. No step is implied by
approval of the clarification wording or by the scientific verdict. All these
operational steps remain unrun and unauthorized for this final revision in this
implementation update. Historical runs, earlier budget approval and this
scientific acceptance are not substitutes for those separate authorizations.