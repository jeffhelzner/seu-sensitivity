# Phase E4: Go/No-Go Validation (Superseded)

Date: 2026-09-06

Status amended 2026-09-07: **NOT READY.** The original technical GO below is
retained as a historical record of the checks completed at that time. An
independent review subsequently found a production-dimensional alpha
identification failure, an RQ4 estimand mismatch, and unresolved Batch
integrity and budget-control failures. The preregistration now specifies an
assessment-anchored primary estimand. E4 is reopened and must be rerun against
that amended model and a hardened collection pipeline before production.

## Frozen-design checks

The explicit production configuration validates with 36 cells across venture
and hiring, 140 menus per pool, two presentations, and 10,080 expected choice
calls. The zero-cost pipeline dry run independently reports 5,040 calls per
pool and 280 observations per cell.

Both retained pools pass under the frozen embedding-axis thresholds and
variant-D recipes:

| Pool | Gate | Worst cross-size eta-gap difference | Cross-pool difference | Threshold |
|---|---|---:|---:|---:|
| Venture | pass | 0.0906 | 0.0601 | 0.25 |
| Hiring | pass | 0.0484 | 0.0601 | 0.25 |

The first E4 validation attempt reported a false venture failure because the
pool-major runner compared venture's new report to hiring's stale pre-freeze
report. Hiring then compared against venture's new report and passed. The
runner now completes every selected pool through validation, refreshes all
selected validation reports after sibling reports are current, and only then
runs later phases. A regression test locks this ordering. With that fix, both
reports independently contain the same 0.0601 cross-pool difference and pass.

The full application test suite passes with 460 tests.

## Reopened-gate progress (2026-09-07)

The assessment-anchored inference and simulation models compile. Subsequent
production-dimensional validation and 40-dataset recovery campaigns for
venture, hiring, and matched RQ5 completed successfully, and prior predictive
checks against the observed assessment spread calibrated the size-slope prior.

The preregistered utility sensitivity grid is executable: the runner emits the
primary `u=0.50` Stan data and separate `u=0.35` and `u=0.65` inputs. Batch
request identity now covers provider-ready bodies and binds state, checkpoints,
and final choice artifacts. Duplicate IDs and changed completed artifacts are
rejected, and partial terminal evidence is persisted before an integrity error.

Prior predictive checks now use all persisted neutral assessments and the exact
frozen menus. Under the amended `gamma_size ~ normal(0, 0.2)` prior, the
size-8/size-2 alpha ratio has an approximately `[0.135, 7.03]` central 90%
interval; 2.92% of draws are below 0.1 and 2.46% are above 10. The inherited SD
0.5 prior had an approximately `[0.007, 146]` interval and about 44% total mass
beyond a factor of 10. Amendment 2 records the prior decision and its rationale.

An exact venture-shape diagnostic reconstructed 18 cells, 60 assessed items,
140 menus with two presentations, and 5,040 observations. A one-chain fit with
100 warmup and 100 retained draws finished in about 4.5 minutes, had zero
divergences and satisfactory treedepth/E-BFMI, and estimated a fixed true
`gamma_size=0.309` with bias `-0.0113` and 90% interval width `0.0429`.
Non-finite-logit proposals were still rejected during warmup. The short
single-chain fit is geometry and timing evidence only: its R-hat values are not
interpretable, and it does not establish bias, coverage, or convergence.

Matched four-chain exact-shape validations subsequently used 500 warmup and 500
retained draws per chain. Both pools had zero divergences, maximum treedepth 7
of 12, no treedepth saturation, and satisfactory E-BFMI. Venture's minimum
structural-parameter bulk ESS was 1,416 and maximum R-hat was 1.0043; hiring's
were 1,469 and 1.0053. The fixed E1 slopes were locally recovered as follows:

| Pool | True `gamma_size` | Bias | 90% interval width | Covered |
|---|---:|---:|---:|---:|
| Venture | 0.309 | -0.0116 | 0.0482 | yes |
| Hiring | 0.019 | -0.0130 | 0.0423 | yes |

Sampling took 11.8 minutes for venture and 10.5 minutes for hiring. Each fit
logged a small number of non-finite-logit proposal rejections during initial
warmup (26 and 28 respectively), with none after retained sampling began.
Compressed chain CSVs were preserved. These runs pass the production-geometry
and convergence check for their simulated datasets, including the lower-spread
hiring assessments. They do not estimate repeated-sampling bias or coverage;
the recovery/SBC campaign decision therefore remains open.

A matched two-dataset pilot then drew `gamma_size` from its amended
`normal(0, 0.2)` prior for each dataset. Four chains with 250 warmup and 250
retained draws were rejected as a campaign setting: venture maximum R-hat was
1.0208 and 1.0162, and one dataset had minimum structural bulk ESS 274. At 500
warmup and 500 retained draws, both venture and both hiring datasets passed the
prespecified sampler gates (R-hat below 1.01, structural bulk ESS at least 400,
E-BFMI at least 0.3, zero divergences, and zero treedepth saturation). This
included a prior draw with `sigma_cell=0.012`, close to the boundary.

The completed recovery campaigns used 40 datasets per pool with four chains and
500 warmup plus 500 retained draws. Five hiring fits and four venture fits that
narrowly missed a prespecified sampler gate were replaced by 1,000-warmup,
1,000-retained-draw fits of the same deterministic datasets. The original fits
were archived, and hashes confirmed that every replacement retained its exact
simulated parameters. Both final campaigns passed all gates for all 40 datasets:

| Pool | Passed | Maximum R-hat | Minimum bulk ESS | Minimum E-BFMI | Divergences | Treedepth saturation |
|---|---:|---:|---:|---:|---:|---:|
| Venture | 40/40 | 1.00925 | 425.273 | 0.6502 | 0 | 0 |
| Hiring | 40/40 | 1.00939 | 411.571 | 0.6990 | 0 | 0 |

The primary size-slope recovery results were:

| Pool | Bias | RMSE | 90% coverage | Mean interval width |
|---|---:|---:|---:|---:|
| Venture | -0.00069 | 0.01359 | 0.900 | 0.04136 |
| Hiring | -0.00270 | 0.01368 | 0.925 | 0.04006 |

At nominal 90% coverage, the Monte Carlo standard error from 40 datasets is
0.047. The observed `gamma_size` coverages are therefore consistent with the
nominal target at this resolution. Total recorded fit time was 5.57 hours for
venture and 5.79 hours for hiring. The repeated-recovery requirement is
complete. Amendment 4 declines a separate formal SBC campaign in favor of
these direct production-geometry recovery results and requires deterministic
presentation-1-only and presentation-2-only fits after collection.

The confirmatory contract now verifies RQ3 against the exact production design.
The additive intercept-plus-design matrix has rank 8; the 10 model-by-prompt
columns have residualized rank 10 and bring the 18-cell matrix to rank 18. Thus
the full interaction exactly spans the cell-residual space already represented
by `sigma_cell * z_alpha`. RQ3 is frozen as secondary/descriptive from the
existing anchored fit; no separate saturated fit will be treated as additional
likelihood evidence.

RQ5 preparation now builds a dedicated assessment-anchored re-slice after both
source pools complete Stan-data preparation. The frozen artifacts contain 24
matched merit keys and 40 menus per task; every procurement menu and its hiring
counterpart have identical matched-key sets and order. The joint design has 36
cells and rank 14 including its intercept, identifying six within-model
hiring-minus-procurement contrasts. Because the amended model consumes fixed
expected utilities rather than embeddings, the former joint-PCA requirement is
obsolete. Primary and `u=0.35/0.65` sensitivity payloads are emitted under
`matched_rq5/`. The recovery harness can reconstruct the exact generated design
and fixed `eta` from that payload and scores all six named task contrasts from
joint posterior draws. A deterministic preproduction builder removes any
dependency on uncollected choice outcomes: placeholder choices encode only the
frozen menus and presentations, while the simulator supplies `y` for recovery.

The exact 36-cell validation fit used 2,880 observations, four chains, 500
warmup draws, and 500 retained draws. It completed in 257 fit-seconds and passed
all sampler gates: maximum R-hat 1.00608, minimum structural bulk/tail ESS
1,045.5/1,092.6, minimum E-BFMI 0.714, zero divergences, and zero treedepth
saturation (maximum depth 8 of 12). All six true RQ5 contrasts fell inside their
joint-draw central 90% intervals.

The repeated RQ5 recovery campaign used 40 deterministic datasets. Ten initial
500-warmup/500-retained fits missed an R-hat or ESS gate and were replaced by
1,000/1,000 fits of the same datasets; archived-versus-replacement hashes match
for every simulated truth. The final campaign passed all gates for 40/40
datasets: maximum R-hat 1.00996, minimum structural bulk/tail ESS 442.6/562.3,
minimum E-BFMI 0.657, zero divergences, and zero treedepth saturation. Across
the six within-model hiring-minus-procurement contrasts, absolute bias ranged
from 0.0177 to 0.0414, RMSE from 0.1820 to 0.2573, central 90% coverage from
0.850 to 0.950, and mean interval width from 0.5430 to 0.8174. With 40 datasets,
the Monte Carlo standard error of nominal 90% coverage is 0.047; the observed
contrast coverages are consistent with that target at this resolution. The RQ5
simulation/recovery gate is complete; final contrast-table reporting remains.

All maintained application tests pass (498 tests). Repository-wide pytest last
passed 475 tests but reported three unrelated collection errors from the legacy
executable `scripts/test_m1_model.py`, whose helper functions are named
`test_*` but require command-line arguments rather than pytest fixtures.

The detailed status of every independent-review finding is tracked in
`REVIEW_DISPOSITION.md`. Open Batch reconciliation, all-attempt budget control,
and production-root preflight continue to block a GO.

## Batch and usage-persistence gate

The approved production mode is provider Batch collection with a $31 ceiling.
The frozen configuration now selects `collection_mode: batch`; other configs
retain the synchronous default. Choice collection supports both OpenAI JSONL
file batches and Anthropic Message Batches. Each cell has an atomic state file
containing its provider batch ID and a digest of the full request set. A first
invocation submits once and returns `batch_pending`; later invocations retrieve
that same batch. Prompt or model drift after submission is rejected rather than
silently attaching old paid results to a changed design.

Results are joined through deterministic `custom_id` values, never provider
output order. Terminal batch failures, per-request failures, incomplete result
sets, and unexpected result IDs stop collection explicitly. Successful results
flow through the same choice parser, presentation-order resolution, schema
validation, and checkpoint format as synchronous results.

Usage is written to `usage_events.jsonl` before each final assessment or choice
artifact is published. Events have deterministic IDs, are append-only and
idempotent on resume, and are flushed with `fsync`. Batch events retain input,
output, cached-input, reasoning, cache-creation, and cache-read token fields
when exposed by the provider, plus the provider, model, request count, Batch
discount, and estimated cost. Later phase summaries cannot overwrite them.

Mocked contract tests cover submit/resume behavior, request-digest mismatch,
OpenAI request JSONL, Anthropic extended-thinking request parameters,
out-of-order results, incomplete output, usage aggregation, collector
presentation mapping, and preservation of the synchronous path.

## Live provider Batch probe

On 2026-09-07, a paid probe submitted two 64-token-capped choice requests to
each provider's cheapest configured production endpoint. Both initial calls
returned pending provider batch IDs. Later invocations resumed those IDs rather
than resubmitting requests, and both providers reached their successful terminal
state. Each provider returned the requested `ANSWER: 1` and `ANSWER: 2` values
under the correct custom IDs.

| Provider | Endpoint | Calls | Input tokens | Output tokens | Recorded Batch cost |
|---|---|---:|---:|---:|---:|
| OpenAI | `gpt-4o-mini` | 2 | 82 | 8 | $0.00000855 |
| Anthropic | `claude-haiku-4-5-20251001` | 2 | 74 | 16 | $0.00007700 |
| **Total** | | **4** | **156** | **24** | **$0.00008555** |

The provider state files contain the terminal statuses and usage summaries;
the append-only ledger contains exactly one matching event per provider. No
cached-input, cache-creation, cache-read, or reasoning tokens were reported for
these non-reasoning probe arms. The measured total was below the authorized $1
probe ceiling.

## Verdict

**SUPERSEDED: the 2026-09-06 technical gate returned GO.** At that time, the
frozen gates, offline call count, application tests, and limited live
provider-specific Batch probe passed. Subsequent review showed that those
checks did not establish scientific identification or cover critical Batch
failure modes.

**Current verdict: NOT READY.** Production choice collection is stopped. A new
GO requires validation of the assessment-anchored model, completion of the
confirmatory analysis contract, correction of the Batch integrity and
all-attempt accounting failures, and an exact-root production preflight.
Passing those gates will still not authorize production spending: a new
explicit authorization will be required for the 10,080-call Batch run.

The synchronous path is technically validated, but using it would invoke the
$61 fallback and also requires separate approval. No production calls have
been launched.