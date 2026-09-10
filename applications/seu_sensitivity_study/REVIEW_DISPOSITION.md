# Independent Review Disposition

Date: 2026-09-07

Source review: `local/seu_sensitivity_frontier_model_review.md`

Current verdict: **NOT READY.** This document records implementation progress;
it does not authorize production collection or supersede the preregistration.

| Finding | Disposition | Evidence and remaining work |
|---|---|---|
| F1: alpha identification | Accepted; remediated for the amended estimand | The primary model conditions on assessment-derived `eta[J,R]` and contains no latent `beta`. Prior predictive checks against observed assessment spread calibrated the size-slope prior. Exact production-dimensional validation and separate 40-dataset venture, hiring, and matched RQ5 recovery campaigns passed all sampler gates with coverage consistent with the central 90% target at the available Monte Carlo resolution. |
| F2: RQ4 estimand mismatch | Accepted; scientific scope changed | Amendment 1 makes RQ4 descriptive for the two selected domains. No population-of-domains variance claim will be made. Reporting code and the final descriptive summaries remain to be specified. |
| F3: ambiguous/duplicate submission | Accepted; partially remediated | Batch state records a durable random submission intent before provider creation. OpenAI provider metadata binds that intent to the cell and full request hash; ambiguous local state is recovered only from exactly one matching provider batch, and failure-injection verifies recovery after provider acceptance but before batch-ID persistence. OS advisory locking rejects live concurrent writers and is automatically released on process death, so stale lock files are reusable. Anthropic ambiguous submissions remain fail-closed for manual reconciliation because Message Batches expose no equivalent request-bound listing metadata. |
| F4: incomplete identity and duplicate IDs | Accepted; remediated for identified paths | The hash covers exact provider-ready bodies, including reasoning reserve and transmitted thinking temperature. Duplicate result IDs are rejected. Batch state, checkpoints, final choice artifacts, and the F7 preflight manifest share the full request hash, including the fully-complete cache path. |
| F5: partial evidence and budget accounting | Accepted; precollection controls implemented | OpenAI output and error files, batch-level errors, aggregate usage, partial responses, failed IDs, duplicate IDs, and raw records are persisted before integrity errors. Successful unambiguous rows are parsed into a request-bound checkpoint before the error is re-raised. The `$31` choice ceiling is machine-readable, and every new Batch intent receives an fsynced append-only reservation before provider submission at `$31 / 10,080` per request. One intent per cell is allowed; reservations are idempotent by intent, never silently released, and block over-ceiling attempts, while malformed or truncated ledgers fail closed. A deterministic report joins reservations to known usage-estimated costs and leaves missing usage unresolved rather than zero. Submission also requires a named wave and explicit cell allowlist; the preregistered production config leaves both unauthorized. Any replacement attempt requires a separately reviewed wave and budget amendment. |
| F6: confirmatory analysis freeze | Accepted; partially remediated | Amendment 3 and `analysis_contract.json` freeze central 90% interval-plus-ROPE decisions, seven RQ1 and two RQ2 contrasts per pool, bulk/tail ESS minima of 400, and the full-family/no-adjustment multiplicity policy. The `0.35/0.50/0.65` utility grid emits executable Stan-data variants, and every anchored payload stops if whole-cell exclusions make the additive design rank deficient. The contract verifies that RQ3's 10 interaction columns exactly span the 10 residual cell dimensions, so RQ3 is secondary/descriptive from the existing cell-residual posterior rather than a redundant saturated fit. RQ5 preparation emits a dedicated rank-14, 36-cell assessment-anchored re-slice over 24 matched items and 40 paired menus per task. Its exact validation and 40-dataset recovery campaign passed every sampler gate; all six task contrasts had central 90% coverage between 0.850 and 0.950. Final contrast-table reporting remains open. |
| F7: unstaged production root/preflight | Accepted; implementation complete, production execution pending | The offline preflight command refreshes configured gates, renders exact provider-ready requests, and creates a single-use read-only wave stage. Its aggregate-hashed manifest binds configuration, source and staged artifacts, prompts, fresh gates, cell authorization, and per-cell request hashes. The reservation callback re-verifies every binding immediately before ledger append or submission. Offline tests cover staging and fail-closed tamper paths. The production YAML remains no-spend, so preflight against an authorized production wave remains pending and no production GO is claimed. |
| F8: whole-cell NA exclusion | Accepted; implemented | Cells above 30% NA are excluded before Stan assembly; cell IDs, design rows, and model mappings are subset and reindexed together. Excluded cells are reported. Contrast-estimability checks after exclusions remain under F6. |
| F9: zero-retention menu size | Accepted; implemented | Balancing includes every designed size, including zero-retention bins; one empty size therefore yields zero balanced menus instead of silently changing support. |
| F10: evidence scope/dependence | Accepted; precollection requirements implemented | Prior predictive checks using persisted assessments and exact menus motivated narrowing the size-slope prior from SD 0.5 to 0.2. The completed 40-dataset venture, hiring, and matched RQ5 campaigns passed every prespecified sampler gate after deterministic longer reruns. Amendment 4 declines a separate formal SBC campaign in favor of direct production-geometry recovery. The pipeline emits presentation-1-only and presentation-2-only primary-utility payloads for both pool-specific and matched RQ5 fits; these remove within-menu duplication without conditioning on observed agreement and are required post-collection sensitivity analyses. |
| F11: gate interpretation/ridge LOO | Accepted; numerical defect fixed | Ridge LOO leverage now includes the unpenalized intercept and is tested against explicit refits. Gate passes remain screens rather than identification proofs; fresh reports must be regenerated in preflight. |
| F12: reasoning treatment/probe | Accepted; partially remediated | The manifest records the effective Anthropic thinking temperature and OpenAI reasoning reserve. The contrast must be reported as a treatment bundle. No live reasoning-arm Batch probe has been authorized or run. |

## Validation Recorded With This Disposition

- Batch contract tests: 7 passed.
- Client, Batch, and provenance module: 45 passed.
- Checkpoint/final-artifact identity slice: 20 passed.
- Anchored Stan-variant and fixed-eta generation tests: 7 passed.
- All maintained application tests: 522 passed.
- Repository-wide pytest discovery: 475 passed, with three unrelated collection
  errors in `scripts/test_m1_model.py` because executable helper functions named
  `test_*` require non-pytest arguments.
- Both anchored Stan models compiled successfully.
- One synthetic anchored recovery iteration completed. It is a plumbing smoke
  test only; 200 draws produced expected short-run R-hat warnings and rejected
  overflow proposals during warmup, with zero divergences and satisfactory
  treedepth and E-BFMI.

No provider calls were made while addressing the review.