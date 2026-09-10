# Independent Review Disposition

Date: 2026-09-07

Source review: `local/seu_sensitivity_frontier_model_review.md`

Current verdict: **NOT READY.** This document records implementation progress;
it does not authorize production collection or supersede the preregistration.

| Finding | Disposition | Evidence and remaining work |
|---|---|---|
| F1: alpha identification | Accepted; remediated for the amended estimand | The primary model conditions on assessment-derived `eta[J,R]` and contains no latent `beta`. Prior predictive checks against observed assessment spread calibrated the size-slope prior. Exact production-dimensional validation and separate 40-dataset venture, hiring, and matched RQ5 recovery campaigns passed all sampler gates with coverage consistent with the central 90% target at the available Monte Carlo resolution. |
| F2: RQ4 estimand mismatch | Accepted; remediated by scientific scope change | Amendment 1 makes RQ4 descriptive for the two selected domains. The reporting path presents each frozen contrast by domain, its median difference, and sign agreement without pairing independent posterior draws, using `sigma_cell` as cross-pool variance, or making a population-of-domains claim. |
| F3: ambiguous/duplicate submission | Accepted; remediated within provider capabilities | Batch state records a durable random submission intent before provider creation. OpenAI provider metadata binds that intent to the cell and full request hash; ambiguous local state is recovered only from exactly one matching provider batch, and failure-injection verifies recovery after provider acceptance but before batch-ID persistence. OS advisory locking rejects live concurrent writers and is automatically released on process death. Anthropic lacks request-bound listing metadata, so automatic reconciliation remains impossible; the audited operator command only attaches an existing provider ID to a locked ambiguous state, checks provider/model and request count when exposed, records the evidence, and never creates a replacement. |
| F4: incomplete identity and duplicate IDs | Accepted; remediated for identified paths | The hash covers exact provider-ready bodies, including reasoning reserve and transmitted thinking temperature. Duplicate result IDs are rejected. Batch state, checkpoints, final choice artifacts, and the F7 preflight manifest share the full request hash, including the fully-complete cache path. |
| F5: partial evidence and budget accounting | Accepted; precollection controls implemented | OpenAI output and error files, batch-level errors, aggregate usage, partial responses, failed IDs, duplicate IDs, and raw records are persisted before integrity errors. Successful unambiguous rows are parsed into a request-bound checkpoint before the error is re-raised. The `$31` choice ceiling is machine-readable, and every new Batch intent receives an fsynced append-only reservation before provider submission at `$31 / 10,080` per request. One intent per cell is allowed; reservations are idempotent by intent, never silently released, and block over-ceiling attempts, while malformed or truncated ledgers fail closed. A deterministic report joins reservations to known usage-estimated costs and leaves missing usage unresolved rather than zero. Submission also requires a named wave and explicit cell allowlist; the preregistered production config leaves both unauthorized. Any replacement attempt requires a separately reviewed wave and budget amendment. |
| F6: confirmatory analysis freeze | Accepted; pre-analysis implementation complete | Amendment 3 and `analysis_contract.json` freeze central 90% interval-plus-ROPE decisions, seven RQ1 and two RQ2 contrasts per pool, six matched RQ5 contrasts, RQ6, bulk/tail ESS minima of 400, and the full-family/no-adjustment policy. The reporting command reconstructs and hashes saved CmdStan chains, enforces every sampler gate, computes contrasts draw by draw, emits descriptive RQ3/RQ4 sections, compares primary with both frozen presentation-only fits, and separately reports the full-data `u=0.35/0.65` utility grid. Synthetic tests exercise all sections without fitting or provider access. |
| F7: unstaged production root/preflight | Accepted; implementation complete, production execution pending | The offline preflight command refreshes configured gates, renders and archives exact provider-ready bodies plus observation mappings, and creates a single-use read-only wave stage. Its aggregate-hashed manifest binds the clean Git commit, complete application Python source surface, active Stan/config sources, toolchain versions, exact configuration, source and staged artifacts, prompts, fresh gates, cell authorization, per-cell request hashes, and request counts. Reservation re-verifies every binding before ledger append or submission. The production YAML remains no-spend, so execution against an authorized production wave remains pending and no production GO is claimed. |
| F8: whole-cell NA exclusion | Accepted; implemented | Cells above 30% NA are excluded before Stan assembly; cell IDs, design rows, and model mappings are subset and reindexed together. Excluded cells are reported, and every anchored payload fails if exclusions make the intercept-plus-design matrix rank deficient. |
| F9: zero-retention menu size | Accepted; implemented | Balancing includes every designed size, including zero-retention bins; one empty size therefore yields zero balanced menus instead of silently changing support. |
| F10: evidence scope/dependence | Accepted; precollection requirements implemented | Prior predictive checks using persisted assessments and exact menus motivated narrowing the size-slope prior from SD 0.5 to 0.2. The completed 40-dataset venture, hiring, and matched RQ5 campaigns passed every prespecified sampler gate after deterministic longer reruns. Amendment 4 declines a separate formal SBC campaign in favor of direct production-geometry recovery. The pipeline emits presentation-1-only and presentation-2-only primary-utility payloads for both pool-specific and matched RQ5 fits; these remove within-menu duplication without conditioning on observed agreement and are required post-collection sensitivity analyses. |
| F11: gate interpretation/ridge LOO | Accepted; numerical defect fixed | Ridge LOO leverage now includes the unpenalized intercept and is tested against explicit refits. Gate passes remain screens rather than identification proofs; fresh reports must be regenerated in preflight. |
| F12: reasoning treatment/probe | Accepted; partially remediated | The manifest records the effective Anthropic thinking temperature and OpenAI reasoning reserve. The contrast must be reported as a treatment bundle. No live reasoning-arm Batch probe has been authorized or run. |

## Validation Recorded With This Disposition

- Batch contract tests: 7 passed.
- Client, Batch, and provenance module: 45 passed.
- Checkpoint/final-artifact identity slice: 20 passed.
- Anchored Stan-variant and fixed-eta generation tests: 7 passed.
- All maintained application tests: 536 passed.
- Repository-wide pytest discovery: 475 passed, with three unrelated collection
  errors in `scripts/test_m1_model.py` because executable helper functions named
  `test_*` require non-pytest arguments.
- Both anchored Stan models compiled successfully.
- One synthetic anchored recovery iteration completed. It is a plumbing smoke
  test only; 200 draws produced expected short-run R-hat warnings and rejected
  overflow proposals during warmup, with zero divergences and satisfactory
  treedepth and E-BFMI.

No provider calls were made while addressing the review.