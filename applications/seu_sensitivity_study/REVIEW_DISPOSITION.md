# Independent Review Disposition

Date: 2026-09-07

Source review: `local/seu_sensitivity_frontier_model_review.md`

Current verdict: **NOT READY.** This document records implementation progress;
it does not authorize production collection or supersede the preregistration.

| Finding | Disposition | Evidence and remaining work |
|---|---|---|
| F1: alpha identification | Accepted; partially remediated | The primary model now conditions on assessment-derived `eta[J,R]` and contains no latent `beta`. A matching fixed-eta simulator compiles and completed one synthetic smoke recovery. Production-dimensional recovery, prior calibration against observed assessment spread, and prior predictive checks remain required. |
| F2: RQ4 estimand mismatch | Accepted; scientific scope changed | Amendment 1 makes RQ4 descriptive for the two selected domains. No population-of-domains variance claim will be made. Reporting code and the final descriptive summaries remain to be specified. |
| F3: ambiguous/duplicate submission | Accepted; partially remediated | Batch state now records a durable `submitting` intent before provider creation, uses a single-writer lock, and stops in `submission_ambiguous` rather than resubmitting. Provider-side reconciliation, stale-lock recovery, and post-acceptance disk-failure tests remain open. |
| F4: incomplete identity and duplicate IDs | Accepted; remediated for identified paths | The hash covers exact provider-ready bodies, including reasoning reserve and transmitted thinking temperature. Duplicate result IDs are rejected. Batch state, checkpoints, and final choice artifacts share the full request hash, including the fully-complete cache path. A broader production-root identity is still part of F7. |
| F5: partial evidence and budget accounting | Accepted; partially remediated | Partial responses, usage, failed IDs, duplicate IDs, and raw result records are persisted before integrity errors. OpenAI error-file retrieval, terminal failed-batch salvage, all-attempt accounting, truncated-ledger handling, and hard reservation/wave controls remain open. |
| F6: confirmatory analysis freeze | Accepted; partially remediated | RQ4 is descriptive and the `0.35/0.50/0.65` utility grid now emits executable Stan-data variants. Exact RQ1-RQ3 contrasts, RQ5, interval/ROPE rules, ESS minima, multiplicity policy, and post-exclusion estimability checks remain open. |
| F7: unstaged production root/preflight | Accepted; partially remediated | Provenance now records reasoning reserve and configured versus effective temperature. Immutable prerequisite staging, source/artifact hashes, exact request rendering, fresh gate binding, authorization binding, and enforced preflight remain open. |
| F8: whole-cell NA exclusion | Accepted; implemented | Cells above 30% NA are excluded before Stan assembly; cell IDs, design rows, and model mappings are subset and reindexed together. Excluded cells are reported. Contrast-estimability checks after exclusions remain under F6. |
| F9: zero-retention menu size | Accepted; implemented | Balancing includes every designed size, including zero-retention bins; one empty size therefore yields zero balanced menus instead of silently changing support. |
| F10: evidence scope/dependence | Accepted; partially remediated | A fixed-eta anchored simulator and smoke recovery now exist. The smoke fit had zero divergences and satisfactory treedepth/E-BFMI, but it is not calibration evidence. Production-dimensional recovery/SBC decision, prior predictive checks, and a dependence-aware presentation sensitivity remain open. |
| F11: gate interpretation/ridge LOO | Accepted; numerical defect fixed | Ridge LOO leverage now includes the unpenalized intercept and is tested against explicit refits. Gate passes remain screens rather than identification proofs; fresh reports must be regenerated in preflight. |
| F12: reasoning treatment/probe | Accepted; partially remediated | The manifest records the effective Anthropic thinking temperature and OpenAI reasoning reserve. The contrast must be reported as a treatment bundle. No live reasoning-arm Batch probe has been authorized or run. |

## Validation Recorded With This Disposition

- Batch contract tests: 7 passed.
- Client, Batch, and provenance module: 45 passed.
- Checkpoint/final-artifact identity slice: 20 passed.
- Anchored Stan-variant and fixed-eta generation tests: 7 passed.
- All maintained application tests: 474 passed.
- Repository-wide pytest discovery: 475 passed, with three unrelated collection
  errors in `scripts/test_m1_model.py` because executable helper functions named
  `test_*` require non-pytest arguments.
- Both anchored Stan models compiled successfully.
- One synthetic anchored recovery iteration completed. It is a plumbing smoke
  test only; 200 draws produced expected short-run R-hat warnings and rejected
  overflow proposals during warmup, with zero divergences and satisfactory
  treedepth and E-BFMI.

No provider calls were made while addressing the review.