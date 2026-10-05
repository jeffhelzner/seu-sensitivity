# Independent Review Disposition

> This is the September review record. The current October review and remaining
> launch conditions are tracked in [the October disposition](REVIEW_DISPOSITION_20261005.md).

Date: 2026-09-07

Source review: `local/seu_sensitivity_frontier_model_review.md`

Current verdict: **NOT READY.** This document records implementation progress;
it does not authorize production collection or supersede the preregistration.

Documentation corrected 2026-09-11 using the verified offline remediation
record and [exact-render, gate, and recovery audit](PHASE_E4_VALIDATION.md#verified-offline-audit-2026-09-11),
then updated to record the separately authorized venture iteration-16 rerun
completed that day. The preceding offline pass involved no new fits. This
documentation-only update made no provider calls, created no environment,
ran no production preflight/staging or fits, and altered no production artifacts.

## Budget approval (2026-09-11)

The user said: "ok, i give my approval on the budget. let's continue".
The current `configs/preregistered.yaml` raises `batch_choice_budget_usd`
from the historical 31 to **122.78**, the calculated $122.7729167015873
reservation rounded up to cents for the frozen **10,080-request campaign**.
The budget blocker is resolved under the current conditional pricing/protocol
liability assumptions, not as a provider billing guarantee or authorization
for replacement attempts. The flat per-request floor remains
`0.0030753968253968253`; design, token limits, `batch_wave_id: null`, and
`batch_wave_cell_ids: []` are unchanged. Recovery remains **120/120**.
No calls were launched for this approval update. The worktree is dirty and
the user has not authorized a Git commit. A clean committed revision, fresh
production staging and manifest, separately authorized preflight, and explicit
wave and production launch authorization remain pending: **NOT READY**.
Historical $31 budget and audit cost figures are retained below.

| Finding | Disposition | Evidence and remaining work |
|---|---|---|
| F1: alpha identification | Accepted; amended estimand implemented, recovery shortfall closed | The primary model conditions on assessment-derived `eta[J,R]` and contains no latent `beta`. Prior predictive checks against observed assessment spread calibrated the size-slope prior. The authorized venture iteration-16 rerun completed 2026-09-11; verified all-gate counts are now 40/40 venture, 40/40 hiring, and 40/40 matched RQ5 (120/120). All have zero divergences and no treedepth saturation. The historical 119/120 audit and iteration-16 `sigma_cell` tail ESS 310.525 were genuine; the earlier all-pass claim was false at that checkpoint. Previously recorded central 90% coverages are historical before-rerun evidence and were not recomputed for this update. The dated E4 rerun record preserves diagnostics and provenance. |
| F2: RQ4 estimand mismatch | Accepted; remediated by scientific scope change | Amendment 1 makes RQ4 descriptive for the two selected domains. Reporting uses independent posterior differences, not index-paired draws, and includes all 15 model-pair rankings and same-sign probabilities across pools. It does not use `sigma_cell` as cross-pool variance, make a population-of-domains claim, or add confirmatory hypotheses. |
| F3: ambiguous/duplicate submission | Accepted; remediated within provider capabilities | Both SDK clients now set `max_retries=0`. Batch state records a durable random submission intent before provider creation; the state file is fsynced before atomic replacement and its parent directory afterward. OpenAI metadata binds intent, cell, and full request hash; recovery requires exactly one matching batch. Failure-injection covers provider acceptance before batch-ID persistence. OS advisory locking rejects live concurrent writers and releases on process death. Anthropic lacks request-bound listing metadata, so automatic reconciliation remains impossible; the operator command only attaches an existing provider ID to locked ambiguous state, validates provider/model and request count, records evidence, and never creates a replacement. These controls are not proof that historical remote duplicates did not occur. |
| F4: incomplete identity and duplicate IDs | Accepted; remediated for identified paths | The hash covers exact provider-ready bodies, including reasoning reserve and transmitted thinking temperature. Duplicate result IDs are rejected. Batch state, checkpoints, final choice artifacts, and the F7 preflight manifest share the full request hash, including the fully-complete cache path. The reasoning probe now also validates completed-cache identity before reuse. |
| F5: partial evidence and budget accounting | Accepted; controls implemented, campaign budget blocker resolved, historical evidence limitation remains | OpenAI output/error files, batch-level errors, aggregate usage, partial responses, failed/duplicate IDs, and raw records are persisted before integrity errors; successful unambiguous rows are checkpointed before re-raising. Future Anthropic results retain raw messages, errors, and usage. Historical live Sonnet raw message metadata was omitted and is irrecoverable offline. The user approved the $122.78 ceiling on 2026-09-11, replacing the historical $31 ceiling. Every new intent receives an fsynced append-only reservation based on rendered UTF-8 message-content bytes plus 1,024 per message plus 1,024 request overhead, exact output caps, pinned input/output pricing, and the 0.5 Batch multiplier. The historical `$31 / 10,080` per-request floor remains `0.0030753968253968253`. The verified 10,080-request audit gives $122.7729167015873 reserved ($118.18225175 before the floor) and $61.867008 maximum output-only, not a spend forecast. The E4 audit contains the per-arm table and exact inputs. This is conditional on the pricing/protocol assumptions, not a provider billing guarantee. Both amounts exceeded the historical $31 ceiling; the approved $122.78 now covers the frozen campaign reservation under those assumptions, without changing token limits/design. One intent per cell is allowed; reservations are idempotent, never silently released, and reject over-ceiling attempts and malformed/truncated ledgers. Usage reports leave missing costs unresolved, not zero. A named wave and cell allowlist remain required and unauthorized; replacements require separate wave and budget approval. |
| F6: confirmatory analysis freeze | Accepted; pre-analysis implementation complete | Amendment 3 and `analysis_contract.json` freeze 26 primary decisions: nine RQ1/RQ2 plus one RQ6 per pool, and six matched RQ5, with central 90% interval-plus-ROPE rules, structural bulk/tail ESS minima of 400, and no multiplicity adjustment. Strict schema-version-1 reporting requires all 15 fits across venture, hiring, and matched RQ5 and their five variants. Each entry binds an absolute chain directory, all four basename-to-SHA-256 chain hashes, and absolute-path/SHA-256 objects for Stan data, its per-variant preparation report, and the analysis contract. Top-level keys are only `schema_version`, `max_treedepth` (12), and `fits`; the complete placeholder template is in `PHASE_E4_VALIDATION.md`. Reporting checks finite structural values/diagnostics, actual four-chain/treedepth-12 metadata, retained design rank after exclusions, posterior consistency, and PPC summaries. Input binding is declared provenance, not execution proof. RQ3 reports raw residuals and model-by-prompt difference-in-differences; RQ4 remains descriptive. Presentation-only and utility-grid comparisons are mandatory; the position-stable subset remains separate. Synthetic tests exercise reporting without fits or provider access, but were not rerun for this documentation update. |
| F7: unstaged production root/preflight | Accepted; implementation complete, production execution pending | The offline preflight command refreshes configured gates, renders and archives exact provider-ready bodies plus observation mappings, and creates a single-use read-only wave stage. Its aggregate-hashed manifest binds the clean Git commit, complete application Python source surface, active Stan/config sources, toolchain versions, exact configuration, source and staged artifacts, prompts, fresh gates, cell authorization, per-cell request hashes, and request counts. Reservation re-verifies every binding before ledger append or submission. The production YAML remains no-spend, so execution against an authorized production wave remains pending and no production GO is claimed. |
| F8: whole-cell NA exclusion | Accepted; implemented | Cells above 30% NA are excluded before Stan assembly; cell IDs, design rows, and model mappings are subset and reindexed together. Excluded cells are reported, and every anchored payload fails if exclusions make the intercept-plus-design matrix rank deficient. Each pool and matched RQ5 variant writes `stan_data_size[_u035\|_u065\|_presentation_1\|_presentation_2]_assembly_report.json`, retaining `cell_ids`, `design_columns`, `rank`, and `presentation_id`; the bracketed suffix is notation, not a literal filename. Exact filenames are listed in E4. |
| F9: zero-retention menu size | Accepted; implemented | Balancing includes every designed size, including zero-retention bins; one empty size therefore yields zero balanced menus instead of silently changing support. |
| F10: evidence scope/dependence | Accepted; dependence sensitivities implemented, recovery shortfall closed | Prior predictive checks using persisted assessments and exact menus motivated narrowing the size-slope prior from SD 0.5 to 0.2. The authorized venture iteration-16 rerun completed 2026-09-11 closes the historical tail-ESS shortfall: verified all-gate counts are now 40/40 venture, 40/40 hiring, and 40/40 matched RQ5 (120/120). The earlier 119/120 audit is retained as historical evidence. Amendment 4 still declines a separate formal SBC campaign in favor of direct production-geometry recovery; no new SBC or power claim is made. Required presentation-1-only and presentation-2-only primary-utility fits for each pool and matched RQ5 remove within-menu duplication without conditioning on observed agreement. The 15 mandatory fits replace the old six-fit schedule, while position-stable-subset robustness remains separately required per preregistration. |
| F11: gate interpretation/ridge LOO | Accepted; numerical defect fixed, offline gates verified, production staging pending | Ridge LOO leverage includes the unpenalized intercept and is tested against explicit refits. On 2026-09-11, pure `item_validation.run_gate` recomputation in memory from saved inputs and fresh sibling summaries gave pooled LOO R-squared 0.8232 venture and 0.6986 hiring; worst cross-size gaps are 0.0906 and 0.0484, and the cross-pool gap is 0.0601. Both pass the frozen 0.30 R-squared and 0.25 gap thresholds, both use LDA fallback, and both have zero assessment parse failures. This is offline evidence, not production preflight. Saved historical gate reports remain stale; fresh production staging, its manifest, and preflight remain pending. Insurance's R-squared 0.052 is a failed screen, not proof of alpha unidentification; structural eta/alpha identification is a separate question. |
| F12: reasoning treatment/probe | Accepted; live probe complete, historical evidence limits retained | The manifest records effective Anthropic thinking temperature and OpenAI reasoning reserve. Sonnet thinking uses budget 4,096, total output cap 4,160, and temperature 1 versus thinking-off temperature 0: a treatment bundle, not a pure thinking causal effect. Authorized live Batches exercised both frozen presentations of the same size-8 venture menu per reasoning arm. All four requests returned visible, parseable `ANSWER:` tokens under expected custom IDs with no failed or duplicate returned IDs; each arm selected the same underlying item across orders. Durable state, exact provider-body hashes, pre-submission reservations, retained result records, and one idempotent usage event per arm remain genuine evidence. Historical Sonnet raw message metadata was omitted and is irrecoverable offline; future raw-result persistence does not backfill it. Measured Batch cost was $0.02277980 under the $1 ceiling. This successful probe does not prove absence of remote duplicates or authorize production. |

## Validation Recorded With This Disposition

### Latest recovery status: authorized rerun completed 2026-09-11

Verified `_summarize_sampler_diagnostics` results are **120/120 all-gate
passes**: venture 40/40, hiring 40/40, and matched RQ5 40/40. Only venture
iteration 16 was rerun, with four chains, 1,000 warmup and 1,000 retained
sampling draws per chain and maximum treedepth 12 under the existing rerun
configuration. Maximum R-hat was 1.00295, minimum structural bulk/tail ESS
1,546.3/1,960.77, and minimum E-BFMI 0.7462447584, with zero divergences,
zero saturation, and maximum depth reached 7. The 24 non-finite rejected
proposals are reported, not a frozen pass/fail gate. Fit time was
525.0818288 seconds; total logged runtime including postprocessing was
approximately 13 minutes 21 seconds.

The [E4 completion record](PHASE_E4_VALIDATION.md#authorized-venture-iteration-16-rerun-completed-2026-09-11)
details the original-artifact archive, prelaunch design/configuration checks,
seeds, and rebuilt Stan executables. Original true-parameter values match
exactly numerically, with only empty `contrasts` metadata added; this does not
claim byte-identical truth files or directly verified simulated `y` bytes.
All 117 hashes covering the other 39 venture fits' truth, diagnostics, and
posterior summaries were verified unchanged. The refreshed venture aggregate
matches recomputation; no other campaigns were rewritten. Historical coverage
estimates were not recomputed. No provider calls, budget change, production
preflight/staging, or environment creation accompanied the rerun.

### Historical before-rerun offline audit: 2026-09-11

- Latest full application suite: **678 passed**, followed by **119 focused
  tests**; the focused count is separate, not an additional full-suite total.
- Runtime loader fix for the exact known `_model` suffix: **117 tests passed**.
  CSV headers sampled in each recovery group verified the actual model name
  `h_m01_size_assessment_anchored_model` and maximum depth 12.
- Historical saved-summary recovery audit: **119/120** passed, comprising venture 39/40,
  hiring 40/40, and matched RQ5 40/40. Venture iteration 16 `sigma_cell` tail
  ESS was 310.525, below 400; this preceding offline pass ran no new fits.
- Exact rendering and in-memory gates are recorded in the E4 audit. No provider
  access, environment creation, production artifact changes, or production
  staging/preflight was involved.

These results were supplied by the verified offline audit; tests were not rerun
for this documentation-only update.

### Historical checkpoints

The following older results are retained separately, not current suite counts
or tests or fits executed for this documentation-only correction:

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

The separately authorized four-request reasoning-arm probes recorded above
remain successful historical evidence; the earlier non-reasoning probe is
also preserved in `PHASE_E4_VALIDATION.md`. No new calls were made for this
update. The authorized rerun closed the venture recovery gate shortfall. Offline gates
and full-campaign liability are now verified, and the user's 2026-09-11 budget
approval resolves the budget blocker under the current conditional liability
assumptions. Fresh production staging and its manifest, a clean committed
revision (commit not yet authorized), separately authorized preflight, and
explicit wave and production launch authorization remain pending.
**NOT READY** remains the verdict; no production wave was authorized or launched.