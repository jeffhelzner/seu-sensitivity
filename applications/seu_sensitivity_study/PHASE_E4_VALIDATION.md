# Phase E4: Go/No-Go Validation (Superseded)

Date: 2026-09-06

Status amended 2026-09-07: **NOT READY.** The original technical GO below is
retained as a historical record of the checks completed at that time. An
independent review subsequently found a production-dimensional alpha
identification failure, an RQ4 estimand mismatch, and unresolved Batch
integrity and budget-control failures. The preregistration now specifies an
assessment-anchored primary estimand. E4 is reopened and must be rerun against
that amended model and a hardened collection pipeline before production.

## Budget approval (2026-09-11)

The user said: "ok, i give my approval on the budget. let's continue".
The current `configs/preregistered.yaml` raises `batch_choice_budget_usd`
from the historical 31 to **122.78**, rounding the calculated
$122.7729167015873 reservation up to cents for the frozen **10,080-request
campaign**. The budget blocker is resolved under the current conditional
pricing/protocol liability assumptions. This is not a provider billing
guarantee or replacement-attempt authorization. The flat per-request floor
remains `0.0030753968253968253`; design and token limits are unchanged, as
are `batch_wave_id: null` and `batch_wave_cell_ids: []`.
Recovery remains **120/120** all-gate passes. No calls were launched for this
approval update. The worktree is dirty; a Git commit is not yet authorized.
A clean committed revision, fresh production staging and manifest, separately
authorized preflight, and explicit wave and production launch authorization
remain pending. The verdict remains **NOT READY**. Historical $31 budget
and audit cost figures below are retained as historical evidence.

## Frozen-design checks

The explicit production configuration validates with 36 cells across venture
and hiring, 140 menus per pool, two presentations, and 10,080 expected choice
calls. The zero-cost pipeline dry run independently reports 5,040 calls per
pool and 280 observations per cell.

Both retained pools historically passed under the frozen embedding-axis
thresholds and variant-D recipes:

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

The application suite passed 460 tests at that historical checkpoint.

The ridge leave-one-out leverage formula has since been corrected to include
the unpenalized intercept. The 2026-09-11 offline recomputation below confirms
that both pools pass. Saved historical gate reports remain stale; fresh
production staging, its manifest, and production-wave preflight remain pending. Insurance's
historical R-squared of 0.052 fails this screen; it does not prove structural
alpha unidentification. The latent eta/alpha invariance is a separate issue.

## Verified offline audit (2026-09-11)

Status remains **NOT READY**. This historical offline pass preceded the
separately authorized recovery rerun recorded below. It used saved inputs under
`applications/seu_sensitivity_study/results/pools/{venture,hiring}/`:
`pool.json`, `problems.json`, `embeddings_reduced.npz`, and `assessments/*.json`.
Frozen configuration inputs, relative to the same application root, were
`configs/prompts_venture.yaml`, `configs/prompts_hiring.yaml`,
`configs/gate_thresholds.json`, and `configs/preregistered.yaml`. No production
artifacts were altered; no providers, new fits, environment creation, production
staging, or production preflight were involved.

Exact rendering produced `36 cells * 140 menus * 2 presentations = 10,080`
requests, or 1,680 per model arm. Summing the actual
`_batch_request_liability` reservations gives **$122.7729167015873**;
the total before the per-request floor is **$118.18225175**. The maximum
output-only amount is **$61.867008**, not a forecast of actual spend.

| Model arm | Requests | Output cap (tokens) | Pinned input/output rates ($/1M tokens) | Maximum output-only ($) | Reservation ($, rounded) |
|---|---:|---:|---:|---:|---:|
| GPT-4o | 1,680 | 64 | 2.5 / 10 | 0.5376 | 13.6766675 |
| GPT-4o-mini | 1,680 | 64 | 0.15 / 0.6 | 0.032256 | 5.166666667 |
| o3-mini | 1,680 | 2,112 | 1.1 / 4.4 | 7.805952 | 13.1449549 |
| Sonnet (thinking off) | 1,680 | 64 | 3 / 15 | 0.8064 | 16.611783 |
| Haiku | 1,680 | 64 | 1 / 5 | 0.2688 | 6.235195635 |
| Sonnet (thinking on) | 1,680 | 4,160 | 3 / 15 | 52.416 | 67.937649 |

The input bound is UTF-8 message-content bytes plus `1,024 * message_count`
plus 1,024 protocol overhead, hence bytes plus 3,072 for the actual two-message
requests. Output caps include reasoning/thinking allowances. Pinned pricing
uses the 0.5 Batch multiplier; the old flat `$31 / 10,080 = $0.003075...`
amount remains only a per-request floor. These reservations are conditional on
the pricing/protocol assumptions, not a provider billing guarantee or expected
spend. At this audit checkpoint, the output maximum alone exceeded the then
unchanged **$31 ceiling**, requiring user approval of greater liability or a
token-limit/design amendment. This audit selected neither policy and
authorized no wave or spending. The later budget approval recorded above
resolves that budget blocker without changing token limits or design.

The pure `item_validation.run_gate` was recomputed in memory from saved inputs
with fresh sibling summaries. Venture pooled LOO R-squared is **0.8232**, with
worst cross-size eta-gap difference **0.0906**; hiring is **0.6986**, with worst
gap **0.0484**. The cross-pool gap is **0.0601**. Both pass the frozen 0.30
R-squared minimum and 0.25 gap threshold, both use LDA fallback, and both have
zero assessment parse failures. This is fresh offline gate evidence, not
production preflight: saved historical gate reports remain stale and a fresh
staging manifest is still required.

The historical before-rerun saved-summary recovery audit found **119/120** fits passing:
venture 39/40 (iteration 16 `sigma_cell` tail ESS **310.525**, below 400),
hiring 40/40, and matched RQ5 40/40. Runtime CSV headers sampled in each group
verified the actual model name `h_m01_size_assessment_anchored_model` and
`max_depth=12`. The loader now handles the exact known `_model` suffix;
117 focused tests passed for that fix. The latest full application suite passed
**678 tests**, followed by **119 focused tests**. These are recorded offline
validation results, not tests rerun for this documentation-only update; older
counts elsewhere in this document remain historical checkpoints.

## Authorized venture iteration-16 rerun completed (2026-09-11)

The separately authorized rerun is complete. Verified
`_summarize_sampler_diagnostics` results now show **40/40 venture, 40/40
hiring, and 40/40 matched RQ5: 120/120 pass all frozen sampler gates**.
The venture recovery shortfall is closed. The earlier 39/40 venture result
and iteration-16 `sigma_cell` tail ESS of 310.525 were genuine before-rerun
findings, not transcription errors.

Only venture iteration 16 was rerun, using the existing
`configs/h_m01_size_assessment_anchored_venture_recovery_rerun_config.json`
with four chains, 1,000 warmup and 1,000 retained sampling draws per chain,
and maximum treedepth 12. The configured rerun used current source and rebuilt
the Stan executables. Regenerated study design and simulation configuration
were checked to match before launch; the simulation seed was 12360 and the
inference seed was 54336.

| Rerun diagnostic | Result |
|---|---:|
| Maximum R-hat | 1.00295 |
| Minimum structural bulk ESS | 1,546.3 |
| Minimum structural tail ESS | 1,960.77 |
| Minimum E-BFMI | 0.7462447584 |
| Divergences | 0 |
| Treedepth saturation | 0 |
| Maximum depth reached (configured limit 12) | 7 |
| Non-finite rejected proposals | 24 |
| Fit time (seconds) | 525.0818288 |

The run log records approximately 13 minutes 21 seconds total, including
postprocessing. The 24 non-finite rejected proposals are reported diagnostics,
not a frozen pass/fail gate.

Archive root:
`results/parameter_recovery/h_m01_size_assessment_anchored_venture_recovery_pilot_500/rerun_audit_20260911_iteration16/`.
It preserves the entire original `iteration_16/`, original `recovery_summary/`,
`config_info.json`, `study_design.json`, and `all_true_parameters.json`, along
with `run.log` and `unchanged_iterations.sha256`. All original true-parameter
values match the rerun exactly numerically; only empty `contrasts` metadata
was added. This is not a claim of byte-identical truth files or a direct
verification of simulated `y` bytes. All 117 hashes for the other 39 venture
fits' truth, diagnostics, and posterior summaries were verified unchanged.
The runner refreshed the venture aggregate; its saved
`recovery_summary/sampler_diagnostics.json` matches recomputation. No other
campaigns were rewritten.

The numerical coverage, bias, RMSE, interval-width, and campaign timing results
in the older sections below remain historical before-rerun evidence; they
have not been recomputed for this documentation update. No new SBC or power
claim is made. The rerun involved no provider calls, budget change, production
preflight/staging, or environment creation. Status remains **NOT READY**:
the later $122.78 budget approval recorded above resolves the historical
$31 ceiling shortfall under the current conditional liability assumptions,
but does not authorize production or replacement attempts.
Fresh staging and its manifest, preflight from a clean committed revision,
and explicit wave and production launch authorization remain pending.

## Reopened-gate progress (2026-09-07)

The assessment-anchored inference and simulation models compile. Subsequent
production-dimensional validation and 40-dataset recovery campaigns for
venture, hiring, and matched RQ5 completed, with the historical venture tail-ESS
shortfall documented below and its 2026-09-11 closure recorded above. Prior predictive
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
the recovery/SBC campaign decision remained open at that checkpoint and was
subsequently recorded in Amendment 4.

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
simulated parameters. The former claim that both final campaigns passed every
gate for all 40 datasets was false at the before-rerun checkpoint. Venture
iteration 16 then had `sigma_cell` tail ESS 310.525, below the required 400;
the historical corrected all-gate counts were:

| Pool | Passed | Maximum R-hat | Minimum bulk ESS | Minimum E-BFMI | Divergences | Treedepth saturation |
|---|---:|---:|---:|---:|---:|---:|
| Venture | 39/40 | 1.00925 | 425.273 | 0.6502 | 0 | 0 |
| Hiring | 40/40 | 1.00939 | 411.571 | 0.6990 | 0 | 0 |

The historical before-rerun primary size-slope recovery results were:

| Pool | Bias | RMSE | 90% coverage | Mean interval width |
|---|---:|---:|---:|---:|
| Venture | -0.00069 | 0.01359 | 0.900 | 0.04136 |
| Hiring | -0.00270 | 0.01368 | 0.925 | 0.04006 |

At nominal 90% coverage, the Monte Carlo standard error from 40 datasets is
0.047. The observed `gamma_size` coverages are therefore consistent with the
nominal target at this resolution. Total recorded fit time was 5.57 hours for
venture and 5.79 hours for hiring. At that checkpoint campaign execution was
complete, but the venture all-gate requirement was unmet. The preceding
offline documentation correction ran no fits; the separately authorized
2026-09-11 rerun above subsequently closed the shortfall. Amendment 4 declines
a separate formal SBC campaign in favor of
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
simulation/recovery gate is complete.

Across all 120 recovery fits at the before-rerun checkpoint, divergences and
treedepth saturation were zero. That did not erase the genuine venture tail-ESS
failure: historical all-gate counts were 39/40 venture, 40/40 hiring, and
40/40 matched RQ5. The dated rerun record above supersedes only the current
sampler-gate status, now 120/120, not these historical recovery estimates.

The offline reporting command reconstructs saved CmdStan chain CSVs from an
explicit fit manifest, hashes every chain artifact, and emits nine RQ1/RQ2
decisions plus one RQ6 decision per pool and six matched RQ5 decisions, for
26 primary decisions total. RQ3 includes raw cell residuals and model-by-prompt
difference-in-differences. RQ4 uses independent posterior differences and
reports all 15 model-pair rankings and same-sign probabilities across pools;
these descriptive summaries introduce no new confirmatory hypotheses. It computes linear
contrasts draw by draw, enforces R-hat, bulk ESS, tail ESS, E-BFMI, divergence,
and treedepth gates, reports the full unadjusted family, and compares the full
paired fit with both frozen presentation-only fits for sign, interval-decision,
and substantive-interpretation changes, as well as the full-data utility grid.
It checks finite structural values and diagnostics, actual four-chain metadata
and configured maximum treedepth 12, preserved rank after exclusions, posterior
consistency, and posterior predictive summaries. Synthetic artifact tests cover
this path without running Stan; they were not rerun for these documentation edits.

`scripts/build_confirmatory_report.py` takes `--fit-manifest` and `--output`
paths. The strict manifest has exactly three top-level keys: `schema_version`
(integer 1), `max_treedepth` (12), and `fits`. All 15 entries are mandatory:
`venture`, `hiring`, and `matched_rq5`, each with `primary`,
`presentation_1_only`, `presentation_2_only`, `utility_035`, and `utility_065`.
Each entry has exactly `chain_path` (an absolute directory), `chain_sha256`
(all four chain basenames mapped to their SHA-256 digests), and `stan_data`,
`preparation_report`, and `analysis_contract` bindings, each exactly
`{"path": "absolute file path", "sha256": "digest"}`. Legacy directory-only
entries are rejected. Hash binding declares the associated fit inputs; it is
not proof that CmdStan actually executed with those inputs.

The runner writes the following files under each pool directory and under
`matched_rq5/`. Every variant's assembly report retains `cell_ids`,
`design_columns`, `rank`, and `presentation_id`, preserving the whole-cell
exclusions and design alignment used for that payload.

| Variant | Stan data | Preparation report |
|---|---|---|
| `primary` | `stan_data_size.json` | `stan_data_size_assembly_report.json` |
| `presentation_1_only` | `stan_data_size_presentation_1.json` | `stan_data_size_presentation_1_assembly_report.json` |
| `presentation_2_only` | `stan_data_size_presentation_2.json` | `stan_data_size_presentation_2_assembly_report.json` |
| `utility_035` | `stan_data_size_u035.json` | `stan_data_size_u035_assembly_report.json` |
| `utility_065` | `stan_data_size_u065.json` | `stan_data_size_u065_assembly_report.json` |

Each group also has `analysis_contract.json`. The matched group's historical
`assembly_report.json` primary alias does not replace the per-variant reports.
The 15 mandatory fits supersede the six-fit schedule; the preregistered
position-stable subset remains a separate robustness analysis.

Template only, not a runnable manifest or evidence of completed fits: replace
every `/ABSOLUTE/PLACEHOLDER` root, each example `chain-N.csv` basename, and
every `<SHA256-...>` with the actual absolute paths and independently recorded
64-hex-character digests. Each chain directory must contain the four declared
chain files, and every binding must refer to that variant's actual artifact.

```json
{
	"schema_version": 1,
	"max_treedepth": 12,
	"fits": {
		"venture": {
			"primary": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/venture/primary",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/venture/stan_data_size.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/venture/stan_data_size_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/venture/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			},
			"presentation_1_only": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/venture/presentation_1_only",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/venture/stan_data_size_presentation_1.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/venture/stan_data_size_presentation_1_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/venture/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			},
			"presentation_2_only": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/venture/presentation_2_only",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/venture/stan_data_size_presentation_2.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/venture/stan_data_size_presentation_2_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/venture/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			},
			"utility_035": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/venture/utility_035",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/venture/stan_data_size_u035.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/venture/stan_data_size_u035_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/venture/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			},
			"utility_065": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/venture/utility_065",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/venture/stan_data_size_u065.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/venture/stan_data_size_u065_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/venture/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			}
		},
		"hiring": {
			"primary": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/hiring/primary",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/stan_data_size.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/stan_data_size_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			},
			"presentation_1_only": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/hiring/presentation_1_only",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/stan_data_size_presentation_1.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/stan_data_size_presentation_1_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			},
			"presentation_2_only": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/hiring/presentation_2_only",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/stan_data_size_presentation_2.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/stan_data_size_presentation_2_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			},
			"utility_035": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/hiring/utility_035",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/stan_data_size_u035.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/stan_data_size_u035_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			},
			"utility_065": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/hiring/utility_065",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/stan_data_size_u065.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/stan_data_size_u065_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/hiring/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			}
		},
		"matched_rq5": {
			"primary": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/matched_rq5/primary",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/stan_data_size.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/stan_data_size_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			},
			"presentation_1_only": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/matched_rq5/presentation_1_only",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/stan_data_size_presentation_1.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/stan_data_size_presentation_1_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			},
			"presentation_2_only": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/matched_rq5/presentation_2_only",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/stan_data_size_presentation_2.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/stan_data_size_presentation_2_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			},
			"utility_035": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/matched_rq5/utility_035",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/stan_data_size_u035.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/stan_data_size_u035_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			},
			"utility_065": {
				"chain_path": "/ABSOLUTE/PLACEHOLDER/fits/matched_rq5/utility_065",
				"chain_sha256": {"chain-1.csv": "<SHA256-CHAIN-1>", "chain-2.csv": "<SHA256-CHAIN-2>", "chain-3.csv": "<SHA256-CHAIN-3>", "chain-4.csv": "<SHA256-CHAIN-4>"},
				"stan_data": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/stan_data_size_u065.json", "sha256": "<SHA256-DATA>"},
				"preparation_report": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/stan_data_size_u065_assembly_report.json", "sha256": "<SHA256-REPORT>"},
				"analysis_contract": {"path": "/ABSOLUTE/PLACEHOLDER/matched_rq5/analysis_contract.json", "sha256": "<SHA256-CONTRACT>"}
			}
		}
	}
}
```

At the recorded checkpoint, all maintained application tests passed (536 tests).
These counts are historical, not new validation of this documentation change.
Repository-wide pytest last
passed 475 tests but reported three unrelated collection errors from the legacy
executable `scripts/test_m1_model.py`, whose helper functions are named
`test_*` but require command-line arguments rather than pytest fixtures.

The detailed status of every independent-review finding is tracked in
`REVIEW_DISPOSITION.md`. The venture recovery shortfall was closed by the
authorized 2026-09-11 iteration-16 rerun, giving 120/120 all-gate passes;
the offline gates and rendered full-campaign liability were verified on
2026-09-11 as recorded above. Budget approval was obtained on 2026-09-11.
Fresh production staging and its manifest, and separately authorized preflight
against a named, authorized production wave from a clean committed revision remain
pending. The live reasoning-arm probes completed successfully on
2026-09-11. No production wave has been authorized or launched.

## Batch and usage-persistence gate

The approved production mode is provider Batch collection with a $122.78 ceiling,
approved 2026-09-11 in place of the historical $31 ceiling.
The frozen configuration now selects `collection_mode: batch`; other configs
retain the synchronous default. Choice collection supports both OpenAI JSONL
file batches and Anthropic Message Batches. Both SDK clients now use
`max_retries=0`, so SDK retries cannot silently repeat creation calls. Each
cell has an atomic state file, fsynced before replacement with its parent
directory fsynced afterward, containing its provider batch ID and a digest of
the full request set. A first
invocation submits once and returns `batch_pending`; later invocations retrieve
that same batch. Prompt or model drift after submission is rejected rather than
silently attaching old paid results to a changed design.

OpenAI submissions also carry a durable random submission-intent ID and the
full request digest in provider metadata. If local persistence fails after
provider acceptance, a later invocation lists provider batches and attaches
only when exactly one batch matches the cell, request digest, and intent ID;
zero or multiple matches remain fail-closed. Older batches with identical
request content cannot satisfy the intent match. Batch-state writers use a
nonblocking OS advisory lock, which is released automatically if its owner
process exits; stale lock files are therefore safely reusable while a live
owner still blocks concurrent mutation. Anthropic ambiguous submissions remain
manual reconciliation cases because Message Batches expose no equivalent
request-bound listing metadata.

Manual Anthropic reconciliation uses `scripts/attach_anthropic_batch.py` only
after an operator identifies the unique candidate in the provider console. The
command accepts only an existing `submission_ambiguous` Anthropic state,
retrieves rather than creates the supplied provider Batch, checks its request
count from either a total or the complete provider status counts, and records
the operator note, remote status, timestamp, and durable attachment before
normal retrieval resumes. Missing count evidence, a count mismatch,
non-ambiguous state, or provider/model mismatch remains blocked. Replacement
submission is never part of this procedure.

The approved choice ceiling is now `$122.78`. The historical `$31 / 10,080`
per-request reservation floor remains `0.0030753968253968253`, unchanged.
Conservative liability is calculated from
rendered UTF-8 message-content bytes plus 1,024 per message and 1,024 request
overhead, with exact provider output caps (including reasoning where applicable),
pinned input/output pricing, and the 0.5 Batch multiplier. This is a conditional
bound under the stated pricing and protocol assumptions, not an absolute
provider billing guarantee. The 2026-09-11 exact-render audit above establishes
a $122.7729167015873 full-campaign reservation, within the approved $122.78
ceiling under those assumptions. That reservation and the $61.867008 maximum
output-only amount exceeded the historical $31 ceiling; neither amount is a
forecast of actual spend or authorization for retries.
Before the provider call begins, each
new submission intent is fsynced to a separately locked append-only reservation
ledger. Duplicate reservation of the same intent is idempotent; a conflicting
intent, an over-ceiling request, or malformed/truncated ledger input stops before
submission. Reservations remain charged after failed, ambiguous, and completed
attempts. They do not authorize retries or any production wave.

For OpenAI terminal statuses, the retriever reads both `output_file_id` and
`error_file_id`, combines their per-request records, captures batch-level
errors, and prefers provider aggregate Batch usage and request counts when
available. Responses, failed and duplicate IDs, raw records, provider errors,
and usage are written to durable batch state before an integrity exception.
Successful, non-duplicate rows from a partially failed cell are then resolved
through the normal choice parser and written to a request-bound checkpoint
before the exception is re-raised. No replacement submission is automatic.

Future Anthropic retrievals now preserve raw per-request results, including
message content, errors, and usage, before integrity failures. The historical
live Sonnet probe omitted raw message metadata; that evidence cannot be
recovered offline, and the new persistence path does not retroactively supply it.
The reasoning probe's completed-cache path now validates exact request identity
before reuse. These remediations do not establish absence of historical remote
duplicates.

After every Batch poll or exception, `batch_budget_report.json` joins each
reservation to its durable provider state. It reports total reserved dollars,
known usage-estimated cost, remaining reservation headroom, per-attempt status,
and the submission IDs whose usage cost is not yet known. Missing usage remains
explicitly unresolved and is never counted as zero or treated as invoice data.

Reservation additionally requires a nonempty `batch_wave_id` and explicit
membership of the cell in `batch_wave_cell_ids`. The preregistered production
YAML pins these to `null` and an empty list, respectively, so the repository is
no-spend by default. A separately reviewed config change must name and allowlist
each production wave before any provider submission can pass the reservation
boundary.

The offline `preflight` command implements the F7 production-root boundary. It
refreshes all configured pool gates, requires passing evidence for every pool
represented in the authorized wave, and locally renders exact provider-ready
request bodies without creating an SDK client or contacting a provider. It
copies the required pool, problem, embedding, PCA, assessment, and fresh gate
artifacts into a single-use `production_stages/<wave_id>/` directory, writes an
archive of exact provider-ready bodies and their custom-ID/problem/presentation/
item-order mapping, and an aggregate-hashed manifest. The manifest binds a clean
Git commit, application and active Stan/config sources, toolchain versions,
exact configuration, prompts, authorization, per-cell request hashes, and
request counts. Staged files and directories are then made read-only.

Immediately before a new budget reservation, the runner verifies the manifest
aggregate, current configuration, prompt hashes, mutable source hashes,
immutable staged-copy hashes, cell authorization, and the in-memory request
hash and count. It also rechecks the clean Git identity, repository source
hashes, and generated archives. Any absent or changed evidence fails before ledger append or provider
submission. The mechanism is covered by offline tests; it has not yet been run
against an authorized production wave, because the checked-in production YAML
remains deliberately no-spend.

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
unique OpenAI provider reconciliation, post-acceptance state-write failure,
advisory-lock contention and stale-file recovery, OpenAI request JSONL,
Anthropic extended-thinking request parameters, out-of-order results,
incomplete output, terminal error-file persistence, aggregate failed-Batch
usage, partial-success checkpoint salvage, hard-ceiling reservation,
truncated-ledger rejection, collector presentation mapping, and preservation
of the synchronous path. Preflight tests additionally cover successful
read-only staging, exact request binding, source tamper rejection, failed fresh
gates, and path-unsafe wave IDs.

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

### Reasoning-arm Batch probe

On 2026-09-10, the authorized reasoning probe submitted both frozen
presentations of size-8 venture menu `VEN0093` to each reasoning arm. The
OpenAI Batch `batch_6aa3035508c08190881be0fa2fe8144f` and Anthropic Batch
`msgbatch_01AZUf5XFHiqRmii5eberLpP` were retrieved by their persisted IDs;
no replacement Batch was submitted. Both providers returned two successful
records under the expected custom IDs with visible `ANSWER:` tokens, no failed
or duplicate IDs, and no provider errors. Within each arm, both presentations
mapped to the same underlying item (`V012` for o3-mini and `V018` for thinking
Sonnet), confirming that the production parser and frozen presentation mapping
worked on the long-menu request shape.

| Provider | Endpoint | Calls | Input tokens | Output tokens | Reasoning tokens | Recorded Batch cost |
|---|---|---:|---:|---:|---:|---:|
| OpenAI | `o3-mini-2025-01-31` | 2 | 1,830 | 1,394 | 1,152 | $0.00407330 |
| Anthropic | `claude-sonnet-4-5-20250929` | 2 | 2,076 | 2,079 | 0 reported separately | $0.01870650 |
| **Total** | | **4** | **3,906** | **3,473** | **1,152** | **$0.02277980** |

The exact provider-body hashes were
`771163588bc63d03d4d00b73f02f2a9d5bc92359d65e2ef1104dfaf103e50861`
for OpenAI and
`e4ea053cd540457b412d2b901453fe0051277465eacb16cb97c4a86b3910ea74`
for Anthropic. An fsynced reservation of $0.50 per arm preceded submission,
and the idempotent append-only usage ledger contains one completed event per
arm. The measured total was below the authorized $1 ceiling. This remains a
genuine successful four-call probe and closes the live-probe component of F12,
but successful local records and a single usage event do not prove absence of
remote duplicates. Historical Sonnet raw message metadata was not retained and
is irrecoverable offline; newly implemented raw-result persistence applies to
future retrievals, not this historical evidence. The Sonnet reasoning arm uses
thinking budget 4,096, total output cap 4,160, and temperature 1 versus the
thinking-off baseline at temperature 0. The contrast must be reported as that
treatment bundle, not as a pure causal effect of thinking.

## Verdict

**SUPERSEDED: the 2026-09-06 technical gate returned GO.** At that time, the
frozen gates, offline call count, application tests, and limited live
provider-specific Batch probe passed. Subsequent review showed that those
checks did not establish scientific identification or cover critical Batch
failure modes.

**Current verdict: NOT READY.** Production choice collection is stopped. The
reasoning-arm probe requirement and recovery sampler gates are complete:
the authorized venture iteration-16 rerun brings the latter to 120/120.
Offline gates now pass and exact rendered
liability is known. The user's 2026-09-11 approval of the $122.78 ceiling
resolves the budget blocker for the frozen 10,080-request campaign under the
current conditional liability assumptions, without changing design or token
limits. It is not a provider billing guarantee or replacement authorization.
Saved historical gate reports remain stale. Fresh production staging and its
manifest remain pending. The worktree is dirty, and the user has not authorized
a Git commit. A new GO still requires an
exact-root preflight executed from a clean committed revision against a named,
authorized production wave. That operation requires separate authorization.
Passing it will still not authorize the 10,080-call production run; that
launch requires a new explicit authorization beyond budget approval.

The synchronous path is technically validated, but using it would invoke the
$61 fallback and also requires separate approval. No production calls have
been launched. The preceding offline audit ran no fits; the separately
authorized recovery rerun is recorded above. This documentation-only update
ran no fits or preflight and made no provider calls. It does not authorize a
production wave; it records the already updated $122.78 configuration ceiling.