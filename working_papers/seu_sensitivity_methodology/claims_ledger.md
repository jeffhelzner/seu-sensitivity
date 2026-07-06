# Claims Ledger

Every empirical/quantitative claim that appears in the paper must trace to a
source report and carry a status. This operationalizes the plan's
honest-reporting commitments and makes claims auditable before the first full
draft. **No claim is promoted from a placeholder to body text until its status
is `computed`.**

Status values:

- `computed` — number is finalized from a completed run; safe to cite.
- `placeholder` — marked `[from report]` in the plan; must be transcribed.
- `re-run-pending` — number exists but the gating re-run will replace it.

| ID | §  | Claim | Source report | Reported as | Status |
| -- | -- | ----- | ------------- | ----------- | ------ |
| C1 | 6.4.1 | m_1 does **not** materially reduce α RMSE at matched choice count (B→C): −8.1%, 90% CI [−22.0, +6.8] (null/slightly worse) — **the pilot's ≈15% gain did not replicate** | `report14_rerun_results.json` (n=100) | bootstrap-over-iterations 90% CI | computed |
| C2 | 6.4.1 | Adding more *uncertain* data alone (A→B) **improves** α RMSE 27.1%, 90% CI [+15.0, +38.5] — **opposite sign to the pilot's claimed slight worsening** | `report14_rerun_results.json` (n=100) | bootstrap-over-iterations 90% CI | computed |
| C3 | 6.4.2 | δ RMSE B→C ≈ 0%, 90% CI [−4.0, +4.5] (null); δ CI-width paired-median reduction only ≈0.8%, 90% CI [0.6, 1.2], narrower in 72% of iters, Wilcoxon p≈1.2e-6 (significant but ≈¼ the pilot's ≈2%) | `report14_rerun_results.json` (n=100) | paired-iteration median + bootstrap 90% CI + Wilcoxon | computed |
| C4 | 3.4 / B.2 | Uncertain choices weakly inform (β, δ) in m_0: recovery returns wide marginal CIs for β and δ (vs tight α CIs) and correlated β–δ posterior-mean errors across replicates (illustrative, non-gauge-invariant; sign regime-dependent, not reported) — the posterior concentrates on a β–δ trade-off, not on either separately ⇒ α separates empirically | `04_parameter_recovery.qmd` (Report 4) | recovery CI-widths + β–δ error correlation | computed |
| C5 | 4.4.3 / 6.5.1 | Marginal SBC ranks uniform for α, β, δ in both models at N_sbc = 999 | `06_sbc_validation.qmd` | rank histograms + ECDF/KS band | computed |
| C6 | 4.3.2 | α recovery: low bias, calibrated 90% intervals | `04_parameter_recovery.qmd` | true-vs-estimated scatter + CIs | computed |
| C7 | 4.3.3 | β/δ recovery: wider CIs, correlated β–δ posterior-mean errors across replicates (illustrative, non-gauge-invariant; sign regime-dependent) | `04_parameter_recovery.qmd` | recovery summaries | computed |
| C8 | 7.3.2 | Insurance α prior Lognormal(3.0, 0.75): median ≈ 20, 90% ≈ [5.5, 67], SEU-max rate ≈ 78% | `temperature_study/01_initial_study.qmd` | grid-search summary | computed |
| C8b | 7.3 | Ellsberg (m_02, K=4) α prior recalibrated to Lognormal(3.5, 0.75): median ≈ 33, 90% ≈ [10, 124], SEU-max rate ≈ 76% — recalibration reflects K=4 changing EU-spread/softmax curvature, not a baseline shift (menu size N_m ∈ {2,3,4} identical across tasks) | `ellsberg_study/01_ellsberg_study.qmd` | grid-search summary | computed |
| C9 | 7.5.1 | GPT-4o × insurance: slope median ≈ −24.6, 90% CI ≈ [−52.4, −6.7], P(slope<0) ≈ 0.99; P(strict monotone) ≈ 0.12 ⚠ corrected from the superseded −31 (ddof-inflated ×1.25; see E.2 correction note) | `spikes/report_2x2_forest_spike.py` → `report_2x2_forest_results.json` (committed α draws; canonical draw-level population-OLS) | template (§7.1b) | computed |
| C10 | 7.5.2 | Claude × insurance: medians ≈ {71,53,73,71,55}; slope median ≈ −2.9, 90% CI ≈ [−42.9, 30.8], P(slope<0) ≈ 0.56 ⚠ corrected from the superseded −3.6 (ddof-inflated ×1.25; see E.2) | `spikes/report_2x2_forest_spike.py` → `report_2x2_forest_results.json` | template (§7.1b) | computed |
| C11 | 7.5.2a | Cross-LLM insurance: P(GPT-4o slope < Claude slope) = **0.817 full grid**, **0.816 restricted** to Claude's T ≤ 1.0 (GPT-4o re-summarized, T=1.5 dropped) — the directional LLM contrast is robust to the unequal-grid confound; GPT-4o slope median −24.6 (P<0 0.991, matches C9; restricted −29.9, P<0 0.943), Claude −2.9 (P<0 0.560). The between-LLM probability answers a different question from either within-cell P (D.6(4)) | `spikes/report11_cross_llm_spike.py` → `report11_cross_llm_results.json` (seed 20260618; canonical population-OLS estimator); cross-checked vs `factorial_synthesis/01_factorial_synthesis.qmd` (~0.80–0.82) | both numbers + qualitative pattern | computed |
| C12 | 7.5.3 | GPT-4o × Ellsberg: α medians {110.4, 106.9, 99.5, 84.0, 52.2} at T={0.0,0.3,0.7,1.0,1.5}, 90% CIs [74.4,167.2]/[72.6,163.8]/[65.4,154.6]/[57.1,126.1]/[35.5,80.3]; slope median −38.4, 90% CI [−72.1,−10.0], P(slope<0) 0.984; P(strict mono ↓) 0.090 | `gpt4o_ellsberg_study/data/primary_analysis.json` | template (§7.1b) | computed |
| C13 | 7.5.4 | Claude × Ellsberg: α medians {85.4, 56.1, 82.3, 53.0, 66.4} at T={0.0,0.2,0.5,0.8,1.0}, 90% CIs [58.9,127.2]/[38.2,84.1]/[52.3,135.3]/[36.0,80.5]/[45.3,99.8]; slope median −15.0, 90% CI [−52.2, 19.6], P(slope<0) 0.766; P(strict mono ↓) 0.0085 (no monotone pattern — oscillating medians) ⚠ slope corrected from the superseded −18.8 (ddof-inflated ×1.25; see E.2) | `spikes/report_2x2_forest_spike.py` → `report_2x2_forest_results.json`; medians/CIs from `ellsberg_study/data/primary_analysis.json` | template (§7.1b) | computed |
| C14 | 7.4.3 | MCMC diagnostics, 20 fits: **α never R-hat-flagged; ESS satisfactory in all 20 fits**; divergences ≤ 0.15%/fit (≤6/4000, total 34 across **12/20** fits — not just the 2 highest GPT-4o T); R-hat > 1.01 in **4/20** fits, confined to weakly-identified β/δ + per-trial nuisance latents (eta/psi/upsilon) in the harder K=4 GPT-4o×Ellsberg fits (+ one Claude×Insurance, eta only) ⚠ plan §7.4.3 wording corrected | `spikes/report1415_diagnostics_ppc_spike.py` → `report1415_diagnostics_ppc_results.json`; `reports/applications/*/data/diagnostics_T*.txt` | R-hat / ESS / divergences | computed |
| C15 | 7.4.4 | PPC p-values all in [0.3, 0.7] across all 20 fits: 60/60 in band, range [0.317, 0.657], mean 0.462 (ll/modal/prob × 20 fits) | `spikes/report1415_diagnostics_ppc_spike.py` → `report1415_diagnostics_ppc_results.json`; `reports/applications/*/data/fit_summary.json` | 3 PPC summaries | computed |
| C16 | 7.5.2 / 7.6.1 | Min-detectable-effect for the Claude null: design resolves a slope only if \|Δα/ΔT\| ≳ 36 α-units/unit T (P(slope<0) ≥ 0.95); observed \|slope\| ≈ 2.9 is ≈ 1/12 of this floor, so the null is "no effect at the achievable resolution," not a positive null | `spikes/report16_mde_spike.py` → `spikes/report16_mde_results.json`, `figures/report16_mde_power_curve.png` (seed 20260618) | power curve P(slope<0) vs \|β\|, MDE crossing | computed |
| C17 | 7.6.6 | Prior sensitivity of the 4 per-cell draw-level population-OLS temperature-slopes: refitting each cell under 3 alt α priors (loc ±0.5, scale 1.25) holding data/features fixed leaves every qualitative reading unchanged. GPT-4o×Insurance slope median ∈ [−30.7,−22.6], P(slope<0) ∈ [0.987,0.991]; GPT-4o×Ellsberg [−44.8,−36.8], [0.984,0.994]; Claude×Insurance [−4.2,−2.7], [0.55,0.58]; Claude×Ellsberg [−17.0,−14.0], [0.75,0.77]. Sign stable in all 4 cells | `spikes/report_prior_sensitivity_spike.py` → `spikes/report_prior_sensitivity_results.json`, `figures/report_prior_sensitivity_forest.png` (base_seed 20260701); refits `reports/applications/*/data/prior_sweep/` via `models/m_0_prior_sweep.stan` | forest of slope posteriors × 4 priors × 4 cells | computed |
| C18 | 5.5 / 5.6 / 7.2 / 7.4 | Design-conditional information diagnostics. (i) Foundational lottery-spanning check: contrast rank 2 = K−1, nonzero singular values 1.08/0.67 (Prop 5.2 hypothesis verified). (ii) β contrast-Jacobian rank: foundational rank 10 = (K−1)D at all 20 prior draws → β locally identified mod gauge; applications rank 29 < 64 (K=3) / 96 (K=4) dof → β not identified. (iii) Application feature matrices full row rank 30 (D=32 ≥ R=30). (iv) Prior-predictive within-menu η-gap medians: foundational ≈ 0.52; insurance ≈ 0.18–0.20; Ellsberg ≈ 0.10–0.12. (v) α contraction: posterior sd(log α) 0.23–0.29 vs prior 0.75 across all 20 fits | `spikes/report_design_diagnostics_spike.py` → `report_design_diagnostics_results.json` (seed 20260705) | ranks, singular values, η-gap and contraction summaries | computed |
| C19 | 7.4 / D.2 | Stricter-sampler refit sensitivity: all 20 application conditions refit at adapt_delta 0.99, max_treedepth 12, warmup 2000 — **0 divergences and 0 treedepth hits across all 20 fits** (baseline: 34 divergences across 12/20 fits); max α-posterior-median shift 0.06 committed-posterior sd; slopes unchanged within MC noise (GPT-4o×Ins −24.6→−24.8, P 0.991→0.986; Claude×Ins −2.9→−2.0, 0.560→0.537; GPT-4o×Ells −38.4→−37.5, 0.984→0.992; Claude×Ells −15.0→−14.7, 0.766→0.767) | `spikes/report_refit_sensitivity_spike.py` → `report_refit_sensitivity_results.json` (seed 42) | per-fit divergences + α posterior deltas | computed |

## Gating rule

The first full-paper draft is blocked until: every `re-run-pending` row is
resolved (Report-14 re-run at n ≥ 100), and every `placeholder` row
is transcribed to `computed` from its source report.

## Report-14 re-run outcome (gate 1 — resolved, ⚠ pilot did not replicate)

The n = 100 re-run (rows C1–C3; `spikes/report14_rerun_analysis.py` →
`spikes/report14_rerun_results.json`, design seed 20260617, 10 000 bootstrap
resamples) **materially contradicts the 30-iteration pilot** that the plan's
§6.4 narrative was built on:

- **C1 (central B vs C test).** Pilot: m_1's β-free risky block reduces α RMSE
  ≈ 15%. n = 100: **−8.1%, 90% CI [−22.0, +6.8]** — the effect is null and, at
  the point estimate, slightly *adverse*. **The headline α gain does not
  replicate.**
- **C2 (A→B data-quantity control).** Pilot: doubling uncertain data alone
  slightly *worsens* α RMSE. n = 100: it **improves** α RMSE **+27.1%, 90% CI
  [+15.0, +38.5]** — opposite sign, strongly bounded away from zero.
- **C3 (δ B vs C).** Pilot: ≈ 2% δ tightening, Wilcoxon significant. n = 100: δ
  RMSE effect null (0.3%, CI straddles 0); δ CI-width narrows only ≈ 0.8% (CI
  [0.6, 1.2], 72% of iters, Wilcoxon p ≈ 1.2e-6) — direction holds and is
  statistically significant, but the magnitude is ≈ ¼ of the pilot's and
  practically negligible.

**Consequence for the draft (author decision needed).** The §6.4.4 two-row
contribution table ("α ≈ 15% material / δ ≈ 2% negligible") is no longer
supported: at n = 100 the α B→C payoff is null and the A→B control flips sign.
The core δ lesson (*identifiable ≠ precisely estimable at realistic n*) is
**strengthened** (δ payoff is even smaller than claimed), but the companion α
"sharpening of an already-identified parameter" story needs to be **rewritten or
dropped**. The §6.4 framing should not be drafted from the pilot numbers; flag
for reframing before §6 is written.

C16 (minimum-detectable-effect / power statement for the Claude null) is
resolved separately below.

## C16 outcome (Claude-null MDE — resolved)

Computed directly from the saved per-condition posterior α draws
(`reports/applications/claude_insurance_study/data/alpha_draws_T*.npz`, 4 000
draws/condition) using the application's own draw-wise population-OLS slope
functional `b_i = Cov(T, α_i)/Var(T)`
(`spikes/report16_mde_spike.py` → `report16_mde_results.json`, seed 20260618).
The recomputed slope posterior reproduces `primary_analysis.json` exactly
(median −2.89, sd 22.09, P(slope<0) 0.560 — self-check passed).

- **MDE ≈ 36 α-units per unit temperature** at criterion P(slope<0) ≥ 0.95.
  Three independent estimators agree: analytic Gaussian 36.3, empirical-quantile
  34.7, constant-CV Monte Carlo 36.0 (headline). At the stricter P ≥ 0.975 the
  MDE is ≈ 41–43.
- Context: that is **≈ 12× the observed |slope|** (≈ 2.9), **≈ 0.53 of the
  grand-mean α** (≈ 67.5), i.e. an end-to-end α change of ≈ 36 over T ∈ [0, 1]
  (roughly halving α). The per-condition CV is near-constant (≈ 0.28), so the
  resolution floor is set by posterior width, not by a level artefact.
- The GPT-4o full-grid |slope| ≈ 25 (C9, reference only — unequal-grid caveat,
  §7.5.2a) is **0.68× the Claude MDE**: even a GPT-4o-sized effect sits
  below what the Claude design could have resolved at the 0.95 criterion.
- **Reading.** The Claude null is *inconclusive at the achievable resolution*,
  not a positive claim of no temperature effect; this licenses the §7.6.1(iii)
  "declines to support" framing and blocks any "no effect" over-reading.
- **Correction note (slope estimator; formerly "slope convention").** Earlier
  ledger revisions treated the −3.6 "report-level" and −2.9 "draw-level"
  insurance slopes as two legitimate conventions. That was wrong: the −3.6 (and
  the −31 / −18.8 headline slopes) came from a per-draw estimator mixing
  `np.cov` (ddof=1) with `np.var` (ddof=0), inflating every slope by exactly
  5/4 = 1.25 on the five-point grids. The draw-level population-OLS values
  (−2.9, −24.6, −15.0) are the single canonical convention paper-wide;
  P(slope<0) is scale-invariant and was never affected. GPT-4o × Ellsberg
  (−38.4) always used the correct formula. See paper Appendix E.2 and
  `spikes/report_2x2_forest_spike.py` (records superseded values + factor).

## C14 outcome (MCMC diagnostics — resolved, ⚠ plan wording corrected)

Aggregating the committed CmdStan `diagnose` output across all 20 factorial
fits (`spikes/report1415_diagnostics_ppc_spike.py` →
`report1415_diagnostics_ppc_results.json`) **refines the plan's §7.4.3
wording**, which claimed diagnostics were "clean across all 20 fits, with at
most 1–2 divergent transitions in the two highest GPT-4o temperature
conditions." The actual picture:

- **The load-bearing claim holds.** α — the parameter the entire §7 temperature
  analysis rests on — is **never** R-hat-flagged, and **ESS is satisfactory in
  every one of the 20 fits**. The per-condition α posteriors are a credible
  basis for the cross-condition comparison.
- **Divergences are negligible but not localized.** ≤ 0.15% per fit (≤ 6/4000;
  34 total), spread across **12/20** fits in both tasks and both providers —
  not confined to the two highest GPT-4o temperatures. Still far below any
  threshold of concern.
- **R-hat > 1.01 occurs in 4/20 fits**, confined to the **weakly-identified
  β/δ** and **high-dimensional per-trial nuisance latents** (`eta`, `psi`,
  `upsilon`) in the harder K = 4 GPT-4o × Ellsberg fits (T = 0.0, 0.7, 1.0) plus
  one Claude × Insurance fit (`eta` only). This is **consistent with — and mild
  corroboration of — the §3.4 / §6.4 weakly-informed β,δ thesis**: the
  poorly-mixing parameters are exactly the ones recovery flags as
  weakly informed. It is not a defect in the α inference.

**Consequence for the draft.** Rewrite §7.4.3 to claim what is true and
load-bearing: *α R-hat ≤ 1.01 and ESS satisfactory in all 20 fits; divergences
≤ 0.15% per fit; the only R-hat exceedances are the weakly-identified β/δ and
per-trial nuisance parameters in the hardest fits* — and fold the last point
into the identifiability narrative rather than presenting it as a blemish.
