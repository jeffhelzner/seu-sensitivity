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
| C1 | 6.4.1 | m_1 reduces α RMSE ≈ 15% at matched choice count (B vs C) | `14_does_m1_identify_delta.qmd` | paired-iteration median + bootstrap 90% CI | re-run-pending |
| C2 | 6.4.1 | Adding more *uncertain* data alone (A→B) slightly worsens α RMSE | `14_does_m1_identify_delta.qmd` | paired-iteration median + bootstrap 90% CI | re-run-pending |
| C3 | 6.4.2 | m_1 reduces δ CI width / δ RMSE ≈ 2% (B vs C); Wilcoxon significant | `14_does_m1_identify_delta.qmd` | paired-iteration median + bootstrap 90% CI | re-run-pending |
| C4 | 3.4 / B.2 | (β, δ) Fisher block has ≥1 near-zero eigenvalue at design n; near-flat direction ≈ near-kernel of η-Jacobian | B.2 spike (to run) | eigenvalues at ≥3 representative draws | re-run-pending |
| C5 | 4.4.3 / 6.5.1 | Marginal SBC ranks uniform for α, β, δ in both models at N_sbc = 999 | `06_sbc_validation.qmd` | rank histograms + ECDF/KS band | computed |
| C6 | 4.3.2 | α recovery: low bias, calibrated 90% intervals | `04_parameter_recovery.qmd` | true-vs-estimated scatter + CIs | computed |
| C7 | 4.3.3 | β/δ recovery: wider CIs, negative β–δ error correlation | `04_parameter_recovery.qmd` | recovery summaries | computed |
| C8 | 7.3.2 | Insurance α prior Lognormal(3.0, 0.75): median ≈ 20, 90% ≈ [5.5, 67], SEU-max rate ≈ 78% | `temperature_study/01_initial_study.qmd` | grid-search summary | computed |
| C9 | 7.5.1 | GPT-4o × insurance: slope median ≈ −31, 90% CI ≈ [−66, −8], P(slope<0) ≈ 0.99; P(strict monotone) ≈ 0.12 | `temperature_study/01_initial_study.qmd` | template (§7.1b) | computed |
| C10 | 7.5.2 | Claude × insurance: medians ≈ {74,55,77,74,57}; slope median ≈ −3.6, 90% CI ≈ [−54,39], P(slope<0) ≈ 0.56 | `claude_insurance_study/01_claude_insurance_study.qmd` | template (§7.1b) | computed |
| C11 | 7.5.2a | Cross-LLM: P(GPT-4o slope < Claude slope) ≈ 0.82 (full grid) + Claude-grid-restricted comparison | `factorial_synthesis/01_factorial_synthesis.qmd` | both numbers + qualitative pattern | placeholder |
| C12 | 7.5.3 | GPT-4o × Ellsberg: per-condition α medians + CIs; slope + P(slope<0); P(strict monotone) | `gpt4o_ellsberg_study/01_gpt4o_ellsberg_study.qmd` | template (§7.1b) | placeholder |
| C13 | 7.5.4 | Claude × Ellsberg: per-condition α medians + CIs; slope + P(slope<0); P(strict monotone) | `ellsberg_study/01_ellsberg_study.qmd` | template (§7.1b) | placeholder |
| C14 | 7.4.3 | MCMC diagnostics clean across all 20 fits (≤1–2 divergences in 2 highest GPT-4o T) | application reports | R-hat / ESS / divergences | placeholder |
| C15 | 7.4.4 | PPC p-values in [0.3, 0.7] across all 20 fits | application reports | 3 PPC summaries | placeholder |
| C16 | 6.4.2 | Min-detectable-effect / power statement for the Claude null (R6) | re-run / power calc (to add) | MDE at design n | re-run-pending |

## Gating rule

The first full-paper draft is blocked until: every `re-run-pending` row is
resolved (B.2 spike + Report-14 re-run at n ≥ 100), and every `placeholder` row
is transcribed to `computed` from its source report.
