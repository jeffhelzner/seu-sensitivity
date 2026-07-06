# Per-Figure Manifest

Provenance for every figure and table in the paper, so each headline artifact is
reproducible from a pinned commit. Build this **as figures are produced**, not as
a post-draft cleanup — it requires touching every figure in §§4, 6, 7 to record
provenance and is easy to defer and never finish.

This manifest is a **precondition for the first full-paper draft** (plan
Appendix E.2).

| Fig/Tab | § | Description | Script | Config | Seed | Status |
| ------- | - | ----------- | ------ | ------ | ---- | ------ |
| Disp 1.8 | 1.8 | Section dependency chain (LaTeX `\underbrace` display, authored in §1.8 body — no mermaid) | — (authored in-body, not generated) | — | — | authored |
| Fig (§4.3) | 4.3.2 | α true-vs-estimated recovery scatter + per-replicate 90% CIs (m_0) — **embedded as Figure 1** (`figures/fig_m0_alpha_recovery.pdf`) | `analysis/parameter_recovery.py` (`reports/foundations/04_parameter_recovery.qmd`) | inline (m_0: M25 K3 D5 R15; 50 iters, 4 chains × 1000) | design np 42; 12345+i sim / 54321+i fit (D.2) | computed |
| Fig 6.4 | 6.4.1 | α/δ RMSE + CI-width: matched-design A/B/C/D (n=100) — **embedded as Figure 2** (`figures/report14_rerun_matched.png`) | `spikes/report14_rerun_analysis.py` | `configs/m1_matched_recovery_n100_config.json` | design 20260617 / boot 20260617 | computed |
| Fig (§6.5) | 6.5 | δ ECDF comparison (m_0 and m_1) with 95% KS band — marginal-SBC demarcation — **embedded as Figure 3** (`figures/fig_sbc_delta_ecdf.pdf`) | `analysis/sbc.py` (`reports/foundations/06_sbc_validation.qmd`) | inline (m_0 and m_1 canonical SBC designs; N_sbc=999 thin4 1 chain) | design np 42; 123+i per draw (D.3) | computed |
| Fig 7.5.2 | 7.5.2 / 7.6.1 | Claude-null MDE / power curve: P(slope<0) vs \|Δα/ΔT\|, MDE ≈ 36 marker — **embedded as Figure 4** (`figures/report16_mde_power_curve.png`) | `spikes/report16_mde_spike.py` | (inline; data `reports/applications/claude_insurance_study/data/alpha_draws_T*.npz`) | 20260618 | computed |
| Fig 7.5.5 | 7.5.5 | 2×2 forest plot of per-cell global-slope posteriors (median, 90% CI, P(slope<0)); companion to @tbl-2x2 — **embedded as Figure 5** (`figures/report_2x2_forest.png`) | `spikes/report_2x2_forest_spike.py` | committed per-condition α draws (computes canonical C9/C10/C12/C13 draw-level population-OLS slopes) | deterministic render (no RNG) | computed |
| Tab 7.5.5 | 7.5.5 | @tbl-2x2: per-cell global-slope posteriors (C9–C13) | `analysis/model_estimation.py` | `applications/*/configs/` | per-cell (see cell data) | computed |
| Fig 7.6.6 | 7.6.6 | Prior-sensitivity forest: per-cell slope posteriors under baseline + 3 alternative α priors — **embedded as Figure 6** (`figures/report_prior_sensitivity_forest.png`; claims_ledger C17) | `spikes/report_prior_sensitivity_spike.py` (+ `models/m_0_prior_sweep.stan`) | committed per-condition Stan data; refits cached under `reports/applications/*/data/prior_sweep/` | base_seed 20260701 | computed |

> **Embedding status (2026-07-01):** the six headline figures above are now
> embedded in the paper body via Quarto `@fig-` cross-references (Figures 1–6),
> for the pre-arXiv review round. The two SVG-only foundational figures (§4.3
> scatter, §6.5 δ ECDF) were converted to PDF with `rsvg-convert`; the other
> four ship as committed PNGs in `figures/`. §4.4 keeps the m_0 SBC result in
> prose (no separate m_0-only artifact); the embedded δ ECDF (Figure 3, §6.5)
> carries the marginal-SBC demarcation for both models.

> The two foundational-report figures (§4.3, §6.5) are driven by the
> canonical design constants fixed **in-script** in
> `reports/foundations/04_parameter_recovery.qmd` and `06_sbc_validation.qmd`
> (via the generic `parameter_recovery.py` / `sbc.py` drivers), not by the
> smoke-scale `configs/*recovery*.json` / `configs/*sbc*.json` files — so their
> Config column records the inline canonical design rather than a JSON path.
>
> The §1.8 dependency chain is an **authored** LaTeX `\underbrace` display in the
> body (not a generated figure, not mermaid). The §7.5.5 forest plot
> (`report_2x2_forest.png`) **computes** the canonical per-cell slope posteriors
> (draw-level population-OLS, claims_ledger C9/C10/C12/C13) directly from the
> committed per-condition α draws and renders them, so the figure, @tbl-2x2, and
> the ledger share a single estimator by construction. The superseded
> report-level values (−31 / −3.6 / −18.8) were inflated by exactly 1.25 by a
> ddof-mismatched estimator; see the E.2 correction note and the spike's results
> JSON, which records the superseded values and the factor.
>
> **Non-figure diagnostic spikes** feeding body numbers: 
> `spikes/report_design_diagnostics_spike.py` (seed 20260705) — §5.5
> lottery-spanning ranks, §5.6 β contrast-Jacobian ranks, §7.2 feature-matrix
> rank, §7.4 η-gap/α-contraction callout (ledger C18). 
> `spikes/report_refit_sensitivity_spike.py` (seed 42) — §7.4/D.2
> stricter-sampler refit check (ledger C19).
