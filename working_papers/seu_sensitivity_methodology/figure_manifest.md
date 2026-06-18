# Per-Figure Manifest

Provenance for every figure and table in the paper, so each headline artifact is
reproducible from a pinned commit. Build this **as figures are produced**, not as
a post-draft cleanup — it requires touching every figure in §§4, 6, 7 to record
provenance and is easy to defer and never finish.

This manifest is a **precondition for the first full-paper draft** (plan
Appendix E.2).

| Fig/Tab | § | Description | Script | Config | Seed | Status |
| ------- | - | ----------- | ------ | ------ | ---- | ------ |
| _tbd_ | 1.8 | Section dependency diagram (mermaid) | — (authored) | — | — | not-started |
| Fig B.2 | 3.4 / B.2 | (β,δ) Fisher spectrum tail + δ info lost to β-coupling | `spikes/b2_fisher_block_spike.py` | (inline constants: K3 D5 M50) | design 20260617 / draws 2026 | computed |
| Fig (§4.3) | 4.3.2 | α true-vs-estimated recovery scatter (m_0) | `analysis/parameter_recovery.py` (`reports/foundations/04_parameter_recovery.qmd`) | inline (m_0: M25 K3 D5 R15; 50 iters, 4 chains × 1000) | design np 42; 12345+i sim / 54321+i fit (D.2) | computed |
| Fig (§4.4) | 4.4 | m_0 SBC rank histograms + ECDF/KS | `analysis/sbc.py` (`reports/foundations/06_sbc_validation.qmd`) | inline (m_0: M25 K3 D5 R15; N_sbc=999 thin4 1 chain) | design np 42; 123+i per draw (D.3) | computed |
| Fig 6.4 | 6.4.1 | α/δ RMSE + CI-width: matched-design A/B/C/D (n=100) | `spikes/report14_rerun_analysis.py` | `configs/m1_matched_recovery_n100_config.json` | design 20260617 / boot 20260617 | computed |
| Fig 7.5.2 | 7.5.2 / 7.6.1 | Claude-null MDE / power curve: P(slope<0) vs \|Δα/ΔT\|, MDE ≈ 36 marker | `spikes/report16_mde_spike.py` | (inline; data `reports/applications/claude_insurance_study/data/alpha_draws_T*.npz`) | 20260618 | computed |
| _tbd_ | 6.5 | m_1 SBC rank histograms + ECDF/KS | `analysis/sbc.py` (`reports/foundations/06_sbc_validation.qmd`) | inline (m_1: M25 N25 K3 D5 R15 S15; N_sbc=999 thin4 1 chain) | design np 42; 123+i per draw (D.3) | computed |
| _tbd_ | 7.5.5 | 2×2 forest plot of per-cell global-slope posteriors | application scripts | `applications/*/` | _tbd_ | not-started |

> The three foundational-report figures (§4.3, §4.4, §6.5) are driven by the
> canonical design constants fixed **in-script** in
> `reports/foundations/04_parameter_recovery.qmd` and `06_sbc_validation.qmd`
> (via the generic `parameter_recovery.py` / `sbc.py` drivers), not by the
> smoke-scale `configs/*recovery*.json` / `configs/*sbc*.json` files — so their
> Config column records the inline canonical design rather than a JSON path. The
> remaining `_tbd_` rows (§1.8 diagram, §7.5.5 forest plot) are authored/applied
> artifacts whose provenance is finalized as those figures are produced.
