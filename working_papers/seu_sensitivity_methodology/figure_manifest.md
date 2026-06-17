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
| _tbd_ | 4.3.2 | α true-vs-estimated recovery scatter (m_0) | `analysis/parameter_recovery.py` | `configs/m1_parameter_recovery_config.json` | _tbd_ | not-started |
| _tbd_ | 4.4 | m_0 SBC rank histograms + ECDF/KS | `analysis/sbc.py` | `configs/m1_sbc_config.json` | _tbd_ | not-started |
| _tbd_ | 6.4.1 | α RMSE: matched-design A/B/C/D | `analysis/parameter_recovery.py` | `configs/m1_matched_recovery_config.json` | _tbd_ | not-started |
| _tbd_ | 6.5 | m_1 SBC rank histograms + ECDF/KS | `analysis/sbc.py` | `configs/m2_sbc_config.json` | _tbd_ | not-started |
| _tbd_ | 7.5.5 | 2×2 forest plot of per-cell global-slope posteriors | application scripts | `applications/*/` | _tbd_ | not-started |

> Update the `configs/` references above to the exact configs that drive each
> headline figure once drafting begins; the entries are best-current-guess
> placeholders pending the gating re-runs.
