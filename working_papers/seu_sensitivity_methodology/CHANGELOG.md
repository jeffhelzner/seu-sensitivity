# Changelog

All notable changes to the methodological paper and its supporting materials.

The paper is a **living working paper** with the following release policy:

- **v1 (frozen):** deposited on arXiv (and mirrored to PhilArchive) to provide a
  stable, citable identifier. Frozen so the planned alignment-study companion
  paper can forward-reference a fixed version.
- **v2 (and later):** revisions after the alignment-study companion lands;
  forward references tightened, errata folded in.

Each archive deposit pins the supporting repository to a specific commit/tag.

## [Unreleased]

### Changed
- **Figure manifest: remaining `_tbd_` rows resolved (E.2 + `figure_manifest.md`).**
  The two standalone-manifest rows that were still `not-started` are now closed:
  - **§7.5.5 2×2 forest plot** of the per-cell global-slope posteriors is
    `computed`. New artifact `spikes/report_2x2_forest_spike.py` →
    `figures/report_2x2_forest.png`, rendering the canonical *report-level* cell
    summaries (claims_ledger C9/C10/C12/C13), so the figure matches @tbl-2x2 by
    construction. The spike also records a population-OLS cross-check
    (`b_i = Cov(T,α_i)/Var(T)` over the committed per-condition draws) in its
    results JSON --- diagnostic only, not plotted: it agrees on P(slope<0) and on
    the two Ellsberg cells, and reflects the insurance-cell slope-convention
    difference already documented in C16. The body retains @tbl-2x2 (the paper's
    no-embedded-figures convention); the forest plot is a tracked reproducible
    artifact. Appendix E.2 gains a matching `Fig (§7.5.5)` row.
  - **§1.8 dependency display** is reclassified from a `not-started` mermaid
    "diagram" to an **authored** LaTeX `\underbrace` dependency chain in the §1.8
    body (no mermaid, no generated figure) --- matching what the section actually
    contains.
- **Figure manifest: pending foundational rows filled (E.2 + `figure_manifest.md`).**
  The three `pending`/`not-started` foundational-report figures --- §4.3 (`m_0`
  α true-vs-estimated recovery scatter), §4.4 (`m_0` SBC), §6.5 (`m_1` SBC) ---
  are now `computed`. Their provenance is the in-script canonical design of
  `reports/foundations/04_parameter_recovery.qmd` and `06_sbc_validation.qmd`
  (generic `parameter_recovery.py` / `sbc.py` drivers), **not** the smoke-scale
  `configs/*recovery*.json` / `configs/*sbc*.json` files the placeholders
  referenced; the Config column now records the inline canonical design
  (recovery `m_0`: M25 K3 D5 R15, 50 iters; SBC `m_0`/`m_1`: N_sbc=999 thin4 1
  chain). Seeds: design np 42; recovery 12345+i sim / 54321+i fit (D.2); SBC
  123+i per draw. E.2 intro prose updated (no rows remain pending).
- **Appendix D.3 SBC seed convention corrected.** D.3 previously claimed SBC
  per-draw seeds follow the D.2 "12345+i / 54321+i simulate/fit" pair; the actual
  `analysis/sbc.py` implementation uses a **single** seed `123 + i` per replicate
  (the `_sbc` program draws the true parameters and simulates its dataset in the
  same pass it fits), with the study design fixed once under NumPy seed 42.
  Corrected to match the code so the manifest's D.3 reference is accurate.


### Added
- **Appendices C/D/E (plan execution-order step 9).** Drafted:
  - **Appendix C** (`sections/_appendix_c.qmd`): full Stan listings for the five
    programs referenced by the body code excerpts --- `m_0.stan` (C.1),
    `m_0_sim.stan` (C.2), `m_1.stan` (C.3), `m_1_sim.stan` (C.4), and the
    calibrated-prior `m_01.stan` given as a one-line prior diff (C.5,
    `Lognormal(3.0,0.75)` vs the `Lognormal(0,1)` default). Notation cross-walk
    to §2.6; the repetitive generated-quantities PPC test-statistic boilerplate
    is shown in full for `m_0` and summarized for `m_1`, with bookkeeping arrays
    elided and a pointer to the pinned `models/`.
  - **Appendix D** (`sections/_appendix_d.qmd`): the **canonical** design
    constants. D.0 foundational design (K=3, D=5, R=15, S=15, M,N∈{25,50}, 2–5
    alts, normal features); D.1 software versions (CmdStan 2.37.0, Python 3.10,
    CmdStanPy ≥1.2, env `seu-sensitivity`); D.2 sampler (4 chains, 1000 warmup /
    2000 sampling, NUTS defaults adapt_delta=0.8 / max_treedepth=10, seeds
    12345+i / 54321+i) + diagnostics policy (R̂≤1.01, bulk/tail ESS, divergence
    fraction; ties to C14); D.3 SBC (N_sbc=999, thin=4, single chain, 20 bins,
    L=N_sbc, ECDF/KS bands); D.4 matched A/B/C/D study (n=100 re-run, pilot 30
    superseded; the four-slice protocol, C1–C3); D.5 application scale (M=100×3
    counterbalanced = 300 choices/condition, 5 temps/cell, R=30, D=32 PCA
    insurance, K=3/4, 20 recovery iters/cell, provider temperature grids,
    calibrated α priors incl. K=4 Ellsberg recalibration).
  - **Appendix E** (`sections/_appendix_e.qmd`): E.0 citation/provenance policy
    (pinned commit + Zenodo DOI single source of truth, reference-not-copy, the
    analysis-script/config table); E.1 compute-budget paragraph (SBC at N_sbc=999
    dominates; matched recovery 400 fits + 20/80 application fits cheaper;
    overnight-scale end-to-end, explicitly flagged as order-of-magnitude
    pending a timed clean-room run); E.2 per-figure manifest table (4 `computed`
    rows: Fig B.2, Fig 6.4, §7.5.5 2×2 table, Fig 7.5.2 MDE; 3 `pending`
    foundational-report figures) plus the C11/C14/C15 spike pointers.
  - `paper.qmd` now includes §§1–8 + Appendices A–E; drafting-status comment
    updated to "all sections drafted." Renders clean (pdfLaTeX), 0 unresolved
    refs (`{??}` count = 0), no undefined citations.

- **Paper draft scaffold + formal core (plan execution-order steps 3–4).**
  `paper/` Quarto project (`_quarto.yml`, `paper.qmd`, `references_extra.bib`,
  `sections/`) compiling to `paper/_output/paper.pdf` (pdfLaTeX; 14 pp. for the
  formal core). Drafted:
  - **§2 Abstract model** (`sections/_02_abstract_model.qmd`): softmax rule, the
    three characterizing properties, SEU specialization, consolidated notation
    table, the υ-endpoint convention and the β (additive row-shift) gauge, and
    the ridge-vs-weak-IV terminological demarcation.
  - **§3 m_0 identifiability** (`sections/_03_m0_identifiability.qmd`): Prop 3.1
    (α from η), Prop 3.2 (weak/ridge identifiability of (β,δ), numerically-
    supported proposition), and the §3.5 choice-probability-preserving
    α-separation argument (per the B.2 spike framing).
  - **§5 m_1 identifiability** (`sections/_05_m1_identifiability.qmd`): two-step
    structural argument (β-free risky EU + lottery diversity), Props 5.1–5.3,
    and the **single-α conditional caveat** (gating item; §5.3.3).
  - **Appendix A** (`sections/_appendix_a.qmd`): full proofs of the three
    softmax properties + scale invariance (Thm A.4).
  - **Appendix B** (`sections/_appendix_b.qmd`): B.1 (α-from-η proof), B.2
    (Fisher-curvature proposition with the spike's κ / Schur-ratio /
    choice-prob-sensitivity numbers), B.3 (δ via affine independence — full
    inversion proof), B.4 (β modulo row-shift gauge).
  - Citations resolve cleanly against `reports/references.bib` +
    `references_extra.bib` (adds Modřák 2023, Ellsberg 2001, Staiger–Stock,
    Stock–Wright–Yogo, Andrews–Cheng). No undefined refs/citations on render.
- **Implementation half (plan execution-order step 5).** Drafted:
  - **§4 m_0 implementation** (`sections/_04_m0_implementation.qmd`): data /
    params / priors (Lognormal(0,1) α, with the §7-prior cross-reference), the
    β-gauge handling, prior-predictive SEU-max-rate, α-recovers-/-βδ-don't,
    and the **marginal-SBC demarcation** as a callout (ledger C5–C7).
  - **§6 m_1 implementation** (`sections/_06_m1_implementation.qmd`): the
    matched A/B/C/D design table, the **n=100 re-run** results (ledger C1–C3),
    the single-α conditional callout (§6.4.1 — the remaining gating item, now
    done), and the corrected §6.4.4 two-phenomena lesson + summary table
    (`@tbl-matched`). The pilot's ≈15% / ≈2% figures are explicitly marked
    superseded throughout.
  - `paper.qmd` now includes §§2–6 + Appendices A–B; renders clean (pdfLaTeX),
    no undefined refs/citations.

- **Application (plan execution-order step 6).** Drafted:
  - **§7 illustrative application** (`sections/_07_application.qmd`): the 2×2
    LLM × task design (GPT-4o / Claude 3.5 × insurance / Ellsberg), the `m_01`
    calibrated-prior fit (ledger C8), application-scale validation (recovery,
    SBC-inheritance lemma, MCMC diagnostics C14, PPC C15), the four-cell results
    (C9–C13) with the 2×2 reading table (`@tbl-2x2`), the Claude-null MDE
    callout (C16) and cross-LLM robustness (C11), the construct-validity
    layering, and the §7.6 negative-scope / §7.7 follow-up subsections. Opens
    with the reading-guide + uniform reporting-template callouts per the plan.
  - `paper.qmd` now includes §§2–7 + Appendices A–B; drafting-status comment
    updated (only §§1, 8 and remaining appendices outstanding).

- **Discussion (plan execution-order step 8).** Drafted:
  - **§8 discussion** (`sections/_08_discussion.qmd`): 8.1 restatement of what
    the instrument measures; 8.2 the identifiability mantra with the δ
    (identifiable-not-estimable) and α (precision-tracks-quantity-not-type)
    sides kept apart plus the n=30→n=100 cautionary instance; 8.3 the
    **marginal-SBC demarcation** callout (carries the `#sec-discussion` label
    that §4.4 and §6.5 forward-reference); 8.4 why α is primary, with the
    model-conditional interpretive caveat; 8.5 limitations/extensions
    (δ-optimal design, single-α / `m_2`, functional form, `h_m01`, companion
    work); 8.6 closing. No new numbers; restates §§3–7 findings.
  - `paper.qmd` now includes §§2–8 + Appendices A–B; drafting-status comment
    updated (only §1 and remaining appendices outstanding). Renders clean
    (pdfLaTeX), no undefined refs/citations.

- **Motivation + abstract polish (plan execution-order step 7).** Drafted:
  - **§1 motivation** (`sections/_01_motivation.qmd`): 1.1 external-vs-procedural
    questions; 1.2 why labeled accuracy under-determines decision quality
    (absent / costly / judgment-dependent / luck-confounded labels); 1.3 the
    case for procedural evaluation; 1.4 SEU as the *stated standard* (conditional
    framing, not a defense of SEU); 1.5 why a graded α-measure over a binary
    verdict; 1.6 scope + consolidated negative-scope flags; 1.7 the four-part
    contribution callout (α-from-η + (β,δ) ridge as numerically-supported
    proposition; in-principle-≠-finite-n demonstration with the n=30→n=100
    reversal; marginal-SBC demarcation; honest full-pipeline application); 1.8
    roadmap with a LaTeX dependency-chain display (mermaid avoided for clean
    pdfLaTeX). No new numbers; all figures cited from later sections.
  - **Abstract polish** (`paper.qmd`): tightened the validation clause and
    replaced the abrupt "We report the findings honestly." with a substantive
    finite-sample-caveats sentence; no substantive content change.
  - `paper.qmd` now includes §§1–8 + Appendices A–B; renders clean (pdfLaTeX),
    no undefined refs/citations.

- `REVIEW.md`: review of the writing plan (risks, improvements, venue/structure/sequencing advice).
- `claims_ledger.md` and `figure_manifest.md` provenance tracking documents.
- Migrated `plan_for_methodological_paper.md` from `local/` into this folder.
- `spikes/b2_fisher_block_spike.py` + `spikes/B2_RESULTS.md` + `figures/b2_fisher_spike.png`: the B.2 Fisher-block spike (gating task 1).
- `spikes/report16_mde_spike.py` + `spikes/report16_mde_results.json` + `figures/report16_mde_power_curve.png`: the Claude-null minimum-detectable-effect / power calculation (gating item C16).
- `spikes/report11_cross_llm_spike.py` + `spikes/report11_cross_llm_results.json`: cross-LLM insurance slope comparison, full-grid vs Claude-grid-restricted (ledger C11).
- `spikes/report1415_diagnostics_ppc_spike.py` + `spikes/report1415_diagnostics_ppc_results.json`: MCMC diagnostics + PPC aggregation across all 20 factorial fits (ledger C14, C15).

### Resolved (gating tasks)
- **B.2 Fisher-block / near-kernel spike (gate 1).** Computed at K=3, D=5, M=50
  over 4 representative draws + α sweep. Findings: gauge-fixed (β,δ) block is
  ill-conditioned (κ ≳ 10³); δ profiled/marginal Fisher ratio 0.01–0.37; flat
  directions preserve choice probabilities to 2–8% of a typical direction.
  Decision: B.2/§3.4 = numerically-supported **proposition** (not theorem);
  §3.5 α-separation phrased as **choice-probability-preserving** (not raw-η);
  no strict (β,δ) invariance group claimed. Ledger row C4 → `computed`.
- **Report-14 re-run at n = 100 (gate 2).** Matched-design A/B/C/D recovery via
  `spikes/report14_rerun_analysis.py` (→ `report14_rerun_results.json`,
  `figures/report14_rerun_matched.png`). **⚠ The 30-iteration pilot did not
  replicate:** central B→C α gain is null (−8.1%, 90% CI [−22.0, +6.8], vs the
  pilot's ≈15%); the A→B control *flips* to a +27.1% improvement (CI [+15.0,
  +38.5]); δ CI-width narrows only ≈0.8% (Wilcoxon p≈1.2e-6) vs the pilot's ≈2%.
  Ledger rows C1–C3 → `computed`; §6.4.4 contribution table flagged for
  reframing. C16 (Claude-null MDE) stays pending — separate §7 power calc.
- **Claude-null minimum-detectable-effect (C16).** Computed directly from the
  saved per-condition posterior α draws (`alpha_draws_T*.npz`, 4 000/condition)
  via the application's own draw-wise OLS slope functional
  (`spikes/report16_mde_spike.py` → `report16_mde_results.json`,
  `figures/report16_mde_power_curve.png`, seed 20260618; recomputed slope
  matches `primary_analysis.json` exactly). **MDE ≈ 36 α-units per unit T** at
  P(slope<0) ≥ 0.95 (Gaussian 36.3 / empirical 34.7 / constant-CV MC 36.0;
  ≈ 41–43 at P ≥ 0.975) — ≈ 12× the observed |slope| (≈ 2.9), ≈ 0.53 of
  grand-mean α. The null is *inconclusive at the achievable resolution*; even a
  GPT-4o-sized slope (≈ 31, full-grid ref) sits below the floor. Ledger row
  C16 → `computed`; §7.5.2 MDE sentence + manifest Fig 7.5.2 added. **This was
  the last computational gating item.**
- **Claims-ledger placeholders C11–C15 transcribed to `computed`.**
  - **C11 (cross-LLM, §7.5.2a).** P(GPT-4o slope < Claude slope) = 0.817 on the
    full grids, **0.824** restricting GPT-4o to Claude's T ≤ 1.0 — the
    directional LLM contrast is robust to the unequal-grid confound
    (`spikes/report11_cross_llm_spike.py`; full-grid recomputation matches
    ledger C9 ≈ −31 and the factorial report's ~0.82).
  - **C12 / C13 (Ellsberg cells, §7.5.3–7.5.4).** GPT-4o × Ellsberg slope median
    −38.4 [−72.1, −10.0], P(slope<0) 0.984, P(mono↓) 0.090; Claude × Ellsberg
    slope median −18.8 [−65.3, 24.5], P(slope<0) 0.766, P(mono↓) 0.0085 (no
    pattern) — transcribed from each study's `primary_analysis.json`.
  - **C14 (diagnostics, §7.4.3) — ⚠ plan wording corrected.**
    `spikes/report1415_diagnostics_ppc_spike.py` over all 20 fits: **α never
    R-hat-flagged, ESS satisfactory in every fit**; divergences ≤ 0.15%/fit but
    across 12/20 fits (not just the 2 highest GPT-4o T); R-hat > 1.01 in 4/20
    fits confined to weakly-identified β/δ + per-trial nuisance latents
    (eta/psi/upsilon) in the hard K=4 GPT-4o×Ellsberg fits — which corroborates
    the §3.4/§6.4 β,δ ill-conditioning thesis. §7.4.3 reframed accordingly.
  - **C15 (PPC, §7.4.4).** Holds as stated: 60/60 PPC p-values in [0.317, 0.656]
    ⊂ [0.3, 0.7], mean 0.462.

### Pending (gating tasks before first full draft)
- Per-figure manifest populated as figures are produced.
- Single-α conditional caveat: **§5.3.3 done**, **§6.4.1 done**.
- Pinned commit/tag + Appendix E compute-budget paragraph.
- §1.7(a) contribution claim matched to what B.2 licenses (proposition, not theorem) — pending (§1 not yet drafted).

### Drafting status (plan execution order)
- [x] (1) B.2 spike · [x] (2) Report-14 re-run · [x] (3) §§2,3,5 formal core ·
  [x] (4) Appendix B alongside §§3,5 · [x] (5) §§4,6 implementation ·
  [ ] (6) §7 application · [ ] (7) §1 + abstract · [ ] (8) §8 + references.
