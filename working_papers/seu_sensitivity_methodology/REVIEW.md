# Review of the Methodological-Paper Plan

A review of [`plan_for_methodological_paper.md`](plan_for_methodological_paper.md)
(*Sensitivity to Subjective Expected Utility Maximization: A Methodological
Study, with an Illustrative Application to LLM Decision-Making*).

The plan is unusually disciplined: it already pre-empts most first-order
objections (identifiability-vs-estimability hygiene, the marginal-SBC
demarcation, Anscombe–Aumann de-emphasis, a consolidated negative-scope list,
and explicit honest-reporting commitments). This review therefore targets
**second-order risks**, **scope/sequencing**, and the **venue/structure**
decisions.

All 14 foundational reports and 5 application reports cited by the plan were
verified to exist under `reports/`.

---

## 1. Risks (with remediation)

### R1 — Thesis written before its gating computations *(highest priority)*
The framing spine depends on two "must-do" computations that have **not been
run**: the Appendix B.2 Fisher-block / near-kernel spike, and the Report-14
re-run at n ≥ 100. The headline numbers (~15% α gain, ~2% δ gain, the
A→B-worsening control) and the "why α survives the ridge" argument (§3.5) are
contingent on results that do not yet exist.
**Remediation:** run both before locking framing. Pre-write a fallback framing
in case (a) the (β, δ) near-flat direction is *not* approximately in the
η-Jacobian's near-kernel (which would break the §3.5 separation of α), or
(b) the re-run shifts the percentages materially. Tracked as `re-run-pending`
rows C1–C4, C16 in [`claims_ledger.md`](claims_ledger.md).

### R2 — Appendix B.2 is load-bearing yet least mature
The plan itself flags B.2 as "the most delicate (a curvature claim, not an
invariance claim)," but contribution bullet §1.7(a) advertises
"(β, δ) ridge identifiability written as **propositions**."
**Remediation:** hedge §1.7(a) to "a numerically-supported proposition" unless
the spike licenses a stronger statement; never promote B.2 from a curvature
claim to an invariance claim without numerical verification of a candidate
η-preserving family. *(Applied as an edit to the plan.)*

### R3 — Unfinished numbers visible to a referee
§7.5.3 and §7.5.4 carry `[from report]` placeholders; the cross-LLM number in
§7.5.2a and the diagnostic counts in §7.4.3–7.4.4 are not yet transcribed.
**Remediation:** gate the first full draft on transcribing all placeholders;
track each in [`claims_ledger.md`](claims_ledger.md) (rows C11–C15) and flip to
`computed` only from the source report.

### R4 — Length vs. content tension
A 35–45 page target must absorb 8 sections, 5 appendices, four proofs, and a
full 2×2 application. §7 alone is budgeted at 6–8 pages and the appendices carry
the proof load.
**Remediation:** add explicit **per-section page budgets** and a decision rule
for what moves to an online supplement vs. cited reports. *(Page-budget block
added to the plan's Part I.)*

### R5 — The single-α assumption is load-bearing but untested in-paper
§5.3.3 makes a *single* α govern both the uncertain and risky blocks, and §6.4.1
explicitly leans on this to attribute the α-RMSE gain to the *type* of data. The
generalization that would test it (`m_2`, block-specific α) is out of scope.
**Remediation:** either add a light posterior-predictive / robustness check that
a shared α is not grossly violated in the application data, or strengthen the
conditional caveat so the attribution is explicitly conditional on the
assumption. *(Caveat strengthened in the plan.)*

### R6 — The Claude "declines to support an effect" framing
§7.6.1(iii) presents the Claude null as a *feature* ("the framework declines to
support an effect"), but the evidence is an **inconclusive** null
(P(slope<0) ≈ 0.56, 90% CI ≈ [−54, 39]). Absence of evidence is not evidence of
absence; a referee will press on this.
**Remediation:** reframe as "the framework **does not detect** an effect" and add
a **minimum-detectable-effect / power** statement at the design's n, so the null
is calibrated rather than asserted. *(Applied to §7.5.2 / §7.6.1; ledger row
C16.)*

### R7 — Reproducibility burden is easy to defer
The per-figure manifest and compute-budget paragraph are declared first-draft
preconditions (Appendix E.1–E.2) but are exactly the artifacts that slip.
**Remediation:** build [`figure_manifest.md`](figure_manifest.md) as figures are
produced; treat it and the compute-budget paragraph as definition-of-done gates.

### R8 — Provenance points at moving targets
Empirical claims trace to internal `.qmd` reports, and the dissemination plan
routes through homepage blog posts. Neither is a stable citation.
**Remediation:** cite the repository at a **pinned commit/tag** (ideally an
archived DOI, e.g. Zenodo) for every load-bearing claim; never cite a blog URL
as load-bearing. *(Recorded in [`README.md`](README.md) and the plan's
Appendix E.)*

### R9 — Thin related-work / positioning
For a 2026 submission in a fast-moving area, the plan under-positions against the
LLM-evaluation literature and the partial/weak-identification literature (beyond
the Stock–Wright–Yogo demarcation it already makes).
**Remediation:** add a short positioning subsection (or expand §1.7) and the
corresponding references.

### R10 — Style/venue identity mismatch
The plan's self-description ("CMU-style formal philosophy," "Target venue:
PhilArchive") undersells what is, at its center of gravity, a Bayesian-workflow /
identifiability **methodology** paper with a decision-theoretic framing.
**Remediation:** reframe the primary identity as statistical methodology with a
decision-theoretic motivation; keep the philosophical framing as voice, not as
the disciplinary home. Ties directly to the venue advice below. *(Target-venue
line updated in the plan.)*

### R11 — The parallel alignment study pressures the scope firewall
Drafting while the alignment study runs creates standing temptation to import
multi-LLM/multi-prompt results and the hierarchical `h_m01` model.
**Remediation:** freeze the application at the 2×2; the alignment study is
**companion-only** with a single forward reference, as the plan already
specifies. Hold the line.

### R12 — SBC configuration may draw a workflow referee's question *(minor)*
Single-chain SBC with thin = 4 at N_sbc = 999, while citing Modřák et al.'s
ECDF method, invites a methods reviewer to ask about multi-chain SBC and
thinning sensitivity.
**Remediation:** justify the configuration in Appendix D.3 or note it as an
explicit limitation.

---

## 2. Improvements

- **I1 — Claims ledger.** A one-page claim → source → status table.
  Operationalizes the honest-reporting commitments. *(Created:
  [`claims_ledger.md`](claims_ledger.md).)*
- **I2 — Consolidated notation table.** The α/β/δ/υ/η/ψ/π symbols and the two
  gauges are currently introduced across §2.6, §3.1, §5.2. A single glossary
  block reduces reader load. *(Added to the plan's §2.6 as a notation table.)*
- **I3 — Two-phenomena summary figure/table.** The core contribution is that the
  α-sharpening story and the δ identifiability≠estimability story are *distinct*.
  A small side-by-side table makes it citable. *(Added to §6.4.4.)*
- **I4 — Gating checklist / definition-of-done.** A single checklist that must be
  green before the first full draft. *(Added to the plan's Part I; mirrored in
  [`CHANGELOG.md`](CHANGELOG.md).)*
- **I5 — Version discipline.** Version + date + changelog for a living working
  paper. *(Created: [`CHANGELOG.md`](CHANGELOG.md); policy = frozen v1 → v2.)*
- **I6 — Optional OSF / registered analysis plan.** The §7.1b reporting template
  is already called "pre-registered"; depositing it on OSF before transcribing
  the Ellsberg cells would make that literal and strengthen credibility.

---

## 3. Venue advice

The paper's center of gravity is **statistical methodology with an AI
application**, not philosophy proper. Recommended (and adopted):

- **Primary archive: arXiv** — `stat.ME` primary, `cs.AI` cross-list
  (optionally `econ.EM` / `stat.AP`). Maximizes discoverability and citability,
  matches the true center of gravity, and the AI cross-list reaches the
  application audience.
- **Mirror: PhilArchive** — a legitimate secondary archive that serves the
  philosophy audience and the author's disciplinary identity. Cross-posting a
  preprint across archives is permitted.
- **SocArXiv** — optional; natural *if* an OSF pre-registration is done (I6) or
  if social/behavioral methodologists are a target audience, but it has lower
  ML/stats visibility than arXiv.
- **License: CC BY** — keeps a later journal-submission path open.
- **Version policy: frozen v1, later v2** — v1 gives the alignment-study
  companion a stable citation; v2 tightens forward references once the companion
  lands.

**Verify before depositing (do not over-assert):**

- arXiv first-time submitters need an **endorser** in the chosen category.
- If a journal is eventually targeted, confirm it permits preprints (most
  stats/ML venues do).

---

## 4. Folder-structure advice

```
working_papers/
  seu_sensitivity_methodology/
    README.md                          # links PDF + pinned commit; dissemination plan
    CHANGELOG.md                       # v1-frozen / v2 policy
    REVIEW.md                          # this document
    plan_for_methodological_paper.md   # migrated from local/
    claims_ledger.md                   # claim → source → status
    figure_manifest.md                 # figure → script + config + seed
    figures/                           # generated figures
    paper/                             # Quarto source (.qmd), once drafting begins
```

- Supporting computational materials are **referenced at a pinned commit, not
  copied** — single source of truth, no drift. Appendix E already points at
  `analysis/`, `configs/`, `models/`, `reports/`.
- The canonical plan is **migrated out of `local/`** into this folder (the user's
  stated intent: the subfolder holds all supporting materials). `local/` remains
  scratch space.
- Commit `working_papers/` to git. Decide whether the compiled PDF is committed
  or CI-built (recommend CI-built + linked, to keep the repo lean).

---

## 5. Sequencing advice

- **Decouple from the alignment study.** The methodological paper does not depend
  on the alignment study; finalize and archive a citable **v1 ASAP** (after the
  two gating computations) so the alignment companion can forward-reference a
  stable identifier.
- **Run the gating computations first.** The B.2 spike and the Report-14 re-run
  are independent of the alignment study and must not be blocked by it. They are
  prerequisites for drafting §3 and §6 respectively.
- **Hold the scope firewall.** The 2×2 application is frozen; the alignment study
  is companion-only.
- **Percolation pipeline is fine for dissemination, not for citation.** The
  reports → homepage blog posts → future working paper flow is good for reach,
  but load-bearing citations must point to archived/pinned artifacts, never blog
  URLs (see R8).

---

## 6. Where each item landed

| Item | Action |
| ---- | ------ |
| R1, R3, I1 | `claims_ledger.md` (gating rows + status tracking) |
| R2, R5, R6, R10, I2, I3, I4, R8 | Edits applied to `plan_for_methodological_paper.md` |
| R7, I5 | `figure_manifest.md`, `CHANGELOG.md` |
| R4 | Page-budget block added to the plan's Part I |
| R9, R11, R12, I6 | Flagged here; lightweight, addressed during drafting |
| Venue, folder, sequencing | This document + `README.md` |
