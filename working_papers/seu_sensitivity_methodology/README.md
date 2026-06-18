# SEU Sensitivity — Methodological Paper

**Title (working):** *Sensitivity to Subjective Expected Utility Maximization: A Methodological Study, with an Illustrative Application to LLM Decision-Making*

**Author:** Jeff Helzner

This folder is the home for the methodological paper and all of its supporting
materials. The paper is authored in Quarto (`.qmd`) and compiled to PDF.

## Status

- **Stage:** complete draft. All sections (§§1–8) and Appendices A–E are drafted
  in `paper/` and compile end-to-end to `paper/_output/paper.pdf` (~46 pp.) with
  no undefined references or citations. Every empirical claim (C1–C16) is
  `computed` in [claims_ledger.md](claims_ledger.md) and every figure-manifest
  row is resolved. Remaining items before deposit are reproducibility-only:
  pinning the v1 commit/DOI (E.0) and a timed clean-room compute-budget run
  (E.1). See [CHANGELOG.md](CHANGELOG.md).
- **Version policy:** a **frozen v1** is deposited for a citable identifier; a
  later **v2** follows once the alignment-study companion paper lands and the
  forward references can be tightened. See [CHANGELOG.md](CHANGELOG.md).

## Contents

| Path | Purpose |
| --- | --- |
| `paper/` | Quarto source for the paper (`paper.qmd` + `sections/*.qmd`), compiled to PDF in `paper/_output/`. |
| `figures/` | Generated figures for the paper. |
| `plan_for_methodological_paper.md` | The canonical writing plan (Part I strategy + Part II outline). |
| `REVIEW.md` | Review of the plan: risks, improvements, venue/structure/sequencing advice. |
| `claims_ledger.md` | Every empirical claim → source report → status (computed / placeholder / re-run-pending). |
| `figure_manifest.md` | Per-figure provenance: figure → script + config + seed. |
| `CHANGELOG.md` | Version history and the v1/v2 policy. |

## Provenance / single source of truth

Supporting computational materials are **referenced at a pinned repository
commit**, not copied into this folder, so there is a single source of truth and
no drift. Pin the commit/tag before any archive deposit.

- Repository: `seu-sensitivity`
- Pinned commit for v1: _to be set at deposit time_ (planning baseline: `0b47efe`)
- Analysis scripts: `analysis/`
- Stan models: `models/`
- Configs: `configs/` and `applications/*/`
- Foundational + application reports: `reports/`

## Dissemination plan

- **Primary archive:** arXiv — `stat.ME` primary, `cs.AI` cross-list
  (optionally `econ.EM` / `stat.AP`).
- **Mirror:** PhilArchive (philosophy audience).
- **License:** CC BY (keeps a later journal-submission path open).
- A compiled PDF is linked from the author's homepage once available.

## Compiled PDF

_Link to the compiled PDF here once built._
