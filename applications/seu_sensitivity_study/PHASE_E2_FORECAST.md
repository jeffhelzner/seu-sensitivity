# Phase E2: Full-Run Cost and Compute Forecast

Date: 2026-09-06

## Basis and labels

**Measured** figures come directly from completed API collections or Stan fits.
**Extrapolated** figures scale measured observations to a proposed design.
**Configured** figures are deterministic counts from the current study config.

Provider prices were rechecked on 2026-09-06. Standard input/output prices per
million tokens remain: GPT-4o $2.50/$10, GPT-4o mini $0.15/$0.60, o3-mini
$1.10/$4.40, Claude Sonnet 4.5 $3/$15, and Claude Haiku 4.5 $1/$5. The Sonnet
thinking arm uses the Sonnet output rate, including thinking tokens. Both
providers publish a 50% Batch API discount for these models.

## API-call scenarios

There are 18 cells per pool and two presentations per menu.

| Scenario | Menus per pool | Pools | Choice calls | Status |
|---|---:|---:|---:|---|
| E1 smoke | 12 | 2 | 864 | measured |
| Primary families only | 100 | 2 | 7,200 | configured alternative |
| Full two-lattice-pool design | 140 | 2 | 10,080 | configured candidate |
| Historical three-pool design | 100/140/140 | 3 | 13,680 | configured, not recommended |

The two-lattice-pool candidate includes venture startup/procurement and hiring
candidates/matched families. Insurance has been demoted and must not silently
return as an estimation pool. Adding any third lattice pool is an E3 decision.

## API-cost forecast

The corrected E1 collection cost $4.143126 for 864 calls, or $0.004795 per call.
Scaling by call count gives the decision-relevant forecast:

| Scenario | Standard API | 50% Batch | Derivation |
|---|---:|---:|---|
| Primary families only, two pools | $34.53 | $17.26 | measured cost x 8.333 |
| Full two-lattice-pool design | $48.34 | $24.17 | measured cost x 11.667 |
| Historical three-pool design | $65.60 | $32.80 | measured cost x 15.833 |

These are extrapolations from a menu-size-balanced smoke. The original
token-based pre-collection estimate was $4.18 after its measured 1.67 payload
multiplier, within 1% of the corrected $4.143 actual, which supports this use as
an aggregate planning anchor. Family wording, retries, provider tokenizer
differences, and reasoning-token variability can still move realized cost.

For budgeting, use a 25% contingency: **$60.42 standard** or **$30.21 Batch**
for the proposed 10,080-call collection. Do not count Batch savings until the
pipeline has passed a small provider-specific Batch dry run.

Exact per-arm E1 usage was overwritten by the subsequent phase-local summary.
Consequently, a numeric per-arm allocation would add false precision. Each of
the six model arms contributes exactly 1,680 calls to the proposed design, but
their dollar shares should be recomputed from persisted input, output, cached,
and thinking token counts during the E4 dry run. The full-run collection must
archive that usage table separately from `run_summary.json`.

## Compute forecast

The final number of retained draws has not yet been frozen. Full validated
J=18 runs provide the primary runtime evidence; E1's 800-draw fits provide
real-data context only.

| Fit | Draws/chain | Hours | Status |
|---|---:|---:|---|
| Phase F hard dataset | 800 | 3.16 | measured |
| Phase G replicate 1 | 800 | 1.22 | measured |
| Phase G replicate 2 | 1,600 | 2.00 | measured |
| Phase G replicate 3 | 1,600 | 8.34 | measured |
| E1 venture | 800 | 1.27 | measured real-data context |
| E1 hiring | 800 | 0.33 | measured real-data context |

Runtime is strongly dataset-dependent and is not linear in menu count or draw
count. For scheduling, reserve the observed 1,600-draw upper envelope of
**8.34 hours per fit**, not the fast E1 times. This remains below the 12-hour
per-launch policy but leaves limited headroom for a harder realized posterior.

The fixed utility-spacing grid requires three fits per pool at middle utilities
0.35, 0.50, and 0.65; the 0.50 fit is the primary fit, not an additional run.
For two pools this is six fits:

| Campaign | Fits | Serial hours | Four-core fit-hours | Status |
|---|---:|---:|---:|---|
| Fast observed 1,600-draw anchor | 6 | 12.0 | 48.0 | extrapolated from G replicate 2 |
| Conservative observed envelope | 6 | 50.0 | 200.2 | extrapolated from G replicate 3 |
| 12-hour policy ceiling | 6 | 72.0 | 288.0 | administrative bound |

Fits should be launched one at a time unless independent hardware is assigned;
parallel local fits would confound timing and compete for cores. If an
individual fit approaches 12 hours, preserve its outputs and stop for review
before launching the next sensitivity value.

The optional `Dirichlet(10,10)` regularization sensitivity is not included. If
E3 promotes it into the final campaign, add two fits and reserve another
16.7 hours at the conservative observed envelope, with a 24-hour policy-ceiling
allocation split into separate launches.

## E3 recommendation

Proceed to an explicit design freeze with these defaults:

1. Keep venture and hiring as the two estimation pools; do not restore insurance.
2. Keep all 140 menus per pool unless cost, rather than inferential scope, forces the 100-menu primary-family alternative.
3. Use `sigma_cell` as the primary RQ4 operationalization and ordering agreement descriptively.
4. Retain the fixed spacing grid 0.35/0.50/0.65 and exclude the optional concentrated-prior fit unless separately justified.
5. Prefer Batch collection after a provider-specific E4 dry run; approve a $31 Batch ceiling or a $61 synchronous fallback ceiling for choices.
6. Archive immutable token usage separately and enforce the 12-hour per-fit stop rule.

No production API collection or final-fit campaign should start until these E3
choices are explicitly approved and frozen.

## Sources

- OpenAI pricing: `https://developers.openai.com/api/docs/pricing`
- Anthropic pricing: `https://platform.claude.com/docs/en/about-claude/pricing`
- E1 evidence: `applications/seu_sensitivity_study/PHASE_E1_VALIDATION.md`
- Full-fit evidence: `applications/seu_sensitivity_study/PHASE_F_VALIDATION.md`
- Recovery evidence: `applications/seu_sensitivity_study/PHASE_G_VALIDATION.md`