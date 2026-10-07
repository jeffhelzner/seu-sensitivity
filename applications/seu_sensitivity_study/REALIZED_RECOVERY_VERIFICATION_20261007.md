# Saved-Draw Recovery Verification

Date: 2026-10-07. Baseline: `0976329`.

This is an offline reanalysis of existing recovery chains, not a new fitting,
simulation, or production campaign. It verifies Amendment 5 realized-cell
estimands and the independent review's reported additive summaries. It does
not change any estimand, prior, decision threshold, or the 24-fit plan.

## Evidence and Selection

[The verifier](../../analysis/verify_realized_recovery.py) checks the current
four referenced chains for every iteration, including replacement schedules,
chain IDs, seeds, draw counts, warmup handling, and required parameters. It
maps the saved generating design to canonical cells and reconstructs truth
and posterior contrasts using fixed cell weights. The selected venture-16
replacement uses seed 54336 and 1,000 warmup plus 1,000 sampling draws per chain.
All 120 current iterations verify and pass the saved-summary sampler gates.

[The new evidence artifact](../../reports/applications/seu_sensitivity_study/data/realized_recovery_verification.json)
contains per-iteration results, denominators, methodology, limitations, and
914 input hashes, including 480 selected chain files. It reproduces exactly:

```bash
python -m analysis.verify_realized_recovery --check --workers 4
```

Use the project Python environment and the recorded local saved inputs.
Historical September evidence and chain/truth artifacts are unchanged.
The new verifier has 32 focused regression tests. Independent review also
checked every selected chain header, all truth contrasts, and exact draw-level
reproduction for venture-16 and matched iteration 2.

## Realized Results

Central intervals have 90% nominal mass. RQ1 rows below deduplicate the exact
OpenAI sign reversal, leaving six distinct comparisons per dataset; the full
named family is retained in the artifact. Detection uses the unchanged
interval-plus-median practical-effect rule. Outside-ROPE detection conditions
on each contrast's actual realized truth, not its additive counterpart.

| Campaign | Question | Coverage | Mean interval width | Outside-ROPE detection |
|---|---|---:|---:|---:|
| Venture | RQ1, distinct | 222/240 | 0.2844 | 164/173 |
| Venture | RQ2 | 71/80 | 0.2020 | 52/55 |
| Hiring | RQ1, distinct | 219/240 | 0.2772 | 164/173 |
| Hiring | RQ2 | 75/80 | 0.1982 | 52/55 |
| Matched | RQ5 | 218/240 | 0.5418 | 157/173 |
| Venture | RQ6 | 36/40 | 0.0414 | 31/31 |
| Hiring | RQ6 | 37/40 | 0.0401 | 30/31 |
| Matched | Size-slope companion | 34/40 | 0.0555 | 31/31 |

For realized RQ5, posterior-mean bias is -0.01159 and RMSE is 0.16597.
Detection by absolute realized truth magnitude is 27/40 for ratios 1.25-1.5,
54/57 for 1.5-2, and 76/76 above 2. There are zero wrong-sign detections among
170 detected effects. The 13/67 detections below the practical-effect threshold
are not null false positives: no exact-null truths are available to estimate
that rate. Boundary conventions and unrounded values are in the verifier.

All five reviewer detection-table rows reproduce at reported precision.
The review's additive RQ5 rate for ratios 1.25-1.5 is 18/49, distinct from
the realized rate 27/40. Different estimands have different truths and bin
memberships; this is not a paired gain in power at identical true effects.

## Generating Assumptions and Dependence

The recorded generating priors are gamma0 N(2.5, 0.5), each gamma N(0, 0.5),
gamma_size N(0, 0.2), sigma_cell half-normal with SD parameter 0.3, and standard
normal cell residuals. Normal parameters here denote mean and SD. These match
the original primary hierarchy, not Amendment 7's L/H/S alternatives.

Venture and hiring share full realized truth vectors in all 40 paired
iterations. All three geometries share leading hierarchy parameters. Matched
full truths differ, but venture residual entries 7-18 match matched entries
1-12 within saved rounding tolerance in all 40 cases, reflecting shifted use
of the same RNG stream. Thus 120 fitted datasets do not supply 120 independent
truth vectors: they form 40 coupled seed clusters. Contrasts within datasets
are dependent too. Coverage counts such as 218/240 are descriptive aggregate
proportions, not 240 independent Bernoulli trials; no naive binomial precision
interval or universal calibration claim is justified from those denominators.

The generating code uses separate categorical draws within each dataset under
the working model. However, all 120 original temporary choice-input files and
the referenced simulator outputs are unavailable. Exact seed-to-choice replay
and equality of original versus replacement choice vectors cannot be verified
from retained artifacts. Shared seeds also prevent a claim of cross-geometry
independent choices. Available hashes, matching truth records and chain
metadata do not establish that missing historical link. This is a remaining
provenance limit, not a demonstrated wrong-fit selection or numerical mismatch.

## Disposition

Saved-draw calculations for realized RQ1/RQ2/RQ5 and unchanged RQ6 are verified,
subject to the stated historical-input limitation. This does not validate
production behavior, the nine alternative-prior fits, A3 ceiling conclusions,
or A4 predictive adequacy. No new null calibration or SBC is claimed. The
September snapshot remains historical and is not regenerated as amended
evidence. Final review must explicitly accept or require remediation of the
missing choice-input provenance; it must not be silently marked verified.

The study remains **NOT READY** pending remaining review reconciliation,
current-revision checks, and separate resource/preflight/collection approvals.
No fits, provider calls, commits, pushes, or launch were performed for this
verification.