# Phase E1: End-to-End Smoke Validation

Date: 2026-09-06

## Scope

The smoke covered venture and hiring, all 18 model-by-prompt cells per pool,
12 menus per pool, and two reversed presentations. The menus were balanced over
sizes 2, 4, 6, and 8 and three difficulty strata. This produced 432 corrected
observations per pool and 864 total.

An initial answer-last collection was stopped after 144 calls because the
64-token output cap truncated responses before the structured answer. Those
responses are quarantined and excluded. The corrected prompts require
`ANSWER: n` first. All 864 corrected responses resolved through that token,
with no NA or fallback observations. One Anthropic HTTP 529 was retried
successfully.

## Collection and position stability

| Pool | Observations | NA/fallback | Reversal agreement | Implied rho | Stable menus retained |
|---|---:|---:|---:|---:|---:|
| Venture | 432 | 0 (0%) | 193/216 (0.894) | 0.793 | 4/12 (33.3%) |
| Hiring | 432 | 0 (0%) | 183/216 (0.847) | 0.703 | 3/12 (25.0%) |

The implied copy probability uses
`agreement = rho + (1 - rho) * 0.486`. Stable-menu retention is the balanced
subset available after applying the preregistered position-stability rule, not
the fraction of individual presentation pairs that agree. The low menu-level
retention makes this subset unsuitable as the sole primary analysis sample.

## Pinned-model fits

Each pool used `models/h_m01_size_pinned.stan`, four chains, 1,000 warmup draws,
800 retained draws per chain, `adapt_delta=0.95`, and `max_treedepth=12`.

| Pool | Hours | Mean/max depth | Depth-12 saturation | Divergences | Min bulk ESS | Max R-hat |
|---|---:|---:|---:|---:|---:|---:|
| Venture | 1.270 | 8.75 / 10 | 0 | 0 | 1,292 | 1.0098 |
| Hiring | 0.332 | 7.00 / 7 | 0 | 0 | 1,229 | 1.0081 |

Both fits pass the convergence and geometry checks. The occasional invalid-
simplex messages were rejected proposals during initialization or sampling;
they did not produce invalid retained draws.

## Posterior evidence

Intervals below are central 90% intervals.

| Pool | sigma_cell mean [90% interval] | gamma_size mean [90% interval] | Model-dummy posterior SD range | Prompt-effect posterior SD range |
|---|---|---|---:|---:|
| Venture | 0.176 [0.013, 0.441] | 0.309 [0.166, 0.452] | 0.334--0.410 | 0.324--0.335 |
| Hiring | 0.164 [0.011, 0.409] | 0.019 [-0.088, 0.129] | 0.307--0.372 | 0.292--0.297 |

Both `sigma_cell` means are below the prior mean of 0.239, but their smoke
intervals remain wide. Model-dummy uncertainty of approximately 0.31--0.41 is
also too large to rehabilitate the preregistered cross-pool ordering-agreement
statistic. Phase E3 should retain the variance component as the primary RQ4
operationalization and keep ordering agreement descriptive.

The venture size effect is clearly positive in this smoke; the hiring size
effect is centered near zero. These are feasibility measurements from a small,
purpose-built sample, not confirmatory substantive results.

## Cost evidence

The corrected 864-call collection cost **$4.143126** at the standard synchronous
API rates. This aggregate is measured. The later offline `stan_data` phase
overwrote the phase-local `run_summary.json`, so exact per-arm token rows are no
longer recoverable from the cached response files; those files contain response
text but not usage metadata. Per-arm allocations in the Phase E2 forecast are
therefore estimates and are labeled as such.

The quarantined 144 calls are excluded from both the analytic sample and the
corrected-cost figure. Corrected plus quarantined spending remained below the
approved $6 E1 ceiling.

## Verdict

E1 passes the collection, parsing, model-geometry, and 12-hour runtime gates.
It supports advancing to an E3 design decision after reviewing the Phase E2
forecast. It does not support using the stability subset alone, reducing the
planned primary-family menu count, or making ordering agreement the primary RQ4
estimand.

## Evidence

- Corrected collection: `applications/seu_sensitivity_study/results/e1_smoke/`
- Quarantine: `applications/seu_sensitivity_study/results/e1_smoke/quarantine_answer_last/`
- Venture fit: `results/power/e1_venture_pinned/`
- Hiring fit: `results/power/e1_hiring_pinned/`
- Recovery evidence: `applications/seu_sensitivity_study/PHASE_G_VALIDATION.md`
