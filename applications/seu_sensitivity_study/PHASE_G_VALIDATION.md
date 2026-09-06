# Phase G: J=18 Re-validation

Date: 2026-09-06

## Scope and stopping rule

The fixed equal-spacing model was evaluated at J=18, 30 menus per cell, two
presentations, `rho_copy=0.9`, and `max_treedepth=12`. The first recovery
replicate measured cost with 800 sampling draws; two further independent
replicates used 1,600 draws. All used 1,000 warmup draws and four chains.

The planned target was 4--6 recovery replicates, conditional on the 12-hour
per-launch compute cap. Replicate 3 took 8 h 21 m, 4.17 times as long as
replicate 2 and longer than its conservative 7 h 12 m bound. That observed
runtime must be treated as a lower bound for another independently simulated
dataset. Replicate 4 was therefore not launched. The resulting n=3 evidence is
a sanity check, not a precise coverage or power estimate.

## Per-replicate results

| Replicate | Seed | Sampling draws | Hours | Max R-hat | Min bulk ESS | Mean depth | Max depth | Divergences | gamma_size covered | CI width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12345 | 800 | 1.22 | 1.0172 | 301 | 8.32 | 9 | 0 | yes | 0.243 |
| 2 | 12346 | 1,600 | 2.00 | 1.0054 | 1,048 | 8.53 | 9 | 0 | yes | 0.135 |
| 3 | 12347 | 1,600 | 8.34 | 1.0049 | 579 | 10.88 | 11 | 2 | yes | 0.407 |

Replicate 1 narrowly misses the approximately 1.01 all-parameter R-hat rule;
replicates 2 and 3 pass it. Replicate 3 has two divergences among 6,400 retained
draws and no treedepth saturation. The non-fatal invalid-simplex messages in all
three logs are rejected proposals during initialization or sampling, not invalid
retained draws.

## G1: Recovery sanity check

Across the three independent datasets:

- `gamma_size` has the correct posterior-mean sign in 3/3 replicates;
- all three 90% intervals cover the generating value;
- there are no type-S errors;
- mean bias is -0.037 and RMSE is 0.100;
- mean 90% interval width is 0.262;
- mean presentation-agreement rate is 0.950; and
- mean balanced stability-subset retention is approximately 0.852.

Three successes do not establish nominal 90% coverage: the exact two-sided 90%
binomial interval for 3/3 is `[0.368, 1.000]`. These results support a limited
sanity verdict that 30 menus per cell is not visibly broken at J=18. Combined
with the earlier J=6 result that coverage recovered by 60 menus per cell, they
do not justify reducing the planned primary-family count of 100 problems.

## G2: Model-effect sampling uncertainty

For each replicate, sampling error is the posterior mean minus the generating
value for each `gamma` coefficient. The pooled RMS sampling SE over the five
model-dummy coefficients is **0.421** at n=3. Its leave-one-replicate-out values
are **0.299, 0.428, and 0.508**, so 0.421 is an unstable planning estimate rather
than a precise constant.

Even the low end of that range exceeds the previous broken-geometry estimate of
0.31 and the prior-mean analytic floor of 0.195. More recovery iterations might
move the estimate, but the present evidence does not support treating the
pre-registered cross-pool ordering-agreement statistic as adequately resolved.
The variance-component formulation remains the recommended inferential
operationalization for RQ4, with ordering agreement retained descriptively. The
real-data `sigma_cell` estimate from Phase E1 is still required to update the
analytic floor.

## G3: SBC decision

The original Phase C SBC at J=6 used 100 simulations and showed mild mean-rank
drift in `gamma0`, `gamma_size`, and `sigma_cell`. The fixed model removes the
J=18 ridge, but the existing SBC result does not validate the fixed model.

Two defensible options remain:

1. **Reduced pinned-model SBC before preregistration.** Port the SBC model to
   fixed equal spacing and run 100 J=6 simulations. The old-model configuration
   estimated about 5 hours for 100 simulations; this is an unmeasured lower
   bound for the new variant, so one timing batch must precede the run. A
   100-simulation rerun is diagnostic and remains at the lower bound of useful
   SBC resolution. A better-powered 400-simulation campaign is approximately
   20--36 hours and must be split across multiple launches to respect the
   12-hour cap.

2. **Proceed without a new SBC.** Use the same-data Phase F convergence result,
   the present J=18 recovery sanity check, and the existing null-calibration
   evidence. Explicitly document that marginal SBC was performed only for the
   original free-utility model and showed a mild drift in parameters later
   implicated in the ridge; therefore it is not inherited as calibration proof
   for the fixed model.

Given the large compute cost, the low resolution of a 100-simulation rerun, and
the clean direct recovery and convergence evidence for the fixed model, the
recommended option was **2: proceed with recovery plus null-calibration evidence
and document the unresolved SBC drift**.

**Decision, 2026-09-06:** the user approved option 2. No pinned-model SBC rerun
will be launched before preregistration. The preregistration must state the
scope of the existing calibration evidence and must not imply that the original
free-utility model's SBC result transfers to the fixed model.

## Evidence

- Replicate 1: `results/power/h_m01_size_pinned_g1_measure/`
- Replicate 2: `results/power/h_m01_size_pinned_g1_recovery_002/`
- Replicate 3: `results/power/h_m01_size_pinned_g1_recovery_003/`
- Phase F: `applications/seu_sensitivity_study/PHASE_F_VALIDATION.md`
- Original SBC config: `configs/h_m01_size_sbc_config.json`