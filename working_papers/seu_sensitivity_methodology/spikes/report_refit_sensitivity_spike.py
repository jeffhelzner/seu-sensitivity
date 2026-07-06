"""
Sampler-settings sensitivity refits for the 20 application fits.

Motivated by the 2026-07-05 review (overall theme C): the application fits
report divergent transitions in 12/20 cells (34 total, <= 0.15% of draws) and
R-hat exceedances confined to weakly informed nuisance components. Because
alpha is a marginal of a joint posterior explored by NUTS, low-rate divergences
could in principle bias the alpha marginal even when its own diagnostics look
clean. This spike re-runs ALL 20 application fits (4 LLM x task cells x 5
temperatures) on the committed Stan data under stricter sampler settings:

    adapt_delta   0.8  -> 0.99
    max_treedepth 10   -> 12
    iter_warmup   1000 -> 2000
    (chains 4, iter_sampling 1000, seed 42: unchanged from D.2/D.5)

and records, per fit:
  - divergence and treedepth-hit counts under the stricter settings,
  - the refit alpha posterior (median, 90% CI, sd of log alpha),
  - the committed alpha posterior for comparison,
  - the shift in alpha median as a fraction of the committed posterior sd,
  - per-menu eta-gap posterior summaries (pooled median / 5-95%), the
    posterior counterpart of the design-level prior eta-gap diagnostic in
    report_design_diagnostics_spike.py.

Per-cell temperature-slope posteriors (canonical draw-level population OLS,
as in report_2x2_forest_spike.py) are recomputed from the refit alpha draws
and compared with the committed-slope posteriors.

Outputs:
  report_refit_sensitivity_results.json    (next to this script)

Run (long-running; ~1-3 h):
  conda run -n seu-sensitivity python \
    working_papers/seu_sensitivity_methodology/spikes/report_refit_sensitivity_spike.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))
DATA_BASE = os.path.join(PROJECT_ROOT, "reports", "applications")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

CHAINS = 4
ITER_WARMUP = 2000
ITER_SAMPLING = 1000
ADAPT_DELTA = 0.99
MAX_TREEDEPTH = 12
SEED = 42

STUDIES = {
    "temperature_study": {
        "cell": "GPT-4o x Insurance", "model": "m_01",
        "temps": [0.0, 0.3, 0.7, 1.0, 1.5],
        "keys": ["T0_0", "T0_3", "T0_7", "T1_0", "T1_5"],
    },
    "claude_insurance_study": {
        "cell": "Claude 3.5 x Insurance", "model": "m_01",
        "temps": [0.0, 0.2, 0.5, 0.8, 1.0],
        "keys": ["T0_0", "T0_2", "T0_5", "T0_8", "T1_0"],
    },
    "gpt4o_ellsberg_study": {
        "cell": "GPT-4o x Ellsberg", "model": "m_02",
        "temps": [0.0, 0.3, 0.7, 1.0, 1.5],
        "keys": ["T0_0", "T0_3", "T0_7", "T1_0", "T1_5"],
    },
    "ellsberg_study": {
        "cell": "Claude 3.5 x Ellsberg", "model": "m_02",
        "temps": [0.0, 0.2, 0.5, 0.8, 1.0],
        "keys": ["T0_0", "T0_2", "T0_5", "T0_8", "T1_0"],
    },
}


def alpha_summary(a: np.ndarray) -> dict:
    return {
        "median": float(np.median(a)),
        "ci_low": float(np.quantile(a, 0.05)),
        "ci_high": float(np.quantile(a, 0.95)),
        "sd": float(np.std(a)),
        "sd_log": float(np.std(np.log(a))),
        "n_draws": int(a.shape[0]),
    }


def slope_posterior(alpha_cols: list[np.ndarray], temps: list[float]) -> dict:
    t = np.asarray(temps, dtype=float)
    tc = t - t.mean()
    w = tc / np.sum(tc**2)
    n = min(c.shape[0] for c in alpha_cols)
    s = np.column_stack([c[:n] for c in alpha_cols]) @ w
    return {
        "median": float(np.median(s)),
        "ci_low": float(np.quantile(s, 0.05)),
        "ci_high": float(np.quantile(s, 0.95)),
        "p_neg": float(np.mean(s < 0.0)),
    }


def eta_gap_summary(fit, stan_data: dict) -> dict:
    """Pooled posterior distribution of within-menu eta gaps (max - min)."""
    eta = fit.stan_variable("eta")            # (draws, sum(N))
    I = np.asarray(stan_data["I"], dtype=int)
    N = I.sum(axis=1)
    gaps = []
    pos = 0
    for n_m in N:
        seg = eta[:, pos:pos + n_m]
        gaps.append(seg.max(axis=1) - seg.min(axis=1))
        pos += n_m
    pooled = np.concatenate(gaps)
    return {
        "pooled_gap_median": float(np.median(pooled)),
        "pooled_gap_q05": float(np.quantile(pooled, 0.05)),
        "pooled_gap_q95": float(np.quantile(pooled, 0.95)),
    }


def main() -> None:
    from cmdstanpy import CmdStanModel

    models = {
        name: CmdStanModel(stan_file=os.path.join(MODELS_DIR, f"{name}.stan"))
        for name in ("m_01", "m_02")
    }

    results = {
        "spec": {
            "review_item": "overall theme C (HMC pathologies in nuisance blocks)",
            "sampler": {
                "chains": CHAINS, "iter_warmup": ITER_WARMUP,
                "iter_sampling": ITER_SAMPLING, "adapt_delta": ADAPT_DELTA,
                "max_treedepth": MAX_TREEDEPTH, "seed": SEED,
            },
            "baseline_sampler": {
                "chains": 4, "iter_warmup": 1000, "iter_sampling": 1000,
                "adapt_delta": 0.8, "max_treedepth": 10, "seed": 42,
            },
        },
        "fits": [],
        "slopes": {},
    }

    for study, spec in STUDIES.items():
        refit_alpha_cols = []
        committed_alpha_cols = []
        for key in spec["keys"]:
            data_path = os.path.join(DATA_BASE, study, "data",
                                     f"stan_data_{key}.json")
            with open(data_path) as f:
                stan_data = json.load(f)

            print(f"[refit] {spec['cell']} {key} ...", flush=True)
            fit = models[spec["model"]].sample(
                data=data_path,
                chains=CHAINS,
                iter_warmup=ITER_WARMUP,
                iter_sampling=ITER_SAMPLING,
                adapt_delta=ADAPT_DELTA,
                max_treedepth=MAX_TREEDEPTH,
                seed=SEED,
                show_progress=False,
            )

            mv = fit.method_variables()
            n_div = int(np.sum(mv["divergent__"]))
            td = np.asarray(mv["treedepth__"])
            n_td_hits = int(np.sum(td >= MAX_TREEDEPTH))

            a_refit = np.asarray(fit.stan_variable("alpha"), dtype=float)
            with np.load(os.path.join(DATA_BASE, study, "data",
                                      f"alpha_draws_{key}.npz")) as z:
                a_comm = np.asarray(z["alpha"], dtype=float)

            refit_alpha_cols.append(a_refit)
            committed_alpha_cols.append(a_comm)

            s_refit = alpha_summary(a_refit)
            s_comm = alpha_summary(a_comm)
            shift = ((s_refit["median"] - s_comm["median"]) /
                     s_comm["sd"] if s_comm["sd"] > 0 else float("nan"))

            row = {
                "study": study, "cell": spec["cell"], "condition": key,
                "n_divergences": n_div,
                "n_treedepth_hits": n_td_hits,
                "alpha_refit": s_refit,
                "alpha_committed": s_comm,
                "alpha_median_shift_in_committed_sd": round(float(shift), 4),
                "eta_gaps_posterior": eta_gap_summary(fit, stan_data),
            }
            results["fits"].append(row)
            print(f"        divergences {n_div}, treedepth hits {n_td_hits}, "
                  f"alpha median {s_comm['median']:.1f} -> "
                  f"{s_refit['median']:.1f} "
                  f"({shift:+.3f} committed-posterior sd)", flush=True)

            # Incremental checkpoint so partial results survive interruption.
            out_path = os.path.join(
                THIS_DIR, "report_refit_sensitivity_results.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

        results["slopes"][study] = {
            "cell": spec["cell"],
            "temps": spec["temps"],
            "refit": slope_posterior(refit_alpha_cols, spec["temps"]),
            "committed": slope_posterior(committed_alpha_cols, spec["temps"]),
        }

    out_path = os.path.join(THIS_DIR, "report_refit_sensitivity_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    total_div = sum(r["n_divergences"] for r in results["fits"])
    max_shift = max(abs(r["alpha_median_shift_in_committed_sd"])
                    for r in results["fits"])
    print("\n=== Refit sensitivity summary ===")
    print(f"total divergences under stricter settings: {total_div} "
          f"(baseline: 34 across 12/20 fits)")
    print(f"max |alpha median shift| (in committed posterior sd): "
          f"{max_shift:.3f}")
    for study, s in results["slopes"].items():
        print(f"{s['cell']:24s} slope committed {s['committed']['median']:7.2f} "
              f"-> refit {s['refit']['median']:7.2f}  "
              f"P(<0) {s['committed']['p_neg']:.3f} -> {s['refit']['p_neg']:.3f}")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
