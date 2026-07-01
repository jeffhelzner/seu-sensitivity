"""
Prior-sensitivity sweep for the 2x2 application (methodological paper, referee
theme 3).

The application reports, per LLM x task cell, a report-level temperature-slope
of alpha (draw-wise population-OLS of the per-condition alpha posteriors on
temperature) together with P(slope < 0).  The referee asks whether these
headline slope findings are an artifact of the calibrated alpha prior.

This spike answers that directly: it REFITS every temperature condition of every
cell under alternative alpha priors while HOLDING THE CHOICE DATA AND FEATURES
FIXED (the committed reports/applications/<cell>/data/stan_data_T*.json files),
then recomputes the report-level slope and P(slope < 0) under each prior with the
SAME draw-wise population-OLS functional the application used.

Baseline priors (== the calibrated models used in the paper):
    insurance cells (K=3):  alpha ~ Lognormal(3.0, 0.75)   (m_01)
    Ellsberg  cells (K=4):  alpha ~ Lognormal(3.5, 0.75)   (m_02)

Alternative priors per cell (bracket the baseline location and widen the scale):
    down :  Lognormal(mu_base - 0.5, 0.75)
    up   :  Lognormal(mu_base + 0.5, 0.75)
    wide :  Lognormal(mu_base,       1.25)

All refits use the data-driven-prior program models/m_0_prior_sweep.stan, which
is structurally identical to m_01/m_02 and only moves the two lognormal
hyperparameters into the data block.  The committed baseline draws are reused for
the baseline row (they were produced by exactly the baseline prior).

Outputs (next to this script / in ../figures):
    report_prior_sensitivity_results.json
    ../figures/report_prior_sensitivity_forest.png
Refitted per-condition draws are cached under
    reports/applications/<cell>/data/prior_sweep/alpha_draws_<prior>_<Tkey>.npz
so reruns are cheap.

Run:
  conda run -n seu-sensitivity python \
    working_papers/seu_sensitivity_methodology/spikes/report_prior_sensitivity_spike.py
"""
from __future__ import annotations

import json
import os

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PAPER_DIR))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "m_0_prior_sweep.stan")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
RESULTS_PATH = os.path.join(THIS_DIR, "report_prior_sensitivity_results.json")

BASE_SIGMA = 0.75

# Per-cell configuration: study dir, temperature grid, temp keys, baseline mu.
CELLS = {
    "gpt4o_insurance": {
        "study": "temperature_study",
        "temps": [0.0, 0.3, 0.7, 1.0, 1.5],
        "keys": ["T0_0", "T0_3", "T0_7", "T1_0", "T1_5"],
        "base_mu": 3.0,
        "ledger": "C9",
        "label": "GPT-4o x Insurance",
    },
    "gpt4o_ellsberg": {
        "study": "gpt4o_ellsberg_study",
        "temps": [0.0, 0.3, 0.7, 1.0, 1.5],
        "keys": ["T0_0", "T0_3", "T0_7", "T1_0", "T1_5"],
        "base_mu": 3.5,
        "ledger": "C12",
        "label": "GPT-4o x Ellsberg",
    },
    "claude_insurance": {
        "study": "claude_insurance_study",
        "temps": [0.0, 0.2, 0.5, 0.8, 1.0],
        "keys": ["T0_0", "T0_2", "T0_5", "T0_8", "T1_0"],
        "base_mu": 3.0,
        "ledger": "C10",
        "label": "Claude x Insurance",
    },
    "claude_ellsberg": {
        "study": "ellsberg_study",
        "temps": [0.0, 0.2, 0.5, 0.8, 1.0],
        "keys": ["T0_0", "T0_2", "T0_5", "T0_8", "T1_0"],
        "base_mu": 3.5,
        "ledger": "C13",
        "label": "Claude x Ellsberg",
    },
}

# Sampler settings for the refits (kept modest; the slope sign and P(slope<0)
# are stable well below the application's production draw count).
CHAINS = 4
WARMUP = 500
SAMPLING = 500
BASE_SEED = 20260701


def temp_key(t: float) -> str:
    return f"T{str(t).replace('.', '_')}"


def prior_grid(base_mu: float) -> dict[str, tuple[float, float]]:
    """Baseline + three alternative alpha priors for one cell."""
    return {
        "baseline": (base_mu, BASE_SIGMA),
        "down": (base_mu - 0.5, BASE_SIGMA),
        "up": (base_mu + 0.5, BASE_SIGMA),
        "wide": (base_mu, 1.25),
    }


def data_dir(study: str) -> str:
    return os.path.join(PROJECT_ROOT, "reports", "applications", study, "data")


def sweep_dir(study: str) -> str:
    d = os.path.join(data_dir(study), "prior_sweep")
    os.makedirs(d, exist_ok=True)
    return d


def slope_weights(temps: np.ndarray) -> np.ndarray:
    """Population-OLS weights w_t = (T_t - Tbar) / sum_j (T_j - Tbar)^2."""
    tc = temps - temps.mean()
    return tc / np.sum(tc**2)


def slope_draws_from(cols: list[np.ndarray], temps: np.ndarray) -> np.ndarray:
    """Draw-wise population-OLS slope = sum_t w_t * alpha_t.

    Columns are truncated to the common minimum draw count so the linear
    functional is applied draw-for-draw within a single prior.
    """
    n = min(c.shape[0] for c in cols)
    A = np.column_stack([c[:n] for c in cols])  # (n, 5)
    return A @ slope_weights(temps)


def load_committed_baseline(study: str, keys: list[str]) -> list[np.ndarray]:
    cols = []
    for k in keys:
        with np.load(os.path.join(data_dir(study), f"alpha_draws_{k}.npz")) as z:
            cols.append(np.asarray(z["alpha"], dtype=float))
    return cols


def refit_condition(model, stan_data_path: str, mu: float, sigma: float,
                    seed: int) -> np.ndarray:
    """Refit one temperature condition under a given alpha prior; return draws."""
    with open(stan_data_path) as f:
        stan_data = json.load(f)
    stan_data["alpha_prior_mu"] = float(mu)
    stan_data["alpha_prior_sigma"] = float(sigma)
    fit = model.sample(
        data=stan_data,
        chains=CHAINS,
        iter_warmup=WARMUP,
        iter_sampling=SAMPLING,
        seed=seed,
        show_progress=False,
    )
    return np.asarray(fit.stan_variable("alpha"), dtype=float)


def refit_cell(model, cell_key: str, cfg: dict, prior_name: str,
               mu: float, sigma: float) -> list[np.ndarray]:
    """Refit all temperatures of one cell under one prior (with caching)."""
    study = cfg["study"]
    keys = cfg["keys"]
    out = []
    for i, k in enumerate(keys):
        cache = os.path.join(sweep_dir(study), f"alpha_draws_{prior_name}_{k}.npz")
        if os.path.exists(cache):
            with np.load(cache) as z:
                out.append(np.asarray(z["alpha"], dtype=float))
            continue
        stan_data_path = os.path.join(data_dir(study), f"stan_data_{k}.json")
        seed = BASE_SEED + 1000 * list(CELLS).index(cell_key) + 10 * i
        draws = refit_condition(model, stan_data_path, mu, sigma, seed)
        np.savez_compressed(cache, alpha=draws)
        out.append(draws)
        print(f"    [{cell_key}/{prior_name}/{k}] refit: "
              f"n={draws.shape[0]}, median alpha={np.median(draws):.2f}")
    return out


def summarize(cols: list[np.ndarray], temps: np.ndarray) -> dict:
    per_cond_median = [float(np.median(c)) for c in cols]
    sl = slope_draws_from(cols, temps)
    return {
        "per_condition_alpha_median": per_cond_median,
        "slope_median": float(np.median(sl)),
        "slope_mean": float(np.mean(sl)),
        "slope_sd": float(np.std(sl, ddof=1)),
        "slope_ci_low": float(np.percentile(sl, 5)),
        "slope_ci_high": float(np.percentile(sl, 95)),
        "p_slope_negative": float(np.mean(sl < 0)),
        "n_draws": int(min(c.shape[0] for c in cols)),
    }


def main() -> None:
    from cmdstanpy import CmdStanModel

    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Compiling {os.path.relpath(MODEL_PATH, PROJECT_ROOT)} ...")
    model = CmdStanModel(stan_file=MODEL_PATH)

    results = {"model": os.path.relpath(MODEL_PATH, PROJECT_ROOT),
               "base_sigma": BASE_SIGMA,
               "sampler": {"chains": CHAINS, "warmup": WARMUP,
                           "sampling": SAMPLING, "base_seed": BASE_SEED},
               "cells": {}}

    for cell_key, cfg in CELLS.items():
        temps = np.asarray(cfg["temps"], dtype=float)
        grid = prior_grid(cfg["base_mu"])
        print(f"\n=== {cfg['label']} ({cell_key}), baseline mu={cfg['base_mu']} ===")
        cell_res = {"label": cfg["label"], "ledger": cfg["ledger"],
                    "temps": cfg["temps"], "base_mu": cfg["base_mu"],
                    "priors": {}}
        for prior_name, (mu, sigma) in grid.items():
            if prior_name == "baseline":
                cols = load_committed_baseline(cfg["study"], cfg["keys"])
                source = "committed"
            else:
                cols = refit_cell(model, cell_key, cfg, prior_name, mu, sigma)
                source = "refit"
            summ = summarize(cols, temps)
            summ.update({"mu": mu, "sigma": sigma, "source": source})
            cell_res["priors"][prior_name] = summ
            print(f"  {prior_name:8s} Lognormal({mu:.2f},{sigma:.2f}) [{source:9s}]"
                  f"  slope median={summ['slope_median']:+8.2f}"
                  f"  P(slope<0)={summ['p_slope_negative']:.3f}")
        # Stability summary across priors.
        signs = {p: np.sign(r["slope_median"]) for p, r in cell_res["priors"].items()}
        pnegs = [r["p_slope_negative"] for r in cell_res["priors"].values()]
        cell_res["slope_sign_stable"] = bool(len(set(signs.values())) == 1)
        cell_res["p_slope_negative_range"] = [float(min(pnegs)), float(max(pnegs))]
        results["cells"][cell_key] = cell_res

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {os.path.relpath(RESULTS_PATH, PROJECT_ROOT)}")

    make_forest(results)


def make_forest(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = results["cells"]
    prior_order = ["baseline", "down", "up", "wide"]
    prior_color = {"baseline": "#000000", "down": "#1b7837",
                   "up": "#762a83", "wide": "#2166ac"}
    prior_label = {
        "baseline": "baseline",
        "down": "mu - 0.5",
        "up": "mu + 0.5",
        "wide": "sigma = 1.25",
    }

    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 4.0),
                             sharex=False)
    if len(cells) == 1:
        axes = [axes]

    for ax, (ck, cr) in zip(axes, cells.items()):
        ys = list(range(len(prior_order)))[::-1]
        for y, p in zip(ys, prior_order):
            r = cr["priors"][p]
            ax.plot([r["slope_ci_low"], r["slope_ci_high"]], [y, y],
                    color=prior_color[p], lw=2)
            ax.plot(r["slope_median"], y, "o", color=prior_color[p], ms=6,
                    label=f"{prior_label[p]}: P(<0)={r['p_slope_negative']:.2f}")
        ax.axvline(0.0, color="grey", ls="--", lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([prior_label[p] for p in prior_order])
        ax.set_title(cr["label"], fontsize=10)
        ax.set_xlabel(r"report-level slope  $d\alpha/dT$")
        ax.legend(fontsize=6, loc="best", frameon=False)
    fig.suptitle("Prior sensitivity of the temperature-slope (90% posterior "
                 "intervals; P(slope<0) in legend)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG_DIR, "report_prior_sensitivity_forest.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {os.path.relpath(out, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
