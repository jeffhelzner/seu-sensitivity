"""
Matched-design parameter recovery: m_0 vs m_1 (Report 14)

Tests whether m_1's likelihood structure (adding risky choices with known
probabilities) actually delivers the delta-identification improvement that
the theoretical argument in Report 5 predicts.

For each iteration:
  1. Generate one set of true (alpha, beta, delta) via m_1_sim
     on a shared M=50 + N=50 design.
  2. Simulate uncertain choices y[1..50] and risky choices z[1..50] from
     those true parameters.
  3. Fit four conditions on slices of (y, z):
       A: m_0 on M=25 uncertain   (the existing m_0 baseline)
       B: m_0 on M=50 uncertain   (data-quantity control)
       C: m_1 on M=25 uncertain + N=25 risky  (== B in total choice count)
       D: m_1 on M=50 uncertain + N=50 risky  (replicates existing m_1)

The key contrast is B vs C: same total choice count, same true parameters
per iteration, only the model + data type differ.  If m_1's likelihood
delivers identification value, C beats B on delta recovery.

Usage:
    python scripts/run_m1_matched_recovery.py \
        --config configs/m1_matched_recovery_config.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.study_design_m1 import StudyDesignM1


CONDITIONS = [
    {"label": "A_m0_M25",   "model": "m_0", "M": 25, "N": 0,  "seed_offset": 100},
    {"label": "B_m0_M50",   "model": "m_0", "M": 50, "N": 0,  "seed_offset": 200},
    {"label": "C_m1_M25N25","model": "m_1", "M": 25, "N": 25, "seed_offset": 300},
    {"label": "D_m1_M50N50","model": "m_1", "M": 50, "N": 50, "seed_offset": 400},
]


def build_inference_data(cond, base, y_full, z_full):
    """Slice the shared design + simulated choices to match a condition."""
    data = {
        "K": base["K"],
        "D": base["D"],
        "R": base["R"],
        "w": base["w"],
    }
    M = cond["M"]
    data["M"] = M
    data["I"] = base["I"][:M]
    data["y"] = y_full[:M].tolist()
    if cond["model"] == "m_1":
        N = cond["N"]
        data["S"] = base["S"]
        data["x"] = base["x"]
        data["N"] = N
        data["J"] = base["J"][:N]
        data["z"] = z_full[:N].tolist()
    return data


def fit_and_summarize(model, data, seed, n_samples, n_chains):
    fit = model.sample(
        data=data,
        seed=seed,
        iter_sampling=n_samples,
        iter_warmup=n_samples // 2,
        chains=n_chains,
        show_console=False,
    )
    return fit.summary()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    dc = cfg["study_design_config"]
    if dc["M"] != 50 or dc["N"] != 50:
        raise SystemExit(
            "This driver expects M=50 and N=50 in the shared design so "
            "all four conditions can be cleanly sliced from one simulation."
        )

    print("Building shared m_1 study design (M=50 + N=50)...")
    study = StudyDesignM1(
        M=dc["M"], N=dc["N"], K=dc["K"], D=dc["D"], R=dc["R"], S=dc["S"],
        min_alts_per_problem=dc.get("min_alts_per_problem", 2),
        max_alts_per_problem=dc.get("max_alts_per_problem", 5),
        risky_probs=dc.get("risky_probs", "fixed"),
        feature_dist=dc.get("feature_dist", "normal"),
        feature_params=dc.get("feature_params", {"loc": 0, "scale": 1}),
    )
    study.generate()

    base = study.get_data_dict()
    base["I"] = np.array(base["I"]).tolist()
    base["J"] = np.array(base["J"]).tolist()
    study.save(os.path.join(output_dir, "study_design.json"))

    print("Compiling Stan models...")
    sim_model = CmdStanModel(stan_file=os.path.join(project_root, "models", "m_1_sim.stan"))
    inf_models = {
        "m_0": CmdStanModel(stan_file=os.path.join(project_root, "models", "m_0.stan")),
        "m_1": CmdStanModel(stan_file=os.path.join(project_root, "models", "m_1.stan")),
    }

    sim_data = dict(base)
    # m_1_sim hyperparams (same defaults the recovery sweep uses)
    sim_data.update({"alpha_mean": 0.0, "alpha_sd": 1.0, "beta_sd": 1.0})

    n_iter = cfg.get("n_iterations", 25)
    n_samples = cfg.get("n_mcmc_samples", 2000)
    n_chains = cfg.get("n_mcmc_chains", 4)

    all_true_params = []
    print(f"\nRunning {n_iter} matched-design iterations across {len(CONDITIONS)} conditions...")
    for it in tqdm(range(n_iter)):
        iter_dir = os.path.join(output_dir, f"iteration_{it+1}")
        os.makedirs(iter_dir, exist_ok=True)

        sim_fit = sim_model.sample(
            data=sim_data,
            seed=12345 + it,
            iter_sampling=1, iter_warmup=0, chains=1,
            fixed_param=True, adapt_engaged=False,
        )
        sim_samples = sim_fit.draws_pd().iloc[0]
        y_full = np.array([int(sim_samples[f"y[{i+1}]"]) for i in range(dc["M"])])
        z_full = np.array([int(sim_samples[f"z[{i+1}]"]) for i in range(dc["N"])])

        K = dc["K"]; D = dc["D"]
        true_params = {
            "alpha": float(sim_samples["alpha"]),
            "beta":  [[float(sim_samples[f"beta[{k+1},{d+1}]"]) for d in range(D)] for k in range(K)],
            "delta": [float(sim_samples[f"delta[{k+1}]"]) for k in range(K - 1)],
        }
        with open(os.path.join(iter_dir, "true_parameters.json"), "w") as f:
            json.dump(true_params, f, indent=2)
        all_true_params.append(true_params)

        for cond in CONDITIONS:
            try:
                data = build_inference_data(cond, base, y_full, z_full)
                summary = fit_and_summarize(
                    inf_models[cond["model"]], data,
                    seed=54321 + it * 17 + cond["seed_offset"],
                    n_samples=n_samples, n_chains=n_chains,
                )
                summary.to_csv(os.path.join(iter_dir, f"summary_{cond['label']}.csv"))
            except Exception as e:
                with open(os.path.join(iter_dir, f"error_{cond['label']}.txt"), "w") as f:
                    f.write(repr(e))

    with open(os.path.join(output_dir, "all_true_parameters.json"), "w") as f:
        json.dump(all_true_params, f, indent=2)
    with open(os.path.join(output_dir, "conditions.json"), "w") as f:
        json.dump(CONDITIONS, f, indent=2)

    print(f"\nDone.  Per-condition summaries under {output_dir}/iteration_*/summary_*.csv")


if __name__ == "__main__":
    main()
