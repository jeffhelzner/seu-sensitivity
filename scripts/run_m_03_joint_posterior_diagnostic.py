"""
beta-delta joint posterior diagnostic for Report 13.

For each concentration value alpha0 in {1, 5, 10}, run ONE sim + ONE inference
fit using m_03_sim / m_03 with the SAME study design used by the recovery
sweep, and save full draws for beta[1,1] and delta[1] (plus a few neighbours)
so Report 13 can visualize the within-posterior coupling.

Why a separate script: ParameterRecovery only saves marginal posterior summaries
per iteration, not full draws.  Refitting one iteration per alpha0 is cheap
(~1-2 min total) and gives us the diagnostic we need without bloating the
recovery output.

Run after the recovery sweep finishes:
    python scripts/run_m_03_joint_posterior_diagnostic.py
"""
import os
import sys
import json

import numpy as np
from cmdstanpy import CmdStanModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.study_design import StudyDesign


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_BASE = os.path.join(PROJECT_ROOT, "results", "parameter_recovery", "m_03_concentration_sweep")
DESIGN_PATH = os.path.join(SWEEP_BASE, "study_design.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "parameter_recovery", "m_03_concentration_sweep", "joint_posterior")

ALPHA0_GRID = [1.0, 5.0, 10.0]
SEED_OFFSET = 999_000  # distinct from the sweep's seeds (12345+i / 54321+i)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DESIGN_PATH):
        raise SystemExit(
            f"Study design not found at {DESIGN_PATH}.\n"
            "Run scripts/run_m_03_concentration_sweep.py first."
        )

    study = StudyDesign.load(DESIGN_PATH)
    sim_data = study.get_data_dict()
    sim_data.update({"alpha_mean": 0.0, "alpha_sd": 1.0, "beta_sd": 1.0})

    sim_model = CmdStanModel(stan_file=os.path.join(PROJECT_ROOT, "models", "m_03_sim.stan"))
    inf_model = CmdStanModel(stan_file=os.path.join(PROJECT_ROOT, "models", "m_03.stan"))

    summary_rows = []

    for alpha0 in ALPHA0_GRID:
        print(f"\n=== alpha0 = {alpha0} ===")
        sim_data_a = dict(sim_data)
        sim_data_a["delta_concentration"] = alpha0

        sim_fit = sim_model.sample(
            data=sim_data_a,
            seed=SEED_OFFSET + int(alpha0 * 10),
            iter_sampling=1, iter_warmup=0, chains=1,
            fixed_param=True, adapt_engaged=False,
        )
        sim_samples = sim_fit.draws_pd().iloc[0]
        y = np.array([int(sim_samples[f'y[{i+1}]']) for i in range(study.M)])

        true_alpha = float(sim_samples["alpha"])
        true_beta11 = float(sim_samples["beta[1,1]"])
        true_delta1 = float(sim_samples["delta[1]"])
        true_delta2 = float(sim_samples["delta[2]"]) if study.K >= 3 else None

        inf_data = dict(sim_data)
        inf_data.pop("alpha_mean", None); inf_data.pop("alpha_sd", None); inf_data.pop("beta_sd", None)
        inf_data["delta_concentration"] = alpha0
        inf_data["y"] = y.tolist()

        inf_fit = inf_model.sample(
            data=inf_data,
            seed=SEED_OFFSET + int(alpha0 * 10) + 1,
            iter_sampling=1000, iter_warmup=500, chains=4,
            show_console=False,
        )

        draws = inf_fit.draws_pd()
        beta_cols = [c for c in draws.columns if c.startswith("beta[")]
        delta_cols = [c for c in draws.columns if c.startswith("delta[")]
        alpha_col = ["alpha"]

        keep_cols = alpha_col + beta_cols + delta_cols
        out_npz = os.path.join(OUTPUT_DIR, f"alpha0={alpha0:g}_draws.npz")
        np.savez(out_npz,
                 columns=np.array(keep_cols, dtype=object),
                 draws=draws[keep_cols].to_numpy(),
                 true_alpha=true_alpha, true_beta11=true_beta11,
                 true_delta1=true_delta1)
        print(f"  saved {len(draws)} draws to {out_npz}")

        # Compute summary statistic: within-posterior Pearson corr(beta[1,1], delta[1])
        b11 = draws["beta[1,1]"].to_numpy()
        d1 = draws["delta[1]"].to_numpy()
        corr = float(np.corrcoef(b11, d1)[0, 1])
        summary_rows.append({
            "alpha0": alpha0,
            "n_draws": int(len(draws)),
            "corr_beta11_delta1": corr,
            "true_beta11": true_beta11,
            "true_delta1": true_delta1,
            "posterior_mean_beta11": float(b11.mean()),
            "posterior_mean_delta1": float(d1.mean()),
        })
        print(f"  corr(beta[1,1], delta[1]) = {corr:+.3f}")

    out_summary = os.path.join(OUTPUT_DIR, "joint_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nWrote summary to {out_summary}")


if __name__ == "__main__":
    main()
