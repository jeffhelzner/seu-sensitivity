"""Extract omega, kappa draws and PPC/parameter summaries for the report."""
import numpy as np
import pandas as pd
import json
from pathlib import Path

SRC = Path("applications/temperature_study_with_risky_alts/results")
DST = Path("reports/applications/temperature_study_with_risky_alts/data")

temps = ["T0_0", "T0_3", "T0_7", "T1_0", "T1_5"]

# --- Extract omega draws for m_21, kappa+omega for m_31 ---
for temp in temps:
    # m_21: extract omega
    fit_dir = SRC / f"fit_m_21_{temp}"
    csvs = sorted(fit_dir.glob("m_21-*.csv"))
    omega_list = []
    for csv_file in csvs:
        df = pd.read_csv(csv_file, comment="#")
        omega_list.append(df["omega"].values)
    omega = np.concatenate(omega_list)
    np.savez(DST / f"omega_draws_m_21_{temp}.npz", omega=omega)
    print(f"m_21 {temp}: omega {omega.shape} mean={omega.mean():.2f}")

    # m_31: extract kappa and omega
    fit_dir = SRC / f"fit_m_31_{temp}"
    csvs = sorted(fit_dir.glob("m_31-*.csv"))
    kappa_list = []
    omega_list = []
    for csv_file in csvs:
        df = pd.read_csv(csv_file, comment="#")
        kappa_list.append(df["kappa"].values)
        omega_list.append(df["omega"].values)
    kappa = np.concatenate(kappa_list)
    omega = np.concatenate(omega_list)
    np.savez(DST / f"kappa_draws_m_31_{temp}.npz", kappa=kappa)
    np.savez(DST / f"omega_draws_m_31_{temp}.npz", omega=omega)
    print(f"m_31 {temp}: kappa {kappa.shape} mean={kappa.mean():.3f}, omega mean={omega.mean():.2f}")

# --- Extract PPC values ---
ppc_data = {}
for model in ["m_11", "m_21", "m_31"]:
    ppc_data[model] = {}
    for temp in temps:
        csv_path = DST / f"summary_{model}_{temp}.csv"
        df = pd.read_csv(csv_path, index_col=0)
        ppc_vars = [
            "ppc_ll_uncertain", "ppc_modal_uncertain", "ppc_prob_uncertain",
            "ppc_ll_risky", "ppc_modal_risky", "ppc_prob_risky", "ppc_ll_combined",
        ]
        ppc_vals = {}
        for var in ppc_vars:
            if var in df.index:
                ppc_vals[var] = float(df.loc[var, "Mean"])
        ppc_data[model][temp] = ppc_vals

with open(DST / "ppc_summary.json", "w") as f:
    json.dump(ppc_data, f, indent=2)
print("\nPPC summary saved.")

# --- Extract parameter summaries ---
param_data = {}
for model in ["m_11", "m_21", "m_31"]:
    param_data[model] = {}
    for temp in temps:
        csv_path = DST / f"summary_{model}_{temp}.csv"
        df = pd.read_csv(csv_path, index_col=0)
        info = {
            "alpha_mean": float(df.loc["alpha", "Mean"]),
            "alpha_median": float(df.loc["alpha", "50%"]),
            "alpha_sd": float(df.loc["alpha", "StdDev"]),
            "alpha_q05": float(df.loc["alpha", "5%"]),
            "alpha_q95": float(df.loc["alpha", "95%"]),
            "alpha_rhat": float(df.loc["alpha", "R_hat"]),
            "alpha_ess_bulk": float(df.loc["alpha", "ESS_bulk"]),
        }
        if model in ("m_21", "m_31") and "omega" in df.index:
            info["omega_mean"] = float(df.loc["omega", "Mean"])
            info["omega_median"] = float(df.loc["omega", "50%"])
            info["omega_sd"] = float(df.loc["omega", "StdDev"])
            info["omega_q05"] = float(df.loc["omega", "5%"])
            info["omega_q95"] = float(df.loc["omega", "95%"])
        if model == "m_21" and "omega" in df.index:
            info["omega_rhat"] = float(df.loc["omega", "R_hat"])
            info["omega_ess_bulk"] = float(df.loc["omega", "ESS_bulk"])
        if model == "m_31" and "kappa" in df.index:
            info["kappa_mean"] = float(df.loc["kappa", "Mean"])
            info["kappa_median"] = float(df.loc["kappa", "50%"])
            info["kappa_sd"] = float(df.loc["kappa", "StdDev"])
            info["kappa_q05"] = float(df.loc["kappa", "5%"])
            info["kappa_q95"] = float(df.loc["kappa", "95%"])
            info["kappa_rhat"] = float(df.loc["kappa", "R_hat"])
            info["kappa_ess_bulk"] = float(df.loc["kappa", "ESS_bulk"])
        param_data[model][temp] = info

with open(DST / "parameter_summary.json", "w") as f:
    json.dump(param_data, f, indent=2)
print("Parameter summary saved.")
print("\nDone.")
