#!/usr/bin/env python3
"""Generate primary_analysis.json for the Ellsberg study report."""

import numpy as np
import json
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "reports" / "applications" / "ellsberg_study" / "data"
temperatures = [0.0, 0.2, 0.5, 0.8, 1.0]

# Load alpha draws
alpha_draws = {}
for t in temperatures:
    key = f"T{str(t).replace('.', '_')}"
    data = np.load(data_dir / f"alpha_draws_{key}.npz")
    alpha_draws[t] = data["alpha"]

# Summary table
summary_table = []
for t in temperatures:
    draws = alpha_draws[t]
    summary_table.append({
        "temperature": t,
        "median": float(np.median(draws)),
        "mean": float(np.mean(draws)),
        "sd": float(np.std(draws)),
        "ci_low": float(np.percentile(draws, 5)),
        "ci_high": float(np.percentile(draws, 95)),
    })

# Pairwise comparisons
pairwise = {}
for i, t1 in enumerate(temperatures):
    for j, t2 in enumerate(temperatures):
        if i < j:
            prob = float(np.mean(alpha_draws[t1] > alpha_draws[t2]))
            pairwise[f"{t1}_vs_{t2}"] = prob

# Strict monotonicity
n_draws = len(alpha_draws[0.0])
strictly_decreasing = 0
for i in range(n_draws):
    vals = [alpha_draws[t][i] for t in temperatures]
    if all(vals[j] > vals[j + 1] for j in range(len(vals) - 1)):
        strictly_decreasing += 1
mono_prob = strictly_decreasing / n_draws

# Slope
temp_array = np.array(temperatures)
slope_draws = []
for draw_idx in range(n_draws):
    alphas_at_draw = np.array([alpha_draws[t][draw_idx] for t in temperatures])
    b = np.cov(temp_array, alphas_at_draw)[0, 1] / np.var(temp_array)
    slope_draws.append(b)
slope_draws = np.array(slope_draws)

slope_info = {
    "median": float(np.median(slope_draws)),
    "mean": float(np.mean(slope_draws)),
    "sd": float(np.std(slope_draws)),
    "ci_low": float(np.percentile(slope_draws, 5)),
    "ci_high": float(np.percentile(slope_draws, 95)),
    "p_negative": float(np.mean(slope_draws < 0)),
}

analysis = {
    "summary_table": summary_table,
    "monotonicity_prob": mono_prob,
    "pairwise_comparisons": pairwise,
    "slope": slope_info,
}

out_path = data_dir / "primary_analysis.json"
with open(out_path, "w") as f:
    json.dump(analysis, f, indent=2)
print(f"Wrote {out_path}")
print(json.dumps(analysis, indent=2))
