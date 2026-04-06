"""Smoke test for h_m01 inference (plan §9.4)."""
import numpy as np
from cmdstanpy import CmdStanModel
from utils.study_design_hierarchical import HierarchicalStudyDesign

# 1. Generate design and simulate data
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
design = HierarchicalStudyDesign(
    J=4, K=3, D=2, R=6, P=2, M_per_cell=10, X=X,
)
data = design.generate()

sim_model = CmdStanModel(exe_file="models/h_m01_sim")
sim_fit = sim_model.sample(
    data=data, fixed_param=True, iter_sampling=1, chains=1, seed=42,
)
sim_draws = sim_fit.draws_pd()

# 2. Extract simulated observations (y) and build inference data
y_cols = sorted([c for c in sim_draws.columns if c.startswith("y[")],
                key=lambda c: int(c.split("[")[1].rstrip("]")))
y_sim = sim_draws[y_cols].values.flatten().astype(int).tolist()

# Build inference data dict (remove sim hyperparams, add y)
inference_data = {
    "J": data["J"],
    "K": data["K"],
    "D": data["D"],
    "R": data["R"],
    "P": data["P"],
    "M_total": data["M_total"],
    "M_per_cell": data["M_per_cell"],
    "cell": data["cell"],
    "I": data["I"],
    "w": data["w"],
    "X": data["X"],
    "y": y_sim,
}

# 3. Fit inference model
inf_model = CmdStanModel(exe_file="models/h_m01")
fit = inf_model.sample(
    data=inference_data,
    chains=2,
    iter_warmup=200,
    iter_sampling=200,
    seed=42,
    show_progress=True,
)

# 4. Check diagnostics
print(fit.diagnose())

# 5. Check summary for key parameters
summary = fit.summary()
print(f"\nSummary columns: {summary.columns.tolist()}")
key_params = ["gamma0", "sigma_cell", "gamma[1]", "gamma[2]",
              "alpha[1]", "alpha[2]", "alpha[3]", "alpha[4]",
              "delta[1]", "delta[2]"]
for p in key_params:
    if p in summary.index:
        row = summary.loc[p]
        print(f"  {p}: mean={row['Mean']:.3f}, sd={row['StdDev']:.3f}, "
              f"rhat={row['R_hat']:.3f}")
    else:
        print(f"  {p}: NOT FOUND in summary")

# 6. Check log_lik exists
ll_cols = [c for c in fit.column_names if c.startswith("log_lik[")]
assert len(ll_cols) == 40, f"Expected 40 log_lik columns, got {len(ll_cols)}"
print(f"\nlog_lik columns: {len(ll_cols)}")

print("\nInference smoke test PASSED")
