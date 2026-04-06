"""Smoke test for h_m01_sim (plan §9.3)."""
import numpy as np
from cmdstanpy import CmdStanModel
from utils.study_design_hierarchical import HierarchicalStudyDesign

# 1. Generate design
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
design = HierarchicalStudyDesign(
    J=4, K=3, D=2, R=6, P=2, M_per_cell=10, X=X,
)
data = design.generate()

# 2. Compile and run simulation model
sim_model = CmdStanModel(exe_file="models/h_m01_sim")
sim_fit = sim_model.sample(
    data=data,
    fixed_param=True,
    iter_sampling=1,
    chains=1,
    seed=42,
)

# 3. Check generated quantities
draws = sim_fit.draws_pd()
print(f"Columns ({len(draws.columns)}): {sorted(draws.columns.tolist())[:30]}...")

# Key parameters should exist
expected_prefixes = ["gamma0", "sigma_cell", "delta"]
for prefix in expected_prefixes:
    matching = [c for c in draws.columns if c.startswith(prefix)]
    assert len(matching) > 0, f"Missing columns with prefix '{prefix}'"
    print(f"  {prefix}: {matching}")

# y should exist for all observations
y_cols = [c for c in draws.columns if c.startswith("y[")]
assert len(y_cols) == 40, f"Expected 40 y columns, got {len(y_cols)}"
print(f"  y columns: {len(y_cols)}")

# alpha should have J values
alpha_cols = [c for c in draws.columns if c.startswith("alpha[")]
assert len(alpha_cols) == 4, f"Expected 4 alpha columns, got {len(alpha_cols)}"
print(f"  alpha columns: {len(alpha_cols)}")

# gamma should have P values
gamma_cols = [c for c in draws.columns if c.startswith("gamma[")]
assert len(gamma_cols) == 2, f"Expected 2 gamma columns, got {len(gamma_cols)}"
print(f"  gamma columns: {len(gamma_cols)}")

# beta should have J*K*D = 4*3*2 = 24 values (array[J] matrix[K, D])
beta_cols = [c for c in draws.columns if c.startswith("beta[")]
assert len(beta_cols) == 24, f"Expected 24 beta columns, got {len(beta_cols)}"
print(f"  beta columns: {len(beta_cols)}")

print("\nSimulation smoke test PASSED")
