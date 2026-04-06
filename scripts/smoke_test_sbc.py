"""Small-scale SBC test (plan §9.6)."""
import numpy as np
from utils.study_design_hierarchical import HierarchicalStudyDesign
from analysis.hierarchical_sbc import HierarchicalSBC

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
design = HierarchicalStudyDesign(
    J=4, K=3, D=2, R=6, P=2, M_per_cell=15, X=X,
)
design.generate()

sbc = HierarchicalSBC(
    study_design=design,
    n_sbc_sims=10,
    n_mcmc_samples=500,
    n_mcmc_chains=1,
    thin=3,
    output_dir="results/sbc/h_m01_smoke",
)
ranks, true_params = sbc.run()

print(f"\nRanks shape: {ranks.shape}")
print(f"True params shape: {true_params.shape}")
expected_n_params = 1 + 2 + 1 + 4 + 2  # gamma0(1) + gamma(P=2) + sigma_cell(1) + alpha(J=4) + delta(K-1=2)
assert ranks.shape == (10, expected_n_params), f"Expected (10, {expected_n_params}), got {ranks.shape}"
print("SBC smoke test PASSED")
