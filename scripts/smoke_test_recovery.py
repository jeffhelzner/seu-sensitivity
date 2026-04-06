"""Small-scale parameter recovery test (plan §9.5)."""
import numpy as np
from utils.study_design_hierarchical import HierarchicalStudyDesign
from analysis.hierarchical_parameter_recovery import HierarchicalParameterRecovery

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
design = HierarchicalStudyDesign(
    J=4, K=3, D=2, R=6, P=2, M_per_cell=15, X=X,
)
design.generate()

recovery = HierarchicalParameterRecovery(
    study_design=design,
    n_iterations=3,
    n_mcmc_samples=500,
    n_mcmc_chains=2,
    output_dir="results/parameter_recovery/h_m01_smoke",
)
true_params, summaries = recovery.run()

print(f"\nCompleted iterations: {len(true_params)}")
if len(true_params) > 0:
    print("Parameter recovery smoke test PASSED")
else:
    print("Parameter recovery smoke test FAILED - no iterations completed")
