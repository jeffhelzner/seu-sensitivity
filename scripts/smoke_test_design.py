"""Smoke test for HierarchicalStudyDesign (plan §9.2)."""
import numpy as np
from utils.study_design_hierarchical import HierarchicalStudyDesign

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
design = HierarchicalStudyDesign(
    J=4, K=3, D=2, R=6, P=2,
    M_per_cell=10, X=X,
)
data = design.generate()

assert len(data["w"]) == 6, f"w: {len(data['w'])}"
assert len(data["w"][0]) == 2, f"w[0]: {len(data['w'][0])}"
assert len(data["cell"]) == 40, f"cell: {len(data['cell'])}"
assert len(data["I"]) == 40, f"I: {len(data['I'])}"
assert len(data["I"][0]) == 6, f"I[0]: {len(data['I'][0])}"
assert len(data["X"]) == 4, f"X: {len(data['X'])}"
assert len(data["X"][0]) == 2, f"X[0]: {len(data['X'][0])}"
assert data["M_total"] == 40, f"M_total: {data['M_total']}"
assert data["M_per_cell"] == [10, 10, 10, 10], f"M_per_cell: {data['M_per_cell']}"

print("Study design smoke test PASSED")
print(f"Keys: {sorted(data.keys())}")
