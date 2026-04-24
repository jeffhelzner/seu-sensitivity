"""
Study design generator for hierarchical models.

Generates a stacked data structure for J experimental cells sharing a
common alternative pool, suitable for h_m01.stan.
"""

import numpy as np
import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


class HierarchicalStudyDesign:
    """
    Generates stacked study designs for hierarchical SEU sensitivity models.

    Parameters
    ----------
    J : int
        Number of experimental cells.
    K : int
        Number of consequences.
    D : int
        Embedding dimension per alternative.
    R : int
        Number of distinct alternatives (shared across cells).
    P : int
        Number of predictors in design matrix (excluding intercept).
    M_per_cell : int or List[int]
        Observations per cell. If int, same for all cells.
    X : np.ndarray, shape (J, P)
        Design matrix for cell-level regression.
    min_alts_per_problem : int
        Minimum alternatives per choice problem.
    max_alts_per_problem : int
        Maximum alternatives per choice problem.
    feature_dist : str
        Distribution for generating feature vectors ("normal" or "uniform").
    feature_params : dict
        Parameters for feature distribution.
    design_name : str
        Human-readable name for this design.
    """

    def __init__(
        self,
        J: int = 6,
        K: int = 3,
        D: int = 2,
        R: int = 10,
        P: int = 2,
        M_per_cell: int | List[int] = 20,
        X: Optional[np.ndarray] = None,
        min_alts_per_problem: int = 2,
        max_alts_per_problem: int = 4,
        feature_dist: str = "normal",
        feature_params: Optional[dict] = None,
        design_name: str = "hierarchical_default",
    ):
        self.J = J
        self.K = K
        self.D = D
        self.R = R
        self.P = P
        self.min_alts = min_alts_per_problem
        self.max_alts = max_alts_per_problem
        self.feature_dist = feature_dist
        self.feature_params = feature_params or {"loc": 0, "scale": 1}
        self.design_name = design_name

        # Handle M_per_cell
        if isinstance(M_per_cell, int):
            self._M_per_cell = [M_per_cell] * J
        else:
            if len(M_per_cell) != J:
                raise ValueError(f"M_per_cell length ({len(M_per_cell)}) must equal J ({J})")
            self._M_per_cell = list(M_per_cell)

        # Handle design matrix
        if X is not None:
            X = np.asarray(X, dtype=float)
            if X.shape != (J, P):
                raise ValueError(f"X shape {X.shape} must be ({J}, {P})")
            self._X = X
        else:
            # Default: dummy coding for J cells with P predictors
            # Generate a simple balanced design matrix
            self._X = self._default_design_matrix()

        # Will be populated by generate()
        self.w = None
        self.I = None
        self.cell = None
        self.M_total = sum(self._M_per_cell)

    def _default_design_matrix(self) -> np.ndarray:
        """Generate a simple balanced design matrix."""
        # Create a design matrix with alternating 0/1 patterns
        X = np.zeros((self.J, self.P), dtype=float)
        for j in range(self.J):
            for p in range(self.P):
                X[j, p] = float((j >> p) & 1)
        return X

    def generate(self) -> dict:
        """
        Generate the complete stacked design.

        Returns dict with keys matching h_m01.stan data block:
        {J, K, D, R, P, w, M_total, cell, I, M_per_cell, X,
         plus sim hyperparams: gamma0_mean, gamma0_sd, gamma_sd,
         sigma_cell_sd, beta_sd}
        """
        # Generate shared feature vectors
        self.w = self._generate_features()

        # Generate stacked indicator array and cell vector
        self.I, self.cell = self._generate_stacked_indicators()

        return self.get_data_dict()

    def _generate_features(self) -> List[np.ndarray]:
        """Generate R feature vectors of dimension D."""
        if self.feature_dist == "normal":
            return [
                np.random.normal(
                    loc=self.feature_params.get("loc", 0),
                    scale=self.feature_params.get("scale", 1),
                    size=self.D,
                )
                for _ in range(self.R)
            ]
        elif self.feature_dist == "uniform":
            low = self.feature_params.get("low", -1)
            high = self.feature_params.get("high", 1)
            return [
                np.random.uniform(low=low, high=high, size=self.D)
                for _ in range(self.R)
            ]
        else:
            raise ValueError(f"Unsupported feature distribution: {self.feature_dist}")

    def _generate_stacked_indicators(self):
        """
        Generate stacked indicator arrays and cell membership vector.

        Returns
        -------
        I : np.ndarray, shape (M_total, R)
            Indicator matrix for all observations across all cells.
        cell : np.ndarray, shape (M_total,)
            Cell membership vector (1-indexed).
        """
        I = np.zeros((self.M_total, self.R), dtype=int)
        cell = np.zeros(self.M_total, dtype=int)

        row = 0
        for j in range(self.J):
            for _ in range(self._M_per_cell[j]):
                # Random number of alternatives for this problem
                n_alts = np.random.randint(self.min_alts, self.max_alts + 1)
                n_alts = min(n_alts, self.R)  # Can't exceed pool size

                # Select which alternatives appear
                alts = np.random.choice(self.R, size=n_alts, replace=False)
                for a in alts:
                    I[row, a] = 1

                cell[row] = j + 1  # 1-indexed for Stan
                row += 1

        return I, cell

    def get_data_dict(self) -> dict:
        """Return Stan-compatible data dictionary."""
        if self.w is None or self.I is None:
            self.generate()

        # Convert w to list of lists
        w_stan = [list(v) for v in self.w]

        data = {
            "J": self.J,
            "K": self.K,
            "D": self.D,
            "R": self.R,
            "P": self.P,
            "w": w_stan,
            "M_total": self.M_total,
            "cell": self.cell.tolist(),
            "I": self.I.tolist(),
            "M_per_cell": self._M_per_cell,
            "X": self._X.tolist(),
            # Default sim hyperparams (match h_m01.stan priors; tightened 2026-04-23)
            "gamma0_mean": 2.5,
            "gamma0_sd": 0.5,
            "gamma_sd": 0.5,
            "sigma_cell_sd": 0.3,
            "beta_sd": 1.0,
        }
        return data

    def save(self, filepath: str) -> Path:
        """Save design to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = self.get_data_dict()
        data["_metadata"] = {
            "design_name": self.design_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "min_alts_per_problem": self.min_alts,
            "max_alts_per_problem": self.max_alts,
            "feature_dist": self.feature_dist,
            "feature_params": self.feature_params,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    @classmethod
    def load(cls, filepath: str) -> "HierarchicalStudyDesign":
        """Load design from JSON."""
        with open(filepath) as f:
            data = json.load(f)

        metadata = data.pop("_metadata", {})

        design = cls(
            J=data["J"],
            K=data["K"],
            D=data["D"],
            R=data["R"],
            P=data["P"],
            M_per_cell=data["M_per_cell"],
            X=np.array(data["X"]),
            min_alts_per_problem=metadata.get("min_alts_per_problem", 2),
            max_alts_per_problem=metadata.get("max_alts_per_problem", 4),
            feature_dist=metadata.get("feature_dist", "normal"),
            feature_params=metadata.get("feature_params", {"loc": 0, "scale": 1}),
            design_name=metadata.get("design_name", "loaded"),
        )

        # Set pre-generated data
        design.w = [np.array(v) for v in data["w"]]
        design.I = np.array(data["I"])
        design.cell = np.array(data["cell"])

        return design
