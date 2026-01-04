"""
Study Design Generator for m_1 Model

This module extends the base StudyDesign class to support the m_1 model,
which combines uncertain and risky choice problems.

Usage:
    from utils.study_design_m1 import StudyDesignM1
    
    # Create design with both uncertain and risky problems
    design = StudyDesignM1(
        M=20,  # uncertain problems
        N=20,  # risky problems
        K=3,   # consequences
        D=2,   # feature dimensions
        R=10,  # uncertain alternatives
        S=8    # risky alternatives
    )
    design.generate()
    design.save("results/designs/m1_study.json")
"""
import numpy as np
import json
import os
import datetime
from pathlib import Path
from utils.study_design import StudyDesign


class StudyDesignM1(StudyDesign):
    """
    Generate study designs for m_1 model with both uncertain and risky choices.
    
    Extends StudyDesign to include risky decision problems with known objective
    probabilities alongside uncertain problems with feature-based probabilities.
    
    Parameters:
        M (int): Number of uncertain decision problems
        N (int): Number of risky decision problems
        K (int): Number of possible consequences
        D (int): Number of dimensions for uncertain alternative features
        R (int): Number of distinct uncertain alternatives
        S (int): Number of distinct risky alternatives
        min_alts_per_problem (int): Minimum alternatives per problem
        max_alts_per_problem (int): Maximum alternatives per problem
        risky_probs (str): How to generate risky probabilities ('uniform', 'fixed', 'random')
        feature_dist (str): Distribution for uncertain features ('normal' or 'uniform')
        feature_params (dict): Parameters for feature distribution
        design_name (str): Name for the design
    """
    def __init__(
        self,
        M=20,              # Number of uncertain decision problems
        N=20,              # Number of risky decision problems
        K=3,               # Number of possible consequences
        D=2,               # Dimensions of uncertain alternative features
        R=10,              # Number of distinct uncertain alternatives
        S=8,               # Number of distinct risky alternatives
        min_alts_per_problem=2,
        max_alts_per_problem=5,
        risky_probs="uniform",  # How to generate risky probabilities
        feature_dist="normal",
        feature_params={"loc": 0, "scale": 1},
        design_name="m1_default"
    ):
        # Initialize parent class for uncertain problems
        super().__init__(
            M=M, K=K, D=D, R=R,
            min_alts_per_problem=min_alts_per_problem,
            max_alts_per_problem=max_alts_per_problem,
            feature_dist=feature_dist,
            feature_params=feature_params,
            design_name=design_name
        )
        
        # Add risky problem parameters
        self.N = N
        self.S = S
        self.risky_probs = risky_probs
    
    @classmethod
    def from_base_study(cls, base_study, N, S, risky_probs="random", design_name=None):
        """
        Create an m_1 study design using an existing StudyDesign for the uncertain component.
        
        This ensures that the uncertain decision problems (features w and indicator I)
        are identical to those in the base study, enabling valid comparisons between
        m_0 and m_1 models. Only risky problems are newly generated.
        
        Parameters:
            base_study (StudyDesign): An existing m_0 study design to use for uncertain problems
            N (int): Number of risky decision problems to add
            S (int): Number of distinct risky alternatives
            risky_probs (str): How to generate risky probabilities ('uniform', 'fixed', 'random')
            design_name (str): Name for the new design (defaults to base name + "_m1")
            
        Returns:
            StudyDesignM1: A new instance with the same uncertain problems as base_study
        """
        if not hasattr(base_study, 'w') or not hasattr(base_study, 'I'):
            raise ValueError("Base study must be generated before creating m_1 design from it")
        
        # Create instance with parameters matching the base study
        design = cls(
            M=base_study.M,
            N=N,
            K=base_study.K,
            D=base_study.D,
            R=base_study.R,
            S=S,
            min_alts_per_problem=base_study.min_alts,
            max_alts_per_problem=base_study.max_alts,
            risky_probs=risky_probs,
            feature_dist=base_study.feature_dist,
            feature_params=base_study.feature_params,
            design_name=design_name or f"{base_study.design_name}_m1"
        )
        
        # Copy the uncertain component from base study
        design.w = [np.array(w_vec) for w_vec in base_study.w]
        design.I = np.array(base_study.I)
        
        # Generate only the risky component
        design.x = design._generate_risky_probabilities()
        design.J = design._generate_risky_indicator_array()
        
        # Generate metadata
        design.metadata = design._generate_metadata()
        design.metadata.update(design._generate_risky_metadata())
        
        return design
        
    def generate(self):
        """
        Generate a complete m_1 study design with uncertain and risky problems.
        
        Returns:
            self: Returns self for method chaining
        """
        # Generate uncertain problems (using parent class)
        super().generate()
        
        # Generate risky problems
        self.x = self._generate_risky_probabilities()
        self.J = self._generate_risky_indicator_array()
        
        # Update metadata to include risky problem info
        self.metadata.update(self._generate_risky_metadata())
        
        return self
    
    def _generate_risky_probabilities(self):
        """
        Generate objective probability simplexes for risky alternatives.
        
        Returns:
            list: List of K-dimensional probability simplexes
        """
        simplexes = []
        
        if self.risky_probs == "uniform":
            # Generate uniform simplexes (all consequences equally likely)
            for _ in range(self.S):
                simplexes.append(np.ones(self.K) / self.K)
                
        elif self.risky_probs == "fixed":
            # Use fixed common probability values (0.25, 0.5, 0.75 for K=3)
            if self.K == 3:
                common_probs = [
                    [0.25, 0.50, 0.25],
                    [0.50, 0.25, 0.25],
                    [0.25, 0.25, 0.50],
                    [0.33, 0.33, 0.34],
                    [0.20, 0.60, 0.20],
                    [0.60, 0.20, 0.20],
                    [0.20, 0.20, 0.60],
                    [0.10, 0.80, 0.10]
                ]
                # Cycle through common probabilities or repeat
                for s in range(self.S):
                    simplexes.append(np.array(common_probs[s % len(common_probs)]))
            else:
                # For other K values, use Dirichlet
                for _ in range(self.S):
                    simplexes.append(np.random.dirichlet(np.ones(self.K)))
                    
        elif self.risky_probs == "random":
            # Generate random simplexes from uniform Dirichlet
            for _ in range(self.S):
                simplexes.append(np.random.dirichlet(np.ones(self.K)))
                
        else:
            raise ValueError(f"Unsupported risky_probs: {self.risky_probs}")
        
        return simplexes
    
    def _generate_risky_indicator_array(self):
        """
        Generate indicator array for which risky alternatives appear in which problems.
        
        Returns:
            ndarray: N×S binary array where J[n,s]=1 if risky alternative s is in problem n
        """
        J = np.zeros((self.N, self.S), dtype=int)
        
        for n in range(self.N):
            # Determine how many alternatives in this problem
            n_alts = np.random.randint(self.min_alts, self.max_alts + 1)
            
            # Select which alternatives appear in this problem
            alts_in_problem = np.random.choice(self.S, size=n_alts, replace=False)
            
            # Set indicator to 1 for these alternatives
            for alt in alts_in_problem:
                J[n, alt] = 1
                
        return J
    
    def _generate_risky_metadata(self):
        """
        Generate metadata about risky problem properties.
        
        Returns:
            dict: Dictionary of metadata for risky problems
        """
        # Calculate number of alternatives per risky problem
        n_alts_per_risky_problem = np.sum(self.J, axis=1)
        
        # Calculate frequency of each risky alternative across problems
        risky_alt_frequency = np.sum(self.J, axis=0)
        
        # Calculate statistics about probability distributions
        prob_entropies = []
        for simplex in self.x:
            # Calculate Shannon entropy
            entropy = -np.sum(simplex * np.log(simplex + 1e-10))
            prob_entropies.append(entropy)
        
        return {
            "risky_n_alts_per_problem_min": int(min(n_alts_per_risky_problem)),
            "risky_n_alts_per_problem_max": int(max(n_alts_per_risky_problem)),
            "risky_n_alts_per_problem_mean": float(np.mean(n_alts_per_risky_problem)),
            "risky_alt_frequency_min": int(min(risky_alt_frequency)),
            "risky_alt_frequency_max": int(max(risky_alt_frequency)),
            "risky_alt_frequency_mean": float(np.mean(risky_alt_frequency)),
            "risky_prob_entropy_min": float(min(prob_entropies)),
            "risky_prob_entropy_max": float(max(prob_entropies)),
            "risky_prob_entropy_mean": float(np.mean(prob_entropies))
        }
    
    def get_data_dict(self):
        """
        Get the data as a dictionary formatted for m_1 Stan models.
        
        Returns:
            dict: Stan-compatible data dictionary with both uncertain and risky data
        """
        # If uncertain data not generated, raise error
        if not hasattr(self, 'w'):
            raise ValueError("Design not generated. Call generate() first.")
        
        # If this is being called from parent's generate(), risky data won't exist yet
        # In that case, just return the uncertain data
        if not hasattr(self, 'x'):
            # Return only uncertain data (parent class format)
            return {
                "M": self.M,
                "K": self.K,
                "D": self.D,
                "R": self.R,
                "w": [list(feature_vec) for feature_vec in self.w],
                "I": self.I.tolist()
            }
        
        # Full m_1 data with both uncertain and risky
        data = {
            # Uncertain problem data
            "M": self.M,
            "K": self.K,
            "D": self.D,
            "R": self.R,
            "w": [list(feature_vec) for feature_vec in self.w],
            "I": self.I.tolist(),
            # Risky problem data
            "N": self.N,
            "S": self.S,
            "x": [list(simplex) for simplex in self.x],
            "J": self.J.tolist()
        }
        
        return data
    
    def save(self, filepath, include_metadata=True, include_plots=True):
        """
        Save the m_1 study design to a JSON file.
        
        Parameters:
            filepath (str): Path to save the JSON file
            include_metadata (bool): Whether to include metadata
            include_plots (bool): Whether to generate visualization plots
        """
        if not hasattr(self, 'metadata'):
            self.metadata = self._generate_metadata()
            self.metadata.update(self._generate_risky_metadata())
        
        # Make sure the filepath is correctly pointing to results/designs
        if not filepath.startswith("/") and not filepath.startswith("results/"):
            filepath = os.path.join("results", "designs", os.path.basename(filepath))
        
        # Prepare data dictionary
        data_dict = self.get_data_dict()
        
        # Add metadata if requested
        if include_metadata:
            data_dict["metadata"] = self.metadata
            data_dict["config"] = {
                "model": "m_1",
                "uncertain_problems": self.M,
                "risky_problems": self.N,
                "consequences": self.K,
                "feature_dimensions": self.D,
                "uncertain_alternatives": self.R,
                "risky_alternatives": self.S,
                "feature_dist": self.feature_dist,
                "feature_params": self.feature_params,
                "risky_probs": self.risky_probs,
                "design_name": self.design_name,
                "min_alts_per_problem": self.min_alts,
                "max_alts_per_problem": self.max_alts
            }
        
        # Create output directory
        filepath = Path(filepath)
        os.makedirs(filepath.parent, exist_ok=True)
        
        # Save JSON
        with open(filepath, 'w') as f:
            json.dumps(data_dict, indent=2)
            f.write(json.dumps(data_dict, indent=2))
        
        print(f"Study design saved to {filepath}")
        
        # Generate plots if requested
        if include_plots:
            plot_dir = filepath.parent / f"{filepath.stem}_plots"
            os.makedirs(plot_dir, exist_ok=True)
            self._generate_plots(plot_dir)
        
        return filepath
    
    def _generate_plots(self, plot_dir):
        """
        Generate visualization plots for the m_1 study design.
        
        Parameters:
            plot_dir (Path): Directory to save plots
        """
        import matplotlib.pyplot as plt
        
        # Plot 1: Number of alternatives per problem (both types)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        n_alts_uncertain = np.sum(self.I, axis=1)
        n_alts_risky = np.sum(self.J, axis=1)
        
        ax1.hist(n_alts_uncertain, bins=range(self.min_alts, self.max_alts + 2), 
                 alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Number of alternatives')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Uncertain Problems: Alternatives per Problem')
        
        ax2.hist(n_alts_risky, bins=range(self.min_alts, self.max_alts + 2),
                 alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of alternatives')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Risky Problems: Alternatives per Problem')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'alternatives_per_problem.png', dpi=150)
        plt.close()
        
        # Plot 2: Uncertain feature space (if D <= 3)
        if self.D == 2:
            fig, ax = plt.subplots(figsize=(8, 8))
            features = np.array(self.w)
            ax.scatter(features[:, 0], features[:, 1], s=100, alpha=0.6)
            for i, (x, y) in enumerate(features):
                ax.annotate(f'{i}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title('Uncertain Alternatives: Feature Space')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / 'uncertain_feature_space.png', dpi=150)
            plt.close()
        
        # Plot 3: Risky probability distributions
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(self.S)
        width = 0.8 / self.K
        
        for k in range(self.K):
            probs = [self.x[s][k] for s in range(self.S)]
            ax.bar(x_pos + k * width, probs, width, 
                   label=f'Consequence {k+1}', alpha=0.7)
        
        ax.set_xlabel('Risky Alternative')
        ax.set_ylabel('Probability')
        ax.set_title('Risky Alternatives: Probability Distributions')
        ax.set_xticks(x_pos + width * (self.K - 1) / 2)
        ax.set_xticklabels([f'{s}' for s in range(self.S)])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(plot_dir / 'risky_probabilities.png', dpi=150)
        plt.close()
        
        print(f"Plots saved to {plot_dir}")
    
    @classmethod
    def from_config(cls, config_path):
        """
        Create a StudyDesignM1 from a JSON configuration file.
        
        Parameters:
            config_path (str): Path to configuration JSON file
            
        Returns:
            StudyDesignM1: Configured design instance
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(
            M=config.get("M", 20),
            N=config.get("N", 20),
            K=config.get("K", 3),
            D=config.get("D", 2),
            R=config.get("R", 10),
            S=config.get("S", 8),
            min_alts_per_problem=config.get("min_alts_per_problem", 2),
            max_alts_per_problem=config.get("max_alts_per_problem", 5),
            risky_probs=config.get("risky_probs", "uniform"),
            feature_dist=config.get("feature_dist", "normal"),
            feature_params=config.get("feature_params", {"loc": 0, "scale": 1}),
            design_name=config.get("design_name", "m1_default")
        )


if __name__ == "__main__":
    """Command-line interface for generating m_1 study designs."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate study design for m_1 model')
    parser.add_argument('--M', type=int, default=20, help='Number of uncertain problems')
    parser.add_argument('--N', type=int, default=20, help='Number of risky problems')
    parser.add_argument('--K', type=int, default=3, help='Number of consequences')
    parser.add_argument('--D', type=int, default=2, help='Feature dimensions')
    parser.add_argument('--R', type=int, default=10, help='Number of uncertain alternatives')
    parser.add_argument('--S', type=int, default=8, help='Number of risky alternatives')
    parser.add_argument('--output', type=str, default='m1_study.json', 
                       help='Output filename')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    
    args = parser.parse_args()
    
    if args.config:
        design = StudyDesignM1.from_config(args.config)
    else:
        design = StudyDesignM1(
            M=args.M, N=args.N, K=args.K, 
            D=args.D, R=args.R, S=args.S
        )
    
    design.generate()
    design.save(args.output)
    print(f"\nDesign generated successfully!")
    print(f"  Uncertain problems: {design.M}")
    print(f"  Risky problems: {design.N}")
    print(f"  Total problems: {design.M + design.N}")
    print(f"  Consequences: {design.K}")
    print(f"  Uncertain alternatives: {design.R}")
    print(f"  Risky alternatives: {design.S}")
