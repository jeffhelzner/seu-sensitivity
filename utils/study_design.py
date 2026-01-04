"""
Study Design Generator for SEU Sensitivity Analysis

This module provides tools for generating and managing study designs for 
experimental or simulation studies in the SEU sensitivity framework.

The StudyDesign class handles:
- Generation of alternative feature vectors (w)
- Assignment of alternatives to decision problems (I)
- Analysis of design properties
- Visualization of design characteristics
- Saving/loading designs from JSON

Usage:
    # Create a design with default parameters
    design = StudyDesign()
    design.generate()
    design.analyze()
    design.save("results/designs/my_design.json")

    # Create from configuration file
    design = StudyDesign.from_config("configs/my_config.json")
    
    # Load existing design
    design = StudyDesign.load("results/designs/existing_design.json")
"""
import numpy as np
import json
import os
import argparse
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

class StudyDesign:
    """
    Generate, analyze, and manage study designs for SEU sensitivity experiments.
    
    A study design consists of:
    - A set of M decision problems
    - A set of R distinct alternatives with D-dimensional feature vectors
    - An indicator matrix showing which alternatives appear in which problems
    
    Parameters:
        M (int): Number of decision problems
        K (int): Number of possible consequences for each alternative
        D (int): Number of dimensions for alternative feature vectors
        R (int): Number of distinct alternatives
        min_alts_per_problem (int): Minimum alternatives per decision problem
        max_alts_per_problem (int): Maximum alternatives per decision problem
        feature_dist (str): Distribution for feature generation ('normal' or 'uniform')
        feature_params (dict): Parameters for feature distribution
        design_name (str): Name for the design (for metadata)
    
    Attributes:
        w (list): List of D-dimensional feature vectors for each alternative
        I (ndarray): M×R indicator array where I[m,r]=1 if alternative r is in problem m
        metadata (dict): Generated metadata about design properties
    """
    def __init__(
        self,
        M=20,              # Number of decision problems
        K=3,               # Number of possible consequences
        D=2,               # Dimensions of alternative features
        R=10,              # Number of distinct alternatives
        min_alts_per_problem=2,
        max_alts_per_problem=5,
        feature_dist="normal", # Distribution for feature generation
        feature_params={"loc": 0, "scale": 1},
        design_name="default"
    ):
        self.M = M
        self.K = K
        self.D = D
        self.R = R
        self.min_alts = min_alts_per_problem
        self.max_alts = max_alts_per_problem
        self.feature_dist = feature_dist
        self.feature_params = feature_params
        self.design_name = design_name
        
    def generate(self):
        """
        Generate a complete study design.
        
        Creates feature vectors for alternatives and an indicator array
        for which alternatives appear in which decision problems.
        
        Returns:
            dict: The generated design as a Stan-compatible data dictionary
        """
        # Generate feature vectors for each distinct alternative
        self.w = self._generate_features()
        
        # Generate indicator array for which alternatives appear in which problems
        self.I = self._generate_indicator_array()
        
        # Calculate design metadata
        self.metadata = self._generate_metadata()
        
        return self.get_data_dict()
    
    def _generate_features(self):
        """
        Generate feature vectors for alternatives.
        
        Creates R feature vectors of dimension D according to the specified distribution.
        
        Returns:
            list: List of D-dimensional numpy arrays representing feature vectors
        """
        if self.feature_dist == "normal":
            return [np.random.normal(
                loc=self.feature_params["loc"],
                scale=self.feature_params["scale"],
                size=self.D
            ) for _ in range(self.R)]
        elif self.feature_dist == "uniform":
            low = self.feature_params.get("low", -1)
            high = self.feature_params.get("high", 1)
            return [np.random.uniform(low=low, high=high, size=self.D) 
                   for _ in range(self.R)]
        else:
            raise ValueError(f"Unsupported feature distribution: {self.feature_dist}")
    
    def _generate_indicator_array(self):
        """
        Generate indicator array for which alternatives appear in which problems.
        
        For each decision problem m, randomly selects between min_alts and max_alts
        distinct alternatives to include.
        
        Returns:
            ndarray: M×R binary array where I[m,r]=1 if alternative r is in problem m
        """
        I = np.zeros((self.M, self.R), dtype=int)
        
        for m in range(self.M):
            # Determine how many alternatives in this problem
            n_alts = np.random.randint(self.min_alts, self.max_alts + 1)
            
            # Select which alternatives appear in this problem
            alts_in_problem = np.random.choice(self.R, size=n_alts, replace=False)
            
            # Set indicator to 1 for these alternatives
            for alt in alts_in_problem:
                I[m, alt] = 1
                
        return I
    
    def _generate_metadata(self):
        """
        Generate metadata about the design properties.
        
        Calculates various statistics about the design:
        - Number of alternatives per problem
        - Frequency of each alternative across problems
        - Feature value distributions and correlations
        
        Returns:
            dict: Dictionary of metadata values
        
        Raises:
            ValueError: If called before generating the design
        """
        if not hasattr(self, 'w') or not hasattr(self, 'I'):
            raise ValueError("Cannot generate metadata before generating design")
            
        # Calculate number of alternatives per problem
        n_alts_per_problem = np.sum(self.I, axis=1)
        
        # Calculate frequency of each alternative across problems
        alt_frequency = np.sum(self.I, axis=0)
        
        # Calculate feature space coverage statistics
        feature_ranges = []
        feature_means = []
        feature_stds = []
        
        for d in range(self.D):
            feature_values = [vec[d] for vec in self.w]
            feature_ranges.append((min(feature_values), max(feature_values)))
            feature_means.append(np.mean(feature_values))
            feature_stds.append(np.std(feature_values))
        
        # Generate pairwise correlations between alternatives
        correlations = {}
        if self.D >= 2:  # Only calculate if there are at least 2 dimensions
            for i in range(self.D):
                for j in range(i+1, self.D):
                    feat_i = [vec[i] for vec in self.w]
                    feat_j = [vec[j] for vec in self.w]
                    corr = np.corrcoef(feat_i, feat_j)[0, 1]
                    correlations[f"dim_{i+1}_dim_{j+1}"] = corr
        
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "n_alts_per_problem_min": int(min(n_alts_per_problem)),
            "n_alts_per_problem_max": int(max(n_alts_per_problem)),
            "n_alts_per_problem_mean": float(np.mean(n_alts_per_problem)),
            "alt_frequency_min": int(min(alt_frequency)),
            "alt_frequency_max": int(max(alt_frequency)),
            "alt_frequency_mean": float(np.mean(alt_frequency)),
            "feature_ranges": feature_ranges,
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "feature_correlations": correlations
        }
    
    def get_data_dict(self):
        """
        Get the data as a dictionary formatted for Stan models.
        
        Converts the study design into a format compatible with Stan's data block.
        Will generate the design first if it hasn't been generated already.
        
        Returns:
            dict: Stan-compatible data dictionary
        """
        if not hasattr(self, 'w') or not hasattr(self, 'I'):
            self.generate()
            
        # Convert w to the format expected by Stan (array of vectors)
        w_stan = []
        for feature_vector in self.w:
            w_stan.append(list(feature_vector))
            
        return {
            "M": self.M,
            "K": self.K,
            "D": self.D,
            "R": self.R,
            "w": w_stan,
            "I": self.I.tolist(),
            # Default parameter generation controls
            "alpha_mean": 0,  # Log-scale mean
            "alpha_sd": 1,    # Log-scale sd
            "beta_sd": 1      # SD for beta coefficients
        }
    
    def save(self, filepath, include_metadata=True, include_plots=True):
        """
        Save the study design to a JSON file with optional metadata and plots.
        
        Parameters:
            filepath (str): Path to save the JSON file
            include_metadata (bool): Whether to include metadata in the JSON
            include_plots (bool): Whether to generate and save visualization plots
            
        Returns:
            Path: The path where the file was saved
            
        Notes:
            - If filepath doesn't start with "/" or "results/", it will be saved
              under results/designs/
            - If include_plots=True, plots will be saved in a directory named
              [filename]_plots next to the JSON file
        """
        if not hasattr(self, 'metadata'):
            self.metadata = self._generate_metadata()
        
        # Make sure the filepath is correctly pointing to results/designs if not specified
        if not filepath.startswith("/") and not filepath.startswith("results/"):
            filepath = os.path.join("results", "designs", os.path.basename(filepath))
        
        # Prepare data dictionary
        data_dict = self.get_data_dict()
        
        # Add metadata if requested
        if include_metadata:
            data_dict["metadata"] = self.metadata
            data_dict["config"] = {
                "feature_dist": self.feature_dist,
                "feature_params": self.feature_params,
                "design_name": self.design_name,
                "min_alts_per_problem": self.min_alts,
                "max_alts_per_problem": self.max_alts
            }
        
        # Create output directory
        filepath = Path(filepath)
        os.makedirs(filepath.parent, exist_ok=True)
        
        # Save JSON with a simpler approach to format the I array compactly
        with open(filepath, 'w') as f:
            # First convert to JSON string with standard formatting
            json_str = json.dumps(data_dict, indent=2)
            
            # Now reformat the I array for better readability
            # Find the I array in the JSON string
            i_start = json_str.find('"I": [')
            if i_start != -1:
                i_start += 6  # Move past the '"I": [' part
                
                # Find the matching closing bracket
                bracket_count = 1
                i_end = i_start
                while bracket_count > 0 and i_end < len(json_str):
                    if json_str[i_end] == '[':
                        bracket_count += 1
                    elif json_str[i_end] == ']':
                        bracket_count -= 1
                    i_end += 1
                
                # Extract the I array content
                i_array_str = json_str[i_start:i_end-1]
                
                # Convert to Python object
                i_array = json.loads(f"[{i_array_str}]")
                
                # Create compact representation
                compact_i = "[\n"
                for row in i_array:
                    compact_i += f"      {row},\n"
                compact_i = compact_i[:-2] + "\n    ]"  # Remove last comma and add closing bracket
                
                # Replace in the original JSON
                json_str = json_str[:i_start-6] + '"I": ' + compact_i + json_str[i_end:]
            
            # Write the reformatted JSON string
            f.write(json_str)
            
        # Generate and save plots if requested
        if include_plots:
            plots_dir = filepath.parent / f"{filepath.stem}_plots"
            os.makedirs(plots_dir, exist_ok=True)
            self._save_plots(plots_dir)
            
        return filepath
    
    def _save_plots(self, plots_dir):
        """
        Generate and save analysis plots for the study design.
        
        Creates several visualization plots:
        1. Feature value distributions
        2. Number of alternatives per problem
        3. Alternative frequency across problems
        4. 2D feature space visualization (if D >= 2)
        
        Parameters:
            plots_dir (Path): Directory to save plots
        """
        # 1. Plot feature distributions
        plt.figure(figsize=(10, 6))
        for d in range(min(self.D, 5)):  # Limit to first 5 dimensions
            feature_values = [vec[d] for vec in self.w]
            # Create histograms for feature distributions
            plt.hist(feature_values, bins=20, alpha=0.5, density=True, 
                     histtype='stepfilled', label=f'Dimension {d+1}')
        plt.title('Feature Value Distributions')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_distributions.png'))
        plt.close()
        
        # 2. Plot alternatives per problem distribution
        n_alts_per_problem = np.sum(self.I, axis=1)
        plt.figure(figsize=(8, 5))
        # Create histogram of number of alternatives per problem
        min_alts = int(min(n_alts_per_problem))
        max_alts = int(max(n_alts_per_problem))
        plt.hist(n_alts_per_problem, bins=range(min_alts, max_alts+2), 
                 align='left', rwidth=0.8)
        plt.title('Number of Alternatives per Problem')
        plt.xlabel('Number of Alternatives')
        plt.ylabel('Count of Problems')
        plt.xticks(range(min_alts, max_alts+1))
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'alts_per_problem.png'))
        plt.close()
        
        # 3. Plot alternative frequency across problems
        alt_frequency = np.sum(self.I, axis=0)
        plt.figure(figsize=(10, 6))
        # Create bar chart of alternative frequency
        plt.bar(range(1, self.R+1), alt_frequency)
        plt.title('Alternative Appearance Frequency')
        plt.xlabel('Alternative ID')
        plt.ylabel('Number of Problems')
        plt.xticks(range(1, self.R+1))
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'alt_frequency.png'))
        plt.close()
        
        # 4. Feature space coverage visualization (for 2D)
        if self.D >= 2:
            plt.figure(figsize=(8, 8))
            feature_0 = [vec[0] for vec in self.w]
            feature_1 = [vec[1] for vec in self.w]
            plt.scatter(feature_0, feature_1)
            for i, (x, y) in enumerate(zip(feature_0, feature_1)):
                plt.annotate(f"{i+1}", (x, y), fontsize=9)
            plt.title('Feature Space Coverage (First 2 Dimensions)')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'feature_space.png'))
            plt.close()
    
    def extend(self, additional_M, design_name=None):
        """
        Create a new study design by extending this one with additional problems.
        
        The new design will have:
        - The same alternatives (w) as this design
        - The original M problems from this design
        - additional_M new problems using the same alternatives
        
        This is useful for comparing recovery across sample sizes while holding
        the alternatives fixed, ensuring any differences are due to sample size
        rather than different feature configurations.
        
        Parameters:
            additional_M (int): Number of additional decision problems to add
            design_name (str): Name for the new design (defaults to appending "_extended")
            
        Returns:
            StudyDesign: A new StudyDesign instance with extended problems
        """
        if not hasattr(self, 'w') or not hasattr(self, 'I'):
            raise ValueError("Cannot extend a design that hasn't been generated")
        
        # Create new design with extended M
        new_M = self.M + additional_M
        new_design = StudyDesign(
            M=new_M,
            K=self.K,
            D=self.D,
            R=self.R,
            min_alts_per_problem=self.min_alts,
            max_alts_per_problem=self.max_alts,
            feature_dist=self.feature_dist,
            feature_params=self.feature_params,
            design_name=design_name or f"{self.design_name}_extended"
        )
        
        # Copy the same alternatives (features)
        new_design.w = [np.array(w_vec) for w_vec in self.w]
        
        # Start with original indicator array and extend with new problems
        I_extended = np.zeros((new_M, self.R), dtype=int)
        I_extended[:self.M, :] = self.I  # Copy original problems
        
        # Generate additional problems using the same alternatives
        for m in range(self.M, new_M):
            n_alts = np.random.randint(self.min_alts, self.max_alts + 1)
            alts_in_problem = np.random.choice(self.R, size=n_alts, replace=False)
            for alt in alts_in_problem:
                I_extended[m, alt] = 1
        
        new_design.I = I_extended
        new_design.metadata = new_design._generate_metadata()
        
        return new_design
    
    def analyze(self):
        """
        Print analysis of the study design.
        
        Outputs a text summary of key design statistics:
        - Basic design parameters
        - Problem statistics
        - Alternative statistics 
        - Feature statistics and correlations
        """
        if not hasattr(self, 'metadata'):
            self.metadata = self._generate_metadata()
            
        print(f"Study Design Analysis: {self.design_name}")
        print("-" * 50)
        print(f"Total problems: {self.M}")
        print(f"Distinct alternatives: {self.R}")
        print(f"Feature dimensions: {self.D}")
        print(f"Consequences: {self.K}")
        print("-" * 50)
        print("Problem statistics:")
        print(f"  Alternatives per problem: {self.metadata['n_alts_per_problem_min']} to "
              f"{self.metadata['n_alts_per_problem_max']} (mean: {self.metadata['n_alts_per_problem_mean']:.2f})")
        print("Alternative statistics:")
        print(f"  Frequency in problems: {self.metadata['alt_frequency_min']} to "
              f"{self.metadata['alt_frequency_max']} (mean: {self.metadata['alt_frequency_mean']:.2f})")
        print("Feature statistics:")
        for d in range(self.D):
            print(f"  Dimension {d+1}: range {self.metadata['feature_ranges'][d]}, "
                  f"mean {self.metadata['feature_means'][d]:.3f}, "
                  f"std {self.metadata['feature_stds'][d]:.3f}")
        
        if self.metadata['feature_correlations']:
            print("Feature correlations:")
            for pair, corr in self.metadata['feature_correlations'].items():
                print(f"  {pair}: {corr:.3f}")
    
    @classmethod
    def from_config(cls, config_path):
        """
        Create a study design from a configuration file.
        
        Parameters:
            config_path (str): Path to JSON configuration file
            
        Returns:
            StudyDesign: Initialized instance with parameters from config file
            
        Example config file:
        {
            "M": 30,
            "K": 4,
            "D": 3,
            "R": 15,
            "min_alts_per_problem": 2,
            "max_alts_per_problem": 6,
            "feature_dist": "uniform",
            "feature_params": {
                "low": -2,
                "high": 2
            },
            "design_name": "uniform_large_study",
            "generate_on_load": true
        }
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Extract parameters with defaults
        params = {
            'M': config.get('M', 20),
            'K': config.get('K', 3),
            'D': config.get('D', 2),
            'R': config.get('R', 10),
            'min_alts_per_problem': config.get('min_alts_per_problem', 2),
            'max_alts_per_problem': config.get('max_alts_per_problem', 5),
            'feature_dist': config.get('feature_dist', 'normal'),
            'feature_params': config.get('feature_params', {"loc": 0, "scale": 1}),
            'design_name': config.get('design_name', 'from_config')
        }
        
        # Create instance with extracted parameters
        design = cls(**params)
        
        # Generate the design if specified
        if config.get('generate_on_load', True):
            design.generate()
            
        return design
    
    @classmethod
    def load(cls, filepath):
        """
        Load a study design from a JSON file.
        
        Parameters:
            filepath (str): Path to saved study design JSON file
            
        Returns:
            StudyDesign: Instance with loaded design data
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Create instance with basic parameters
        design = cls(
            M=data["M"],
            K=data["K"],
            D=data["D"],
            R=data["R"]
        )
        
        # Load configuration if available
        if "config" in data:
            design.feature_dist = data["config"].get("feature_dist", design.feature_dist)
            design.feature_params = data["config"].get("feature_params", design.feature_params)
            design.design_name = data["config"].get("design_name", design.design_name)
            design.min_alts = data["config"].get("min_alts_per_problem", design.min_alts)
            design.max_alts = data["config"].get("max_alts_per_problem", design.max_alts)
        
        # Convert loaded data back to numpy arrays
        design.w = [np.array(w_vector) for w_vector in data["w"]]
        design.I = np.array(data["I"])
        
        # Load metadata if available
        if "metadata" in data:
            design.metadata = data["metadata"]
        
        return design


def main():
    """
    Command line interface for testing study design.
    
    Allows running the study design generator from the command line with options:
    --config: Path to configuration file
    --output: Path for saving the generated design
    --seed: Random seed for reproducibility
    
    Examples:
        python study_design.py --output my_design.json
        python study_design.py --config configs/my_config.json --seed 42
    """
    parser = argparse.ArgumentParser(description="Generate and test study designs")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output', type=str, default='study_design.json', 
                        help='Output path for generated design')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Create design from config or default
    if args.config:
        design = StudyDesign.from_config(args.config)
    else:
        design = StudyDesign()
        design.generate()
    
    # Analyze and save the design
    design.analyze()
    output_path = design.save(args.output, include_metadata=True, include_plots=True)
    print(f"\nDesign saved to {output_path}")
    print(f"Plots saved to {str(Path(output_path).parent / f'{Path(output_path).stem}_plots')}")


if __name__ == "__main__":
    main()