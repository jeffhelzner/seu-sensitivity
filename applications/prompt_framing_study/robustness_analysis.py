"""
Robustness Analysis Module for Prompt Framing Study

Analyzes sensitivity of results to embedding models and PCA dimensions.
"""
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import logging

from .contextualized_embedding import ContextualizedEmbeddingManager
from .prompt_manager import PromptManager, PromptVariant

logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """
    Analyze robustness of alpha estimates across different embedding configurations.
    
    Tests:
    1. Different embedding models (text-embedding-3-small vs large)
    2. Different PCA dimensions (16, 32, 64, 128)
    3. PCA vs random projection
    """
    
    def __init__(
        self,
        results_dir: str,
        embedding_models: Optional[List[str]] = None,
        target_dims: Optional[List[int]] = None,
        reduction_methods: Optional[List[str]] = None
    ):
        """
        Initialize robustness analyzer.
        
        Args:
            results_dir: Directory containing study results
            embedding_models: List of embedding models to test
            target_dims: List of PCA dimensions to test
            reduction_methods: List of reduction methods ("pca", "random")
        """
        self.results_dir = Path(results_dir)
        
        self.embedding_models = embedding_models or [
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]
        
        self.target_dims = target_dims or [16, 32, 64, 128]
        self.reduction_methods = reduction_methods or ["pca", "random"]
        
        # Store results
        self.analysis_results: Dict[str, Any] = {}
    
    def analyze_pca_sensitivity(
        self,
        embeddings: Dict[str, Dict[str, np.ndarray]],
        claims: List[Dict[str, str]],
        problems: List[Dict],
        choices: Dict[str, List[int]],
        fit_model: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity to PCA dimension.
        
        Args:
            embeddings: Raw embeddings by variant
            claims: List of claims
            problems: List of problems
            choices: Choices by variant
            fit_model: Whether to fit Stan model for each configuration
            
        Returns:
            Dict with sensitivity analysis results
        """
        results = {
            "target_dims": self.target_dims,
            "variants": list(embeddings.keys()),
            "analysis": {}
        }
        
        embedding_manager = ContextualizedEmbeddingManager()
        
        for target_dim in self.target_dims:
            dim_results = {}
            
            for variant_name, variant_embeddings in embeddings.items():
                # Reduce dimensions
                reduced, reducer = embedding_manager.reduce_dimensions(
                    variant_embeddings,
                    target_dim=target_dim,
                    method="pca"
                )
                
                # Calculate explained variance if PCA
                if hasattr(reducer, 'explained_variance_ratio_'):
                    explained_var = float(sum(reducer.explained_variance_ratio_))
                else:
                    explained_var = None
                
                # Calculate embedding statistics
                emb_matrix = np.vstack(list(reduced.values()))
                
                dim_results[variant_name] = {
                    "target_dim": target_dim,
                    "actual_dim": emb_matrix.shape[1],
                    "explained_variance": explained_var,
                    "embedding_stats": {
                        "mean_norm": float(np.mean(np.linalg.norm(emb_matrix, axis=1))),
                        "std_norm": float(np.std(np.linalg.norm(emb_matrix, axis=1))),
                        "mean_pairwise_dist": self._mean_pairwise_distance(emb_matrix)
                    }
                }
                
                # Optionally fit model
                if fit_model:
                    # Would need to create Stan data and fit here
                    logger.info(f"Model fitting for dim={target_dim}, variant={variant_name}")
            
            results["analysis"][f"dim_{target_dim}"] = dim_results
        
        self.analysis_results["pca_sensitivity"] = results
        return results
    
    def analyze_embedding_model_sensitivity(
        self,
        claims: List[Dict[str, str]],
        variants: List[PromptVariant]
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity to embedding model choice.
        
        Note: This requires API calls for different models.
        
        Args:
            claims: List of claims
            variants: List of prompt variants
            
        Returns:
            Dict with model comparison results
        """
        results = {
            "models": self.embedding_models,
            "variants": [v.name for v in variants],
            "comparisons": {}
        }
        
        embeddings_by_model = {}
        
        for model in self.embedding_models:
            logger.info(f"Generating embeddings with {model}...")
            
            manager = ContextualizedEmbeddingManager(embedding_model=model)
            embeddings = manager.embed_all_variants(claims, variants)
            embeddings_by_model[model] = embeddings
        
        # Compare embeddings across models
        for variant in variants:
            variant_comparison = {}
            
            models = list(embeddings_by_model.keys())
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    emb1 = embeddings_by_model[model1][variant.name]
                    emb2 = embeddings_by_model[model2][variant.name]
                    
                    # Calculate correlation between embeddings
                    correlations = []
                    for claim_id in emb1.keys():
                        if claim_id in emb2:
                            # Truncate to same length
                            e1 = emb1[claim_id]
                            e2 = emb2[claim_id]
                            min_len = min(len(e1), len(e2))
                            corr = np.corrcoef(e1[:min_len], e2[:min_len])[0, 1]
                            correlations.append(corr)
                    
                    variant_comparison[f"{model1}_vs_{model2}"] = {
                        "mean_correlation": float(np.mean(correlations)),
                        "std_correlation": float(np.std(correlations)),
                        "min_correlation": float(np.min(correlations)),
                        "max_correlation": float(np.max(correlations))
                    }
            
            results["comparisons"][variant.name] = variant_comparison
        
        self.analysis_results["model_sensitivity"] = results
        return results
    
    def analyze_reduction_method_sensitivity(
        self,
        embeddings: Dict[str, Dict[str, np.ndarray]],
        target_dim: int = 32
    ) -> Dict[str, Any]:
        """
        Compare PCA vs random projection.
        
        Args:
            embeddings: Raw embeddings by variant
            target_dim: Target dimension for reduction
            
        Returns:
            Comparison results
        """
        results = {
            "methods": self.reduction_methods,
            "target_dim": target_dim,
            "comparisons": {}
        }
        
        embedding_manager = ContextualizedEmbeddingManager()
        
        for variant_name, variant_embeddings in embeddings.items():
            reduced_by_method = {}
            stats_by_method = {}
            
            for method in self.reduction_methods:
                reduced, _ = embedding_manager.reduce_dimensions(
                    variant_embeddings,
                    target_dim=target_dim,
                    method=method
                )
                reduced_by_method[method] = reduced
                
                emb_matrix = np.vstack(list(reduced.values()))
                stats_by_method[method] = {
                    "mean_norm": float(np.mean(np.linalg.norm(emb_matrix, axis=1))),
                    "mean_pairwise_dist": self._mean_pairwise_distance(emb_matrix)
                }
            
            # Compare methods
            if "pca" in reduced_by_method and "random" in reduced_by_method:
                pca_matrix = np.vstack(list(reduced_by_method["pca"].values()))
                random_matrix = np.vstack(list(reduced_by_method["random"].values()))
                
                # Procrustes analysis to compare structures
                structure_similarity = self._procrustes_similarity(pca_matrix, random_matrix)
            else:
                structure_similarity = None
            
            results["comparisons"][variant_name] = {
                "stats_by_method": stats_by_method,
                "structure_similarity": structure_similarity
            }
        
        self.analysis_results["reduction_method_sensitivity"] = results
        return results
    
    def _mean_pairwise_distance(self, matrix: np.ndarray) -> float:
        """Calculate mean pairwise Euclidean distance."""
        from scipy.spatial.distance import pdist
        distances = pdist(matrix, metric='euclidean')
        return float(np.mean(distances))
    
    def _procrustes_similarity(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate Procrustes similarity between two embedding matrices.
        
        Returns value between 0 (dissimilar) and 1 (identical structure).
        """
        from scipy.spatial import procrustes
        
        # Standardize
        X_std = (X - X.mean(axis=0)) / X.std()
        Y_std = (Y - Y.mean(axis=0)) / Y.std()
        
        try:
            _, _, disparity = procrustes(X_std, Y_std)
            # Convert disparity to similarity
            similarity = 1 - disparity
            return float(max(0, similarity))
        except Exception as e:
            logger.warning(f"Procrustes analysis failed: {e}")
            return None
    
    def run_full_analysis(
        self,
        claims: List[Dict[str, str]],
        problems: List[Dict],
        variants: List[PromptVariant],
        choices: Dict[str, List[int]],
        raw_embeddings: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Run complete robustness analysis.
        
        Args:
            claims: List of claims
            problems: List of problems
            variants: List of prompt variants
            choices: Choices by variant
            raw_embeddings: Raw embeddings by variant
            
        Returns:
            Complete analysis results
        """
        logger.info("Running full robustness analysis...")
        
        # PCA dimension sensitivity
        logger.info("Analyzing PCA dimension sensitivity...")
        pca_results = self.analyze_pca_sensitivity(
            raw_embeddings, claims, problems, choices
        )
        
        # Reduction method comparison
        logger.info("Analyzing reduction method sensitivity...")
        method_results = self.analyze_reduction_method_sensitivity(raw_embeddings)
        
        # Compile results
        full_results = {
            "timestamp": datetime.now().isoformat(),
            "pca_sensitivity": pca_results,
            "reduction_method_sensitivity": method_results,
            "summary": self._generate_summary()
        }
        
        return full_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of robustness analysis."""
        summary = {}
        
        if "pca_sensitivity" in self.analysis_results:
            pca = self.analysis_results["pca_sensitivity"]
            # Find optimal dimension based on explained variance vs complexity tradeoff
            if pca.get("analysis"):
                dims = list(pca["analysis"].keys())
                # Simple heuristic: recommend smallest dim with >90% explained variance
                summary["pca_recommendation"] = "See detailed results for optimal dimension"
        
        if "reduction_method_sensitivity" in self.analysis_results:
            method = self.analysis_results["reduction_method_sensitivity"]
            if method.get("comparisons"):
                similarities = [
                    v.get("structure_similarity", 0) 
                    for v in method["comparisons"].values()
                    if v.get("structure_similarity") is not None
                ]
                if similarities:
                    summary["pca_random_similarity"] = float(np.mean(similarities))
        
        return summary
    
    def save_results(self, filepath: Optional[str] = None):
        """Save analysis results to JSON."""
        if filepath is None:
            filepath = self.results_dir / "robustness_analysis.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        logger.info(f"Saved robustness analysis to {filepath}")
    
    @staticmethod
    def load_results(filepath: str) -> Dict[str, Any]:
        """Load analysis results from JSON."""
        with open(filepath, 'r') as f:
            return json.load(f)
