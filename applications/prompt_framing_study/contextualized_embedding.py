"""
Contextualized Embedding Module for Prompt Framing Study

Generates embeddings for claims within the full prompt context.
Key insight: Same claim gets DIFFERENT embeddings under different prompt variants.
"""
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import os
import logging

from .prompt_manager import PromptVariant

logger = logging.getLogger(__name__)


class ContextualizedEmbeddingManager:
    """
    Generate contextualized embeddings for claims.
    
    Each claim is embedded within the prompt context, so the embedding
    captures how the claim is perceived given the decision-making framing.
    This means we get DIFFERENT embeddings for the same claim under
    different prompt variants.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """
        Initialize the embedding manager.
        
        Args:
            embedding_model: OpenAI embedding model to use
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.embedding_model = embedding_model
        self.embedding_cache: Dict[Tuple, np.ndarray] = {}
        
        import openai
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        # Track token usage for cost estimation
        self.total_tokens = 0
    
    def embed_claims_for_variant(
        self,
        claims: List[Dict[str, str]],
        variant: PromptVariant
    ) -> Dict[str, np.ndarray]:
        """
        Generate contextualized embeddings for all claims under a specific prompt variant.
        
        Args:
            claims: List of claim dicts with 'id' and 'description' keys
            variant: The prompt variant providing the context
            
        Returns:
            Dict mapping claim_id -> embedding vector
        """
        embeddings = {}
        texts_to_embed = []
        claim_ids = []
        
        for claim in claims:
            # Create the contextualized prompt for this claim
            contextualized_text = variant.format_embedding_prompt(claim["description"])
            
            # Check cache
            cache_key = (variant.name, claim["id"], self.embedding_model)
            if cache_key in self.embedding_cache:
                embeddings[claim["id"]] = self.embedding_cache[cache_key]
            else:
                texts_to_embed.append(contextualized_text)
                claim_ids.append(claim["id"])
        
        # Batch embed uncached texts
        if texts_to_embed:
            logger.info(f"Embedding {len(texts_to_embed)} claims for variant '{variant.name}'")
            new_embeddings = self._get_embeddings_batch(texts_to_embed)
            for claim_id, emb in zip(claim_ids, new_embeddings):
                cache_key = (variant.name, claim_id, self.embedding_model)
                self.embedding_cache[cache_key] = emb
                embeddings[claim_id] = emb
        else:
            logger.debug(f"All embeddings for variant '{variant.name}' were cached")
        
        return embeddings
    
    def embed_all_variants(
        self,
        claims: List[Dict[str, str]],
        variants: List[PromptVariant]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate embeddings for all claims under all variants.
        
        Args:
            claims: List of claim dicts with 'id' and 'description' keys
            variants: List of prompt variants
            
        Returns:
            Dict mapping variant_name -> {claim_id -> embedding}
        """
        all_embeddings = {}
        for variant in variants:
            logger.info(f"Generating embeddings for variant: {variant.name}")
            all_embeddings[variant.name] = self.embed_claims_for_variant(claims, variant)
        return all_embeddings
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings from OpenAI API in batches."""
        all_embeddings = []
        batch_size = 100  # OpenAI limit
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch,
                encoding_format="float"
            )
            
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens += response.usage.total_tokens
            
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def reduce_dimensions(
        self,
        embeddings: Dict[str, np.ndarray],
        target_dim: int,
        method: str = "pca"
    ) -> Tuple[Dict[str, np.ndarray], Any]:
        """
        Reduce embedding dimensions.
        
        Args:
            embeddings: Dict mapping claim_id -> embedding
            target_dim: Target dimensionality
            method: Reduction method ("pca" or "random")
            
        Returns:
            Tuple of (reduced_embeddings dict, fitted reducer for later use)
        """
        # Stack embeddings into matrix
        claim_ids = list(embeddings.keys())
        X = np.vstack([embeddings[cid] for cid in claim_ids])
        
        if method == "pca":
            from sklearn.decomposition import PCA
            # Handle case where target_dim > n_samples or n_features
            actual_dim = min(target_dim, X.shape[0], X.shape[1])
            if actual_dim < target_dim:
                logger.warning(
                    f"Reducing target_dim from {target_dim} to {actual_dim} "
                    f"(limited by data shape {X.shape})"
                )
            reducer = PCA(n_components=actual_dim)
            X_reduced = reducer.fit_transform(X)
            logger.info(
                f"PCA reduction: {X.shape[1]} -> {actual_dim} dims "
                f"(explained variance: {reducer.explained_variance_ratio_.sum():.3f})"
            )
        elif method == "random":
            # Random projection (deterministic with seed)
            np.random.seed(42)
            proj_matrix = np.random.normal(size=(X.shape[1], target_dim))
            proj_matrix /= np.sqrt(np.sum(proj_matrix**2, axis=0))
            X_reduced = X @ proj_matrix
            reducer = proj_matrix  # Store for consistency
            logger.info(f"Random projection: {X.shape[1]} -> {target_dim} dims")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'random'.")
        
        reduced_embeddings = {cid: X_reduced[i] for i, cid in enumerate(claim_ids)}
        return reduced_embeddings, reducer
    
    def apply_reducer(
        self,
        embeddings: Dict[str, np.ndarray],
        reducer: Any,
        method: str = "pca"
    ) -> Dict[str, np.ndarray]:
        """
        Apply a previously fitted reducer to new embeddings.
        
        Args:
            embeddings: Dict mapping claim_id -> embedding
            reducer: Fitted reducer (PCA object or projection matrix)
            method: Reduction method used
            
        Returns:
            Dict of reduced embeddings
        """
        claim_ids = list(embeddings.keys())
        X = np.vstack([embeddings[cid] for cid in claim_ids])
        
        if method == "pca":
            X_reduced = reducer.transform(X)
        elif method == "random":
            X_reduced = X @ reducer
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {cid: X_reduced[i] for i, cid in enumerate(claim_ids)}
    
    def save_embeddings(
        self,
        embeddings: Dict[str, Dict[str, np.ndarray]],
        filepath: str
    ):
        """
        Save all embeddings to npz file.
        
        Args:
            embeddings: Dict mapping variant_name -> {claim_id -> embedding}
            filepath: Path to save the npz file
        """
        save_dict = {}
        for variant_name, variant_embeddings in embeddings.items():
            for claim_id, emb in variant_embeddings.items():
                key = f"{variant_name}__{claim_id}"
                save_dict[key] = emb
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        np.savez(filepath, **save_dict)
        logger.info(f"Saved embeddings to {filepath}")
    
    @staticmethod
    def load_embeddings(filepath: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load embeddings from npz file.
        
        Args:
            filepath: Path to the npz file
            
        Returns:
            Dict mapping variant_name -> {claim_id -> embedding}
        """
        data = np.load(filepath)
        embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        
        for key in data.files:
            variant_name, claim_id = key.split("__", 1)
            if variant_name not in embeddings:
                embeddings[variant_name] = {}
            embeddings[variant_name][claim_id] = data[key]
        
        logger.info(f"Loaded embeddings from {filepath}")
        return embeddings
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of embedding API usage."""
        return {
            "model": self.embedding_model,
            "total_tokens": self.total_tokens,
            "cache_size": len(self.embedding_cache)
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.debug("Embedding cache cleared")
