"""
Module for embedding base claims and creating Stan data packages.
"""
import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import openai  # Add this import

# Make dotenv optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Continue without dotenv
    pass

class ClaimEmbeddingManager:
    """Manages embeddings for base claims and creates Stan data packages."""
    
    def __init__(self, 
                 claims_file_path: str,
                 embedding_model: str = "text-embedding-3-small",
                 target_dim: Optional[int] = None):
        """
        Initialize with base claims file.
        
        Args:
            claims_file_path: Path to the claims JSON file
            embedding_model: OpenAI model to use for embeddings
            target_dim: Target dimensionality after reduction (optional)
        """
        self.claims_file_path = claims_file_path
        self.embedding_model = embedding_model
        self.target_dim = target_dim
        self.claim_embeddings = {}  # Will store embeddings keyed by claim ID
        
        # Load claims
        with open(claims_file_path, 'r') as f:
            data = json.load(f)
            
        if "claims" in data:
            self.claims = data["claims"]
            self.context = data.get("context", "")
        else:
            self.claims = data
            self.context = ""
            
        # Map claim IDs to indices (position in the claims list)
        self.claim_id_to_index = {claim["id"]: i for i, claim in enumerate(self.claims)}
        
    def generate_embeddings(self):
        """Generate embeddings for all base claims."""
        # Extract texts to embed
        texts = [claim["description"] for claim in self.claims]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} base claims...")
        embeddings = self._get_embeddings(texts)
        
        # Apply dimension reduction if requested
        # FIX: Use len() instead of .size and convert to numpy array first
        if self.target_dim and len(embeddings[0]) > self.target_dim:
            print(f"Reducing dimensions from {len(embeddings[0])} to {self.target_dim}...")
            # Convert to numpy array for dimension reduction
            embeddings_array = np.array(embeddings)
            embeddings = self._reduce_dimensions(embeddings_array)
        
        # Store embeddings by claim ID
        for i, claim in enumerate(self.claims):
            self.claim_embeddings[claim["id"]] = embeddings[i]
            
        return self.claim_embeddings
        
    def _get_embeddings(self, texts):
        """Get embeddings from OpenAI API."""
        try:
            # Use openai directly (already imported at the top)
            client = openai.OpenAI()
            
            # Process in batches
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                response = client.embeddings.create(
                    model=self.embedding_model,
                    input=batch,
                    encoding_format="float"
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            return all_embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
            
    def _reduce_dimensions(self, embeddings):
        """Reduce embedding dimensions using PCA."""
        # Cap target_dim to the number of samples if needed
        effective_target_dim = min(self.target_dim, embeddings.shape[0])
        if effective_target_dim < self.target_dim:
            print(f"Warning: Reduced target dimensions to {effective_target_dim} due to sample count limitations")
        
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=effective_target_dim)
            return pca.fit_transform(embeddings)
        except ImportError:
            print("Warning: scikit-learn not installed. Using random projection.")
            # Simple random projection
            np.random.seed(42)  # For reproducibility
            projection_matrix = np.random.normal(
                size=(embeddings.shape[1], effective_target_dim)
            )
            # Normalize columns
            projection_matrix /= np.sqrt(np.sum(projection_matrix**2, axis=0))
            return embeddings @ projection_matrix
            
    def create_stan_data(self, problems_file_path, choices, K=None):
        """
        Create Stan data package using base claim embeddings and choices.
        
        Args:
            problems_file_path: Path to problems JSON file
            choices: List of choices (1-indexed)
            K: Number of consequences (optional, will be extracted from claims file if not provided)
            
        Returns:
            Dictionary with Stan data
        """
        # Load problems
        with open(problems_file_path, 'r') as f:
            data = json.load(f)
            problems = data["problems"]
    
        # If K is not provided, extract it from claims file
        if K is None:
            # Look for consequences field in the original claims file
            claims_data = json.load(open(self.claims_file_path, 'r'))
            consequences = claims_data.get("consequences", ["bad", "neutral", "good"])
            K = len(consequences)
            print(f"Using K={K} based on consequences: {consequences}")
        
        # Ensure embeddings are generated
        if not self.claim_embeddings:
            self.generate_embeddings()
            
        # Extract unique claim IDs used in problems
        used_claim_ids = set()
        for problem in problems:
            # Access claim_ids directly from the metadata
            for claim_id in problem["metadata"]["claim_ids"]:
                used_claim_ids.add(claim_id)
                
        # Create mapping from used claim IDs to positions in w array
        claim_id_to_pos = {claim_id: i for i, claim_id in enumerate(sorted(used_claim_ids))}
        
        # Prepare w array with embeddings
        R = len(used_claim_ids)
        D = len(next(iter(self.claim_embeddings.values())))
        w = np.zeros((R, D))
        
        for claim_id, pos in claim_id_to_pos.items():
            w[pos] = self.claim_embeddings[claim_id]
            
        # Create indicator matrix
        M = len(problems)
        I = np.zeros((M, R), dtype=int)
        
        for m, problem in enumerate(problems):
            for claim_id in problem["metadata"]["claim_ids"]:
                r = claim_id_to_pos[claim_id]
                I[m, r] = 1
                
        # Create Stan data package
        stan_data = {
            "M": M,
            "K": K,
            "D": D,
            "R": R,
            "w": w.tolist(),
            "I": I.tolist(),
            "y": choices
        }
        
        return stan_data
    
    # Fix save_stan_data method
    def save_stan_data(self, data_package, filepath):
        """Save Stan data package to file."""
        with open(filepath, 'w') as f:
            json.dump(data_package, f, indent=2)
        print(f"Stan data saved to {filepath}")