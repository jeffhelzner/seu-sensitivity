"""
LLM Rationality Benchmark Runner

This script coordinates the complete benchmarking process to assess LLM rationality
through decision problems and Stan modeling.

Purpose:
--------
Evaluate how "rational" different Large Language Models (LLMs) are by:
1. Presenting them with a set of text-based decision problems
2. Collecting their choices for each problem
3. Generating embeddings for the underlying claims
4. Creating Stan data packages for rationality parameter estimation

Usage:
------
# Basic usage - run with default settings
python applications/llm_rationality/run_benchmark.py

# Environment setup required:
# - OpenAI API key in .env file or environment variables
# - Python packages: openai, python-dotenv, numpy, scikit-learn (optional)

Workflow:
---------
1. Loads decision problems from property_problems.json
2. Initializes OpenAI clients for different models (GPT-4, GPT-3.5-Turbo, etc.)
3. For each model:
   - Presents each decision problem and collects choices
   - Records which alternative was selected
4. Generates embeddings for all unique claims using ClaimEmbeddingManager
5. Creates and saves:
   - Raw choice data (raw_choices.json)
   - Claim embeddings (embeddings.npz)
   - Stan data packages for each model (stan_data_*.json)
   - Run metadata (run_metadata.json)

All results are saved in a timestamped subfolder under results/run_YYYYMMDD_HHMMSS/

Output:
-------
- raw_choices.json: Raw choices from all models
- embeddings.npz: Embedded representations of claims
- stan_data_GPT-4.json: (and similar files) Data ready for Stan modeling
- run_metadata.json: Summary information about the benchmark run

Parameters:
-----------
Key parameters that can be modified in the script:
- EMBEDDING_DIM: Target dimensionality for claim embeddings (default: 20)
- K: Number of consequences in the Stan model (default: 3)
- LLM models and their parameters (temperature, etc.)

See README.md for detailed information about the full LLM rationality benchmarking process.
"""
import os
import numpy as np
import sys
from pathlib import Path
import json
from datetime import datetime

# Set base directory and add to path to ensure imports work
base_dir = Path(__file__).parent
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import directly from the files rather than through the package structure
# This avoids triggering imports in __init__.py that require sentence_transformers
sys.path.append(str(base_dir))
from llm_client import OpenAIClient
from claim_embedding import ClaimEmbeddingManager

# Define function to collect choices without needing feature data
def collect_raw_choices(llm_client, problems):
    """
    Collect choices from an LLM for a set of decision problems.
    
    Args:
        llm_client: LLM client to use
        problems: List of decision problems
        
    Returns:
        Dictionary with choices and metadata
    """
    choices = []
    raw_responses = []
    
    for i, problem in enumerate(problems):
        print(f"Processing problem {i+1}/{len(problems)}")
        context = problem["context"]
        alternatives = problem["alternatives"]
        
        # Get LLM choice
        choice_idx = llm_client.make_choice(context, alternatives)
        
        # Convert to 1-indexed for Stan (categorical distribution in Stan is 1-indexed)
        choices.append(choice_idx + 1)
        
        # Store some metadata about the choice
        raw_responses.append({
            "problem_id": problem.get("id", f"P{i+1}"),
            "choice_index": choice_idx,
            "choice_index_1based": choice_idx + 1,
            "chosen_alternative": alternatives[choice_idx]
        })
    
    return {
        "choices": choices,
        "raw_responses": raw_responses,
        "timestamp": datetime.now().isoformat()
    }

def main():
    print("Loading property problems...")
    # Load problems
    with open(base_dir / "problems/property_problems.json", 'r') as f:
        data = json.load(f)
        problems = data["problems"]
    
    print(f"Found {len(problems)} problems")
    
    # Initialize model clients
    print("Initializing OpenAI clients...")
    llm_clients = {
        "GPT-4": OpenAIClient(model="gpt-4", temperature=0.0),
        "GPT-3.5-Turbo": OpenAIClient(model="gpt-3.5-turbo", temperature=0.0),
        "GPT-4-Turbo": OpenAIClient(model="gpt-4-turbo", temperature=0.0)
    }
    
    # Collect choices for each model
    results = {}
    
    # Collect choices first (no embeddings needed)
    for model_name, client in llm_clients.items():
        print(f"\nCollecting choices from {model_name}...")
        model_results = collect_raw_choices(client, problems)
        results[model_name] = model_results
        print(f"Choices from {model_name}: {model_results['choices']}")
    
    # Initialize embedding manager with base claims
    embedding_manager = ClaimEmbeddingManager(
        base_dir / "data/property_claims.json",
        embedding_model="text-embedding-3-small",
        target_dim=20  # Explicitly reduce to 20 dimensions
    )
    
    # Generate embeddings for base claims (once)
    claim_embeddings = embedding_manager.generate_embeddings()
    
    # Create timestamp for consistent filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create run-specific subfolder
    results_dir = base_dir / "results"
    run_dir = results_dir / f"run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\nSaving all results to: {run_dir}")
    
    # Save raw choices
    choices_file = run_dir / "raw_choices.json"
    with open(choices_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Raw choices saved to {choices_file}")
    
    # Save embeddings
    embeddings_file = run_dir / "embeddings.npz"
    np.savez(
        embeddings_file,
        claim_ids=list(claim_embeddings.keys()),
        embeddings=np.array(list(claim_embeddings.values()))
    )
    print(f"Embeddings saved to {embeddings_file}")
    
    # Extract K from property_claims.json
    with open(base_dir / "data/property_claims.json", 'r') as f:
        claims_data = json.load(f)
        consequences = claims_data.get("consequences", ["bad", "neutral", "good"])
        K = len(consequences)
    
    print(f"Using K={K} consequences from property_claims.json: {consequences}")
    
    # Create and save Stan data packages using base claim embeddings
    for model_name, result in results.items():
        # Use a safe filename (replace spaces and special chars)
        safe_model_name = model_name.replace(" ", "-").replace(".", "")
        
        stan_data = embedding_manager.create_stan_data(
            base_dir / "problems/property_problems.json",
            result["choices"],
            K=K  # Pass the dynamically determined K
        )
        
        # Save Stan data
        stan_data_file = run_dir / f"stan_data_{safe_model_name}.json"
        with open(stan_data_file, 'w') as f:
            json.dump(stan_data, f, indent=2)
        print(f"Stan data for {model_name} saved to {stan_data_file}")
    
    # Save run metadata
    metadata = {
        "timestamp": timestamp,
        "num_problems": len(problems),
        "models": list(results.keys()),
        "embedding_model": embedding_manager.embedding_model,  # Use the actual model name
        "embedding_dimensions": next(iter(claim_embeddings.values())).shape[0],
        "run_completed": datetime.now().isoformat()
    }
    
    metadata_file = run_dir / "run_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nRun completed and saved to {run_dir}")

if __name__ == "__main__":
    main()