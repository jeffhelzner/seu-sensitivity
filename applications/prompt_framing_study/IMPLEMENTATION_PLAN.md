# Implementation Plan: Prompt Framing Study Module

## Overview

This document specifies the implementation plan for a new `prompt_framing_study` module that investigates how prompt framing (specifically, rationality emphasis) affects an LLM's sensitivity to subjective expected utility maximization.

**Location**: `applications/prompt_framing_study/`

**Key Innovation**: Rather than simple claim embeddings, we use **contextualized embeddings** where each claim is embedded within the full prompt context, capturing how the LLM "perceives" the claim given the decision-making framing.

**Standalone Module**: This module is self-contained and does not depend on the legacy `llm_rationality` module. It incorporates improved versions of shared functionality (LLM client, problem generation, embedding) directly.

---

## Research Design Summary

### Research Question
How does increasing or decreasing the emphasis on rationality in the prompt affect an LLM's sensitivity parameter (α) when making SEU-based decisions?

### Experimental Design
- **Independent Variable**: Prompt rationality emphasis (4 levels)
- **Dependent Variable**: Estimated α from m_0 model
- **Robustness Checks**: Multiple embedding models, multiple PCA dimensions

### Fixed Parameters
- K = 3 consequences (both judges agree / one agrees / neither agrees)
- Insurance claims triage domain

---

## Directory Structure

```
applications/prompt_framing_study/
├── __init__.py
├── README.md
├── llm_client.py              # LLM API interfaces (standalone)
├── prompt_manager.py          # Prompt variant management
├── contextualized_embedding.py # Embedding with full prompt context
├── choice_collector.py        # LLM choice collection  
├── problem_generator.py       # Decision problem generation
├── study_runner.py            # Orchestrates full study pipeline
├── robustness_analysis.py     # Embedding model & PCA sensitivity
├── visualization.py           # Result visualization
├── cost_estimator.py          # API cost estimation utilities
├── validation.py              # Data and schema validation
├── cli.py                     # Command-line interface
├── configs/
│   ├── study_config.yaml      # Main study configuration
│   ├── prompt_variants.yaml   # Prompt template definitions
│   └── embedding_config.yaml  # Embedding model settings
├── data/
│   └── claims.json            # Base claim descriptions
├── results/                   # Study outputs (gitignored)
└── tests/
    ├── __init__.py
    ├── test_pipeline.py       # Unit tests
    ├── test_validation.py     # Validation tests
    └── conftest.py            # Pytest fixtures
```

---

## Module Specifications

### 0. `__init__.py`

**Purpose**: Define public API and configure logging.

```python
"""
Prompt Framing Study Module

Investigates how prompt framing affects LLM sensitivity to 
subjective expected utility maximization.
"""
import logging

# Configure module-level logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public API
from .prompt_manager import PromptManager, PromptVariant
from .contextualized_embedding import ContextualizedEmbeddingManager
from .choice_collector import ChoiceCollector
from .problem_generator import ProblemGenerator
from .study_runner import StudyRunner
from .robustness_analysis import RobustnessAnalyzer
from .visualization import StudyVisualizer
from .cost_estimator import CostEstimator
from .validation import validate_stan_data, validate_config

__all__ = [
    "PromptManager",
    "PromptVariant", 
    "ContextualizedEmbeddingManager",
    "ChoiceCollector",
    "ProblemGenerator",
    "StudyRunner",
    "RobustnessAnalyzer",
    "StudyVisualizer",
    "CostEstimator",
    "validate_stan_data",
    "validate_config",
]

__version__ = "0.1.0"
```

---

### 0.5 `llm_client.py`

**Purpose**: Standalone LLM client interfaces with no legacy dependencies.

```python
"""
LLM Client Module for Prompt Framing Study

Standalone LLM client interfaces - no dependency on legacy modules.
"""
from typing import List, Optional, Dict, Any
import os
import re
import time
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """Base class for LLM clients."""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the LLM."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def make_choice(self, context: str, alternatives: List[str], **kwargs) -> int:
        """
        Present a decision problem and get choice.
        
        Returns:
            Index of chosen alternative (0-indexed)
        """
        prompt = f"{context}\n\nPlease choose one of the following options:\n"
        for i, alt in enumerate(alternatives):
            prompt += f"{i+1}. {alt}\n"
        prompt += f"\nIMPORTANT: Respond with ONLY the option number (1-{len(alternatives)})."
        
        response = self.generate(prompt, **kwargs)
        return self._parse_choice(response, len(alternatives))
    
    def make_choice_with_prompt(self, prompt: str, num_alternatives: int, **kwargs) -> int:
        """
        Make a choice given a fully-formed prompt.
        
        Use this when the prompt already contains all framing and alternatives.
        
        Args:
            prompt: Complete prompt including alternatives
            num_alternatives: Number of alternatives to expect
            
        Returns:
            Index of chosen alternative (0-indexed)
        """
        response = self.generate(prompt, **kwargs)
        return self._parse_choice(response, num_alternatives)
    
    def _parse_choice(self, response: str, num_alternatives: int) -> int:
        """Parse a choice from LLM response."""
        choice_patterns = [
            r'[Cc]laim\s*(\d+)',                 # "Claim 1" or "claim 2"
            r'(\d+)\s*$',                         # Number at end
            r'^[^\d]*?(\d+)[^\d]*$',              # Only one number in response
            r'(?:option|choice|select|choose|pick|answer)[^\d]*?(\d+)',
            r'(\d+)'                              # Any number as fallback
        ]
        
        for pattern in choice_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                choice = int(match.group(1)) - 1
                if 0 <= choice < num_alternatives:
                    logger.debug(f"Parsed choice {choice+1} from response: {response[:50]}...")
                    return choice
        
        # Fallback: find any valid number
        for num in re.findall(r'\d+', response):
            choice = int(num) - 1
            if 0 <= choice < num_alternatives:
                logger.warning(f"Used fallback parsing for choice {choice+1}")
                return choice
        
        logger.warning(f"Could not parse choice from: {response}. Defaulting to 0.")
        return 0


class OpenAIClient(LLMClient):
    """OpenAI API client with retry logic and cost tracking."""
    
    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    
    def __init__(self, 
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None, 
                 temperature: float = 0.0,
                 max_retries: int = 3,
                 retry_delay: float = 2.0,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Track usage for cost estimation
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        import openai
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI's chat API with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", self.temperature),
                )
                
                # Track token usage
                if response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_error = e
                logger.warning(f"API call failed (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")
    
    def get_estimated_cost(self) -> float:
        """Get estimated cost in USD based on token usage."""
        pricing = self.PRICING.get(self.model, {"input": 0, "output": 0})
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def reset_usage(self):
        """Reset token usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
```

---

### 1. `prompt_manager.py`

**Purpose**: Manage prompt templates with varying rationality emphasis levels.

**Classes**:

```python
class PromptVariant:
    """Represents a single prompt variant with metadata."""
    
    def __init__(self, 
                 name: str,
                 level: int,  # 0=minimal, 1=baseline, 2=enhanced, 3=maximal
                 template: str,
                 description: str):
        self.name = name
        self.level = level
        self.template = template
        self.description = description
    
    def format_choice_prompt(self, claims: List[str]) -> str:
        """Format the full prompt for choice collection with all claims listed."""
        claims_list = "\n".join([f"- Claim {i+1}: {claim}" for i, claim in enumerate(claims)])
        return self.template.format(claims_list=claims_list)
    
    def format_embedding_prompt(self, claim: str) -> str:
        """
        Format a prompt for embedding a single claim in context.
        Uses the variant's framing but with only one claim.
        """
        ...


class PromptManager:
    """Load and manage prompt variants from configuration."""
    
    def __init__(self, config_path: str = "configs/prompt_variants.yaml"):
        self.variants = self._load_variants(config_path)
    
    def get_variant(self, name: str) -> PromptVariant:
        """Get a specific prompt variant by name."""
        ...
    
    def get_all_variants(self) -> List[PromptVariant]:
        """Get all prompt variants ordered by level."""
        ...
    
    def list_variant_names(self) -> List[str]:
        """Return list of available variant names."""
        ...
```

**Configuration File** (`configs/prompt_variants.yaml`):

```yaml
# Prompt variants with increasing rationality emphasis
# Level 0-3: minimal → baseline → enhanced → maximal

variants:
  - name: "minimal"
    level: 0
    description: "Minimal framing - just pick one, no outcome structure"
    template: |
      You are reviewing flagged insurance claims. Choose one to send for investigation.
      
      {claims_list}
      
      Which claim do you choose? Respond with only the claim number (e.g., "Claim 1").
    
    embedding_template: |
      You are reviewing flagged insurance claims. Choose one to send for investigation.
      
      Claim to evaluate: {claim}

  - name: "baseline"
    level: 1
    description: "Standard framing with K=3 outcome structure"
    template: |
      You are a claims adjuster working for a large insurance company. The claims 
      described below have been flagged as potentially suspicious. You are to choose 
      one of the claims to send to a human investigator for further review.
      
      Your decision will be reviewed by two expert judges:
      - Best outcome: Both judges agree your selection warrants investigation
      - Middle outcome: One judge supports your decision, one doesn't  
      - Worst outcome: Neither judge agrees with your decision
      
      {claims_list}
      
      Which claim do you choose to send to the human investigator for further review? 
      Please respond with only the claim number (e.g., "Claim 1").
    
    embedding_template: |
      You are a claims adjuster working for a large insurance company. A claim has been 
      flagged as potentially suspicious. Your decision to investigate will be reviewed 
      by two expert judges who will evaluate whether the claim warrants investigation.
      
      Claim to evaluate: {claim}

  - name: "enhanced"
    level: 2
    description: "Explicit utility values and rationality language"
    template: |
      You are a claims adjuster working for a large insurance company. The claims 
      described below have been flagged as potentially suspicious. You are to choose 
      one of the claims to send to a human investigator for further review.
      
      Your decision will be reviewed by two expert judges. The possible outcomes are:
      - Best outcome (utility = 1.0): Both judges agree your selection warrants investigation
      - Middle outcome (utility = 0.5): One judge supports your decision, one doesn't
      - Worst outcome (utility = 0.0): Neither judge agrees with your decision
      
      Make your choice to maximize the expected quality of the outcome.
      
      {claims_list}
      
      Which claim do you choose? Respond with only the claim number (e.g., "Claim 1").
    
    embedding_template: |
      You are a claims adjuster making a rational decision about which claims warrant 
      investigation. Your choice will be evaluated by two expert judges, with outcomes 
      ranging from utility 0.0 (neither agrees) to utility 1.0 (both agree).
      
      Claim to evaluate: {claim}

  - name: "maximal"
    level: 3
    description: "Full decision-theoretic framing with explicit EU maximization"
    template: |
      You are a rational decision-making agent optimizing claim investigation decisions.
      
      DECISION FRAMEWORK:
      You must select one claim to investigate. Two independent expert judges will 
      evaluate your selection. Each judge independently decides whether your choice 
      was correct (probability p_i for judge i).
      
      PAYOFF STRUCTURE:
      - Both judges agree (correct): Utility = 1.0
      - One judge agrees: Utility = 0.5  
      - Neither agrees: Utility = 0.0
      
      OBJECTIVE: Maximize expected utility E[U] = p_1*p_2*1.0 + (p_1*(1-p_2) + (1-p_1)*p_2)*0.5
      
      Analyze each claim's likelihood of being judged investigation-worthy, then select 
      the EU-maximizing option.
      
      {claims_list}
      
      Which claim do you choose? Respond with only the claim number (e.g., "Claim 1").
    
    embedding_template: |
      You are a rational decision-making agent evaluating insurance claims. Your task 
      is to maximize expected utility where utility depends on whether expert judges 
      agree that a claim warrants investigation.
      
      Claim to evaluate: {claim}
```

---

### 2. `contextualized_embedding.py`

**Purpose**: Generate embeddings for claims within the full prompt context using OpenAI's embedding API.

**Key Insight**: Instead of embedding just the claim text, we embed the claim within the prompt context to capture how the LLM "perceives" the claim given the decision-making framing.

**Classes**:

```python
class ContextualizedEmbeddingManager:
    """
    Generate contextualized embeddings for claims.
    
    Each claim is embedded within the prompt context, so the embedding
    captures how the claim is perceived given the decision-making framing.
    This means we get DIFFERENT embeddings for the same claim under
    different prompt variants.
    """
    
    def __init__(self,
                 embedding_model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None):
        """
        Initialize the embedding manager.
        
        Args:
            embedding_model: OpenAI embedding model to use
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.embedding_model = embedding_model
        self.client = openai.OpenAI(api_key=api_key)
        self.embedding_cache = {}  # Cache to avoid redundant API calls
    
    def embed_claims_for_variant(self,
                                  claims: List[Dict[str, str]],
                                  variant: PromptVariant) -> Dict[str, np.ndarray]:
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
            new_embeddings = self._get_embeddings_batch(texts_to_embed)
            for claim_id, emb in zip(claim_ids, new_embeddings):
                cache_key = (variant.name, claim_id, self.embedding_model)
                self.embedding_cache[cache_key] = emb
                embeddings[claim_id] = emb
        
        return embeddings
    
    def embed_all_variants(self,
                           claims: List[Dict[str, str]],
                           variants: List[PromptVariant]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate embeddings for all claims under all variants.
        
        Returns:
            Dict mapping variant_name -> {claim_id -> embedding}
        """
        all_embeddings = {}
        for variant in variants:
            print(f"Generating embeddings for variant: {variant.name}")
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
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def reduce_dimensions(self,
                          embeddings: Dict[str, np.ndarray],
                          target_dim: int,
                          method: str = "pca") -> Tuple[Dict[str, np.ndarray], Any]:
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
            reducer = PCA(n_components=min(target_dim, X.shape[0], X.shape[1]))
            X_reduced = reducer.fit_transform(X)
        elif method == "random":
            # Random projection
            np.random.seed(42)
            proj_matrix = np.random.normal(size=(X.shape[1], target_dim))
            proj_matrix /= np.sqrt(np.sum(proj_matrix**2, axis=0))
            X_reduced = X @ proj_matrix
            reducer = proj_matrix  # Store for consistency
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced_embeddings = {cid: X_reduced[i] for i, cid in enumerate(claim_ids)}
        return reduced_embeddings, reducer
    
    def save_embeddings(self, 
                        embeddings: Dict[str, Dict[str, np.ndarray]], 
                        filepath: str):
        """Save all embeddings to npz file."""
        save_dict = {}
        for variant_name, variant_embeddings in embeddings.items():
            for claim_id, emb in variant_embeddings.items():
                key = f"{variant_name}__{claim_id}"
                save_dict[key] = emb
        np.savez(filepath, **save_dict)
    
    @staticmethod
    def load_embeddings(filepath: str) -> Dict[str, Dict[str, np.ndarray]]:
        """Load embeddings from npz file."""
        data = np.load(filepath)
        embeddings = {}
        for key in data.files:
            variant_name, claim_id = key.split("__", 1)
            if variant_name not in embeddings:
                embeddings[variant_name] = {}
            embeddings[variant_name][claim_id] = data[key]
        return embeddings
```

**Configuration File** (`configs/embedding_config.yaml`):

```yaml
# Embedding configuration for robustness analysis

embedding_models:
  - name: "text-embedding-3-small"
    dimensions: 1536
    description: "OpenAI's small embedding model - fast and cheap"
  
  - name: "text-embedding-3-large"
    dimensions: 3072
    description: "OpenAI's large embedding model - higher quality"

dimension_reductions:
  - target_dim: 16
    method: "pca"
  - target_dim: 32
    method: "pca"
  - target_dim: 64
    method: "pca"
  - target_dim: 128
    method: "pca"

default:
  embedding_model: "text-embedding-3-small"
  target_dim: 32
  method: "pca"
```

---

### 3. `choice_collector.py`

**Purpose**: Collect LLM choices for decision problems under different prompt variants.

**Note**: Uses the module's own `llm_client.py` (no legacy dependency).

```python
from .llm_client import OpenAIClient


class ChoiceCollector:
    """
    Collect choices from LLMs for decision problems under different prompt variants.
    """
    
    def __init__(self,
                 llm_model: str = "gpt-4",
                 temperature: float = 0.7,  # Non-zero for variability
                 api_key: Optional[str] = None):
        """
        Initialize the choice collector.
        
        Args:
            llm_model: OpenAI model for choice generation
            temperature: Sampling temperature (>0 recommended for choice variability)
            api_key: OpenAI API key
        """
        self.client = OpenAIClient(model=llm_model, temperature=temperature, api_key=api_key)
        self.llm_model = llm_model
        self.temperature = temperature
    
    def collect_choices(self,
                        problems: List[Dict],
                        variant: PromptVariant,
                        num_repetitions: int = 1) -> Dict[str, Any]:
        """
        Collect choices for all problems under a specific prompt variant.
        
        Args:
            problems: List of decision problems
            variant: Prompt variant to use
            num_repetitions: Number of times to present each problem (for T>0)
            
        Returns:
            Dict with choices and metadata
        """
        all_choices = []
        all_responses = []
        
        for rep in range(num_repetitions):
            rep_choices = []
            rep_responses = []
            
            for i, problem in enumerate(problems):
                # Get claim descriptions for this problem
                claims = problem["alternatives"]
                
                # Format prompt using variant
                prompt = variant.format_choice_prompt(claims)
                
                # Get LLM choice using prompt_override
                choice_idx = self.client.make_choice_with_prompt(prompt, len(claims))
                
                rep_choices.append(choice_idx + 1)  # 1-indexed for Stan
                rep_responses.append({
                    "problem_id": problem.get("id", f"P{i+1}"),
                    "repetition": rep,
                    "choice_1indexed": choice_idx + 1,
                    "chosen_claim_id": problem["metadata"]["claim_ids"][choice_idx]
                })
            
            all_choices.append(rep_choices)
            all_responses.append(rep_responses)
        
        return {
            "variant": variant.name,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "num_problems": len(problems),
            "num_repetitions": num_repetitions,
            "choices": all_choices,  # List of lists if num_repetitions > 1
            "responses": all_responses,
            "timestamp": datetime.now().isoformat()
        }
    
    def collect_all_variants(self,
                             problems: List[Dict],
                             variants: List[PromptVariant],
                             num_repetitions: int = 1) -> Dict[str, Dict]:
        """
        Collect choices for all problems under all variants.
        
        Returns:
            Dict mapping variant_name -> choice results
        """
        results = {}
        for variant in variants:
            print(f"Collecting choices for variant: {variant.name}")
            results[variant.name] = self.collect_choices(problems, variant, num_repetitions)
        return results
```

**Note**: The `OpenAIClient.make_choice` method in the legacy module may need a small modification to accept a `prompt_override` parameter. If not, we can add a new method or modify the call pattern.

---

### 4. `problem_generator.py`

**Purpose**: Generate decision problems from base claims.

**Reuses**: Pattern from `applications/llm_rationality/claim_design.py`

```python
class ProblemGenerator:
    """
    Generate decision problems by sampling from base claims.
    
    Adapted from legacy ClaimDesignGenerator with improvements.
    """
    
    def __init__(self, claims_file: str):
        """Load claims from file."""
        with open(claims_file, 'r') as f:
            data = json.load(f)
        
        self.claims = data["claims"]
        self.consequences = data.get("consequences", ["bad", "neutral", "good"])
        self.K = len(self.consequences)
    
    def generate_problems(self,
                          num_problems: int = 100,
                          min_alternatives: int = 2,
                          max_alternatives: int = 4,
                          seed: Optional[int] = None) -> List[Dict]:
        """
        Generate decision problems.
        
        Args:
            num_problems: Number of problems to generate
            min_alternatives: Minimum claims per problem
            max_alternatives: Maximum claims per problem
            seed: Random seed for reproducibility
            
        Returns:
            List of decision problem dicts
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        problems = []
        for i in range(num_problems):
            n_alts = random.randint(min_alternatives, min(max_alternatives, len(self.claims)))
            selected = random.sample(self.claims, n_alts)
            
            problems.append({
                "id": f"P{i+1:04d}",
                "alternatives": [c["description"] for c in selected],
                "metadata": {
                    "claim_ids": [c["id"] for c in selected],
                    "num_alternatives": n_alts
                }
            })
        
        return problems
    
    def save_problems(self, problems: List[Dict], filepath: str):
        """Save problems to JSON file."""
        output = {
            "problems": problems,
            "K": self.K,
            "consequences": self.consequences,
            "metadata": {
                "num_problems": len(problems),
                "generated_at": datetime.now().isoformat()
            }
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
    
    @staticmethod
    def load_problems(filepath: str) -> Tuple[List[Dict], int]:
        """Load problems from file. Returns (problems, K)."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data["problems"], data.get("K", 3)
```

---

### 5. `study_runner.py`

**Purpose**: Orchestrate the full study pipeline.

```python
class StudyRunner:
    """
    Orchestrate the complete study pipeline with checkpointing support.
    
    Pipeline:
    1. Load configuration
    2. Generate problems (or load existing)
    3. Generate embeddings for all variants
    4. Collect choices for all variants
    5. Prepare Stan data packages
    6. Fit m_0 model for each variant
    7. Save results and generate visualizations
    """
    
    def __init__(self, config_path: str = "configs/study_config.yaml"):
        """Load study configuration."""
        import yaml
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate configuration
        from .validation import validate_config
        warnings = validate_config(self.config)
        for w in warnings:
            logger.warning(w)
        
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "results" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file for resumption
        self.checkpoint_file = self.results_dir / ".checkpoint.json"
        self.checkpoint = self._load_checkpoint()
        
        # Setup logging to file
        self._setup_file_logging()
        
        # Initialize components
        self.prompt_manager = PromptManager(self.base_dir / "configs/prompt_variants.yaml")
        self.embedding_manager = ContextualizedEmbeddingManager(
            embedding_model=self.config.get("embedding_model", "text-embedding-3-small")
        )
        self.choice_collector = ChoiceCollector(
            llm_model=self.config.get("llm_model", "gpt-4"),
            temperature=self.config.get("temperature", 0.7)
        )
    
    def _setup_file_logging(self):
        """Configure logging to file in results directory."""
        log_file = self.results_dir / "study.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logger.info(f"Logging to {log_file}")
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Resuming from checkpoint: completed steps = {checkpoint.get('completed_steps', [])}")
            return checkpoint
        return {"completed_steps": []}
    
    def _save_checkpoint(self, step: str, data: Optional[Dict] = None):
        """Save checkpoint after completing a step."""
        self.checkpoint["completed_steps"].append(step)
        if data:
            self.checkpoint[step] = data
        self.checkpoint["last_updated"] = datetime.now().isoformat()
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2, default=str)
        
        logger.debug(f"Checkpoint saved: {step}")
    
    def _step_completed(self, step: str) -> bool:
        """Check if a step was already completed."""
        return step in self.checkpoint.get("completed_steps", [])
    
    def run(self) -> Dict[str, Any]:
        """Execute the full study pipeline with checkpointing."""
        results = {"config": self.config, "timestamp": datetime.now().isoformat()}
        
        try:
            # Step 1: Load or generate problems
            if not self._step_completed("problems"):
                logger.info("Step 1: Preparing decision problems...")
                problems = self._prepare_problems()
                self._save_checkpoint("problems", {"num_problems": len(problems)})
            else:
                logger.info("Step 1: Loading problems from checkpoint...")
                problems, _ = ProblemGenerator.load_problems(self.results_dir / "problems.json")
            results["num_problems"] = len(problems)
            
            # Step 2: Load claims for embedding
            logger.info("Step 2: Loading claims...")
            claims = self._load_claims()
            results["num_claims"] = len(claims)
            
            # Step 3: Get prompt variants
            logger.info("Step 3: Loading prompt variants...")
            variants = self.prompt_manager.get_all_variants()
            results["variants"] = [v.name for v in variants]
            
            # Step 4: Generate embeddings for all variants
            logger.info("Step 4: Generating contextualized embeddings...")
            all_embeddings = self.embedding_manager.embed_all_variants(claims, variants)
            self.embedding_manager.save_embeddings(
                all_embeddings, 
                self.results_dir / "raw_embeddings.npz"
            )
            self._save_checkpoint("embeddings")
            
            # Step 5: Apply dimension reduction
            logger.info("Step 5: Reducing embedding dimensions...")
            target_dim = self.config.get("target_dim", 32)
            reduced_embeddings = {}
            for variant_name, embs in all_embeddings.items():
                reduced_embeddings[variant_name], _ = self.embedding_manager.reduce_dimensions(
                    embs, target_dim=target_dim
                )
            
            # Step 6: Collect choices for all variants
            logger.info("Step 6: Collecting LLM choices...")
            num_reps = self.config.get("num_repetitions", 1)
            all_choices = self.choice_collector.collect_all_variants(problems, variants, num_reps)
            
            # Save raw choices
            with open(self.results_dir / "raw_choices.json", 'w') as f:
                json.dump(all_choices, f, indent=2)
            self._save_checkpoint("choices")
            
            # Step 7: Prepare Stan data packages
            logger.info("Step 7: Preparing Stan data packages...")
            stan_data_packages = {}
            for variant in variants:
                stan_data = self._create_stan_data(
                    problems=problems,
                    embeddings=reduced_embeddings[variant.name],
                    choices=all_choices[variant.name]["choices"][0],  # First repetition
                    K=self.config.get("K", 3)
                )
                stan_data_packages[variant.name] = stan_data
                
                # Save each package
                with open(self.results_dir / f"stan_data_{variant.name}.json", 'w') as f:
                    json.dump(stan_data, f, indent=2)
            
            self._save_checkpoint("stan_data")
            
            # Step 8: Fit models (optional, can be done separately)
            if self.config.get("fit_models", False):
                logger.info("Step 8: Fitting m_0 models...")
                results["model_fits"] = self._fit_models(stan_data_packages)
            
            # Step 9: Save metadata
            logger.info("Step 9: Saving results...")
            results["results_dir"] = str(self.results_dir)
            with open(self.results_dir / "run_metadata.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Study complete! Results saved to {self.results_dir}")
            return results
        
        except Exception as e:
            logger.error(f"Study failed at step: {e}", exc_info=True)
            raise
        
        # Clear checkpoint on successful completion
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
```

---

### NEW: `cost_estimator.py`

**Purpose**: Estimate and track API costs before and during study execution.

```python
"""
Cost estimation utilities for the prompt framing study.
"""
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class CostEstimator:
    """
    Estimate API costs for study execution.
    
    Helps users understand costs before running expensive studies.
    """
    
    # OpenAI embedding pricing per 1M tokens
    EMBEDDING_PRICING = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10,
    }
    
    # OpenAI chat pricing per 1M tokens (input/output)
    CHAT_PRICING = {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    
    # Rough token estimates
    AVG_CLAIM_TOKENS = 150  # Average tokens per claim description
    AVG_PROMPT_OVERHEAD = 200  # Tokens for prompt template
    AVG_RESPONSE_TOKENS = 10  # Short response expected
    
    def __init__(self):
        self.estimates = {}
    
    def estimate_embedding_cost(self,
                                 num_claims: int,
                                 num_variants: int,
                                 embedding_model: str = "text-embedding-3-small") -> Dict:
        """
        Estimate cost for generating contextualized embeddings.
        
        Args:
            num_claims: Number of base claims
            num_variants: Number of prompt variants
            embedding_model: OpenAI embedding model
            
        Returns:
            Dict with token and cost estimates
        """
        # Each claim embedded once per variant
        total_embeddings = num_claims * num_variants
        tokens_per_embedding = self.AVG_CLAIM_TOKENS + self.AVG_PROMPT_OVERHEAD
        total_tokens = total_embeddings * tokens_per_embedding
        
        price_per_million = self.EMBEDDING_PRICING.get(embedding_model, 0.02)
        cost = (total_tokens / 1_000_000) * price_per_million
        
        return {
            "num_embeddings": total_embeddings,
            "estimated_tokens": total_tokens,
            "estimated_cost_usd": cost,
            "model": embedding_model
        }
    
    def estimate_choice_collection_cost(self,
                                         num_problems: int,
                                         num_variants: int,
                                         num_repetitions: int,
                                         avg_alternatives: float,
                                         llm_model: str = "gpt-4") -> Dict:
        """
        Estimate cost for collecting LLM choices.
        
        Args:
            num_problems: Number of decision problems
            num_variants: Number of prompt variants
            num_repetitions: Times each problem presented
            avg_alternatives: Average alternatives per problem
            llm_model: OpenAI chat model
            
        Returns:
            Dict with token and cost estimates
        """
        total_calls = num_problems * num_variants * num_repetitions
        
        # Input tokens: prompt overhead + (alternatives * claim tokens)
        input_per_call = self.AVG_PROMPT_OVERHEAD + (avg_alternatives * self.AVG_CLAIM_TOKENS)
        total_input = total_calls * input_per_call
        total_output = total_calls * self.AVG_RESPONSE_TOKENS
        
        pricing = self.CHAT_PRICING.get(llm_model, {"input": 30.0, "output": 60.0})
        input_cost = (total_input / 1_000_000) * pricing["input"]
        output_cost = (total_output / 1_000_000) * pricing["output"]
        
        return {
            "num_api_calls": total_calls,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost_usd": input_cost + output_cost,
            "model": llm_model
        }
    
    def estimate_total_study_cost(self,
                                   num_claims: int,
                                   num_problems: int,
                                   num_variants: int = 4,
                                   num_repetitions: int = 1,
                                   avg_alternatives: float = 3.0,
                                   embedding_model: str = "text-embedding-3-small",
                                   llm_model: str = "gpt-4") -> Dict:
        """
        Estimate total cost for a complete study run.
        
        Returns comprehensive cost breakdown.
        """
        embedding_est = self.estimate_embedding_cost(
            num_claims, num_variants, embedding_model
        )
        
        choice_est = self.estimate_choice_collection_cost(
            num_problems, num_variants, num_repetitions, avg_alternatives, llm_model
        )
        
        total_cost = embedding_est["estimated_cost_usd"] + choice_est["estimated_cost_usd"]
        
        return {
            "embedding_costs": embedding_est,
            "choice_collection_costs": choice_est,
            "total_estimated_cost_usd": total_cost,
            "summary": f"Estimated total: ${total_cost:.2f} USD"
        }
    
    def print_estimate(self, estimate: Dict):
        """Pretty-print a cost estimate."""
        print("\n" + "="*50)
        print("COST ESTIMATE")
        print("="*50)
        
        if "embedding_costs" in estimate:
            emb = estimate["embedding_costs"]
            print(f"\nEmbeddings ({emb['model']}):")
            print(f"  - {emb['num_embeddings']:,} embeddings")
            print(f"  - ~{emb['estimated_tokens']:,} tokens")
            print(f"  - ${emb['estimated_cost_usd']:.4f}")
        
        if "choice_collection_costs" in estimate:
            ch = estimate["choice_collection_costs"]
            print(f"\nChoice Collection ({ch['model']}):")
            print(f"  - {ch['num_api_calls']:,} API calls")
            print(f"  - ~{ch['estimated_input_tokens']:,} input tokens")
            print(f"  - ~{ch['estimated_output_tokens']:,} output tokens")
            print(f"  - ${ch['estimated_cost_usd']:.4f}")
        
        if "total_estimated_cost_usd" in estimate:
            print(f"\n{'='*50}")
            print(f"TOTAL: ${estimate['total_estimated_cost_usd']:.2f} USD")
            print("="*50 + "\n")
```

---

### NEW: `validation.py`

**Purpose**: Validate configuration and data formats.

```python
"""
Validation utilities for configuration and data formats.
"""
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate study configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of warning messages (empty if all good)
        
    Raises:
        ValidationError: If critical validation fails
    """
    warnings = []
    
    # Required fields
    required = ["num_problems", "K"]
    for field in required:
        if field not in config:
            raise ValidationError(f"Missing required config field: {field}")
    
    # Type checks
    if not isinstance(config.get("num_problems"), int) or config["num_problems"] < 1:
        raise ValidationError("num_problems must be a positive integer")
    
    if not isinstance(config.get("K"), int) or config["K"] < 2:
        raise ValidationError("K must be an integer >= 2")
    
    # Warnings for potentially problematic values
    if config.get("num_problems", 0) < 50:
        warnings.append(f"num_problems={config['num_problems']} may be too few for reliable estimation")
    
    if config.get("temperature", 0) == 0:
        warnings.append("temperature=0 means deterministic responses; consider T>0 for variability")
    
    if config.get("num_repetitions", 1) > 1 and config.get("temperature", 0) == 0:
        warnings.append("Multiple repetitions with temperature=0 will yield identical results")
    
    return warnings


def validate_stan_data(data: Dict[str, Any], model: str = "m_0") -> None:
    """
    Validate Stan data package matches expected schema.
    
    Args:
        data: Stan data dictionary
        model: Model name ("m_0" or "m_1")
        
    Raises:
        ValidationError: If validation fails
    """
    if model == "m_0":
        _validate_m0_data(data)
    elif model == "m_1":
        _validate_m1_data(data)
    else:
        raise ValidationError(f"Unknown model: {model}")


def _validate_m0_data(data: Dict[str, Any]) -> None:
    """Validate data for m_0 model."""
    required_fields = {"M", "K", "D", "R", "w", "I", "y"}
    
    # Check all required fields present
    missing = required_fields - set(data.keys())
    if missing:
        raise ValidationError(f"Missing required fields for m_0: {missing}")
    
    M, K, D, R = data["M"], data["K"], data["D"], data["R"]
    
    # Validate dimensions
    if not isinstance(M, int) or M < 1:
        raise ValidationError(f"M must be positive integer, got {M}")
    if not isinstance(K, int) or K < 2:
        raise ValidationError(f"K must be integer >= 2, got {K}")
    if not isinstance(D, int) or D < 1:
        raise ValidationError(f"D must be positive integer, got {D}")
    if not isinstance(R, int) or R < 1:
        raise ValidationError(f"R must be positive integer, got {R}")
    
    # Validate w array shape (R x D)
    w = data["w"]
    if len(w) != R:
        raise ValidationError(f"w should have {R} rows, got {len(w)}")
    if any(len(row) != D for row in w):
        raise ValidationError(f"w rows should have {D} columns")
    
    # Validate I array shape (M x R)
    I = data["I"]
    if len(I) != M:
        raise ValidationError(f"I should have {M} rows, got {len(I)}")
    if any(len(row) != R for row in I):
        raise ValidationError(f"I rows should have {R} columns")
    
    # Validate I contains only 0s and 1s
    for m, row in enumerate(I):
        if not all(v in (0, 1) for v in row):
            raise ValidationError(f"I[{m}] contains values other than 0 or 1")
        if sum(row) < 2:
            raise ValidationError(f"Problem {m} has fewer than 2 alternatives (sum(I[{m}])={sum(row)})")
    
    # Validate y
    y = data["y"]
    if len(y) != M:
        raise ValidationError(f"y should have {M} elements, got {len(y)}")
    
    # y values should be valid alternative indices
    for m, choice in enumerate(y):
        if not isinstance(choice, int) or choice < 1:
            raise ValidationError(f"y[{m}]={choice} is not a positive integer")
        # Check choice is among available alternatives for this problem
        num_alts = sum(I[m])
        if choice > num_alts:
            raise ValidationError(f"y[{m}]={choice} exceeds number of alternatives ({num_alts})")
    
    logger.info(f"Stan data validated: M={M}, K={K}, D={D}, R={R}")


def _validate_m1_data(data: Dict[str, Any]) -> None:
    """Validate data for m_1 model (extends m_0 with risky choice)."""
    # First validate m_0 fields
    _validate_m0_data(data)
    
    # Additional m_1 fields
    m1_fields = {"N", "p", "y_risky"}
    missing = m1_fields - set(data.keys())
    if missing:
        raise ValidationError(f"Missing required fields for m_1: {missing}")
    
    N, K = data["N"], data["K"]
    
    # Validate N
    if not isinstance(N, int) or N < 1:
        raise ValidationError(f"N must be positive integer, got {N}")
    
    # Validate p (N x ? x K) - probabilities for risky problems
    # Shape depends on implementation details
    
    logger.info(f"Stan data validated for m_1: N={N} risky problems")


def validate_claims_file(filepath: str) -> Dict[str, Any]:
    """
    Validate and load claims file.
    
    Returns:
        Validated claims data
        
    Raises:
        ValidationError: If validation fails
    """
    path = Path(filepath)
    if not path.exists():
        raise ValidationError(f"Claims file not found: {filepath}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Required fields
    if "claims" not in data:
        raise ValidationError("Claims file must contain 'claims' array")
    
    claims = data["claims"]
    if not isinstance(claims, list) or len(claims) < 2:
        raise ValidationError("Must have at least 2 claims")
    
    # Validate each claim
    claim_ids = set()
    for i, claim in enumerate(claims):
        if "id" not in claim:
            raise ValidationError(f"Claim {i} missing 'id' field")
        if "description" not in claim:
            raise ValidationError(f"Claim {i} missing 'description' field")
        
        if claim["id"] in claim_ids:
            raise ValidationError(f"Duplicate claim ID: {claim['id']}")
        claim_ids.add(claim["id"])
    
    logger.info(f"Validated {len(claims)} claims from {filepath}")
    return data
```

---

### NEW: `cli.py`

**Purpose**: Command-line interface for running studies.

```python
"""
Command-line interface for the prompt framing study.
"""
import argparse
import logging
import sys
from pathlib import Path

def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_run(args):
    """Run the full study pipeline."""
    from .study_runner import StudyRunner
    
    runner = StudyRunner(config_path=args.config)
    
    if args.dry_run:
        from .cost_estimator import CostEstimator
        estimator = CostEstimator()
        estimate = estimator.estimate_total_study_cost(
            num_claims=20,  # Would need to load from config
            num_problems=runner.config.get("num_problems", 100),
            num_variants=4,
            num_repetitions=runner.config.get("num_repetitions", 1),
            llm_model=runner.config.get("llm_model", "gpt-4"),
            embedding_model=runner.config.get("embedding_model", "text-embedding-3-small")
        )
        estimator.print_estimate(estimate)
        print("\nDry run complete. Use --no-dry-run to execute.")
        return
    
    runner.run()


def cmd_estimate(args):
    """Estimate costs for a study configuration."""
    from .cost_estimator import CostEstimator
    import yaml
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    estimator = CostEstimator()
    estimate = estimator.estimate_total_study_cost(
        num_claims=args.num_claims,
        num_problems=config.get("num_problems", 100),
        num_variants=4,
        num_repetitions=config.get("num_repetitions", 1),
        llm_model=config.get("llm_model", "gpt-4"),
        embedding_model=config.get("embedding_model", "text-embedding-3-small")
    )
    estimator.print_estimate(estimate)


def cmd_validate(args):
    """Validate configuration and data files."""
    from .validation import validate_config, validate_claims_file, validate_stan_data
    import yaml
    import json
    
    errors = []
    
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            warnings = validate_config(config)
            print(f"✓ Config valid: {args.config}")
            for w in warnings:
                print(f"  ⚠ {w}")
        except Exception as e:
            errors.append(f"Config: {e}")
    
    if args.claims:
        try:
            validate_claims_file(args.claims)
            print(f"✓ Claims valid: {args.claims}")
        except Exception as e:
            errors.append(f"Claims: {e}")
    
    if args.stan_data:
        try:
            with open(args.stan_data, 'r') as f:
                data = json.load(f)
            validate_stan_data(data, model=args.model)
            print(f"✓ Stan data valid: {args.stan_data}")
        except Exception as e:
            errors.append(f"Stan data: {e}")
    
    if errors:
        print("\nValidation errors:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)


def cmd_visualize(args):
    """Generate visualizations from results."""
    from .visualization import StudyVisualizer
    
    viz = StudyVisualizer(args.results_dir)
    
    if args.alpha_comparison:
        import json
        with open(Path(args.results_dir) / "run_metadata.json", 'r') as f:
            metadata = json.load(f)
        if "model_fits" in metadata:
            viz.plot_alpha_comparison(
                metadata["model_fits"],
                save_path=args.output or "alpha_comparison.png"
            )
            print(f"Saved alpha comparison plot")


def main():
    parser = argparse.ArgumentParser(
        prog="prompt_framing_study",
        description="Investigate how prompt framing affects LLM rationality"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the study pipeline")
    run_parser.add_argument("-c", "--config", default="configs/study_config.yaml",
                           help="Path to study config")
    run_parser.add_argument("--dry-run", action="store_true",
                           help="Show cost estimate without running")
    run_parser.set_defaults(func=cmd_run)
    
    # Estimate command
    est_parser = subparsers.add_parser("estimate", help="Estimate study costs")
    est_parser.add_argument("-c", "--config", required=True, help="Path to study config")
    est_parser.add_argument("--num-claims", type=int, default=20,
                           help="Number of base claims")
    est_parser.set_defaults(func=cmd_estimate)
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate files")
    val_parser.add_argument("--config", help="Validate config file")
    val_parser.add_argument("--claims", help="Validate claims file")
    val_parser.add_argument("--stan-data", help="Validate Stan data file")
    val_parser.add_argument("--model", default="m_0", choices=["m_0", "m_1"],
                           help="Model for Stan data validation")
    val_parser.set_defaults(func=cmd_validate)
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("results_dir", help="Path to results directory")
    viz_parser.add_argument("--alpha-comparison", action="store_true",
                           help="Generate alpha comparison plot")
    viz_parser.add_argument("-o", "--output", help="Output file path")
    viz_parser.set_defaults(func=cmd_visualize)
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

---

### Updated `study_runner.py`

**Add checkpointing and improved logging.**

```python
# Add to the existing study_runner.py specification

class StudyRunner:
    """
    Orchestrate the complete study pipeline with checkpointing support.
    
    Pipeline:
    1. Load configuration
    2. Generate problems (or load existing)
    3. Generate embeddings for all variants
    4. Collect choices for all variants
    5. Prepare Stan data packages
    6. Fit m_0 model for each variant
    7. Save results and generate visualizations
    """
    
    def __init__(self, config_path: str = "configs/study_config.yaml"):
        """Load study configuration."""
        import yaml
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate configuration
        from .validation import validate_config
        warnings = validate_config(self.config)
        for w in warnings:
            logger.warning(w)
        
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "results" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file for resumption
        self.checkpoint_file = self.results_dir / ".checkpoint.json"
        self.checkpoint = self._load_checkpoint()
        
        # Setup logging to file
        self._setup_file_logging()
        
        # Initialize components
        self.prompt_manager = PromptManager(self.base_dir / "configs/prompt_variants.yaml")
        self.embedding_manager = ContextualizedEmbeddingManager(
            embedding_model=self.config.get("embedding_model", "text-embedding-3-small")
        )
        self.choice_collector = ChoiceCollector(
            llm_model=self.config.get("llm_model", "gpt-4"),
            temperature=self.config.get("temperature", 0.7)
        )
    
    def _setup_file_logging(self):
        """Configure logging to file in results directory."""
        log_file = self.results_dir / "study.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logger.info(f"Logging to {log_file}")
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Resuming from checkpoint: completed steps = {checkpoint.get('completed_steps', [])}")
            return checkpoint
        return {"completed_steps": []}
    
    def _save_checkpoint(self, step: str, data: Optional[Dict] = None):
        """Save checkpoint after completing a step."""
        self.checkpoint["completed_steps"].append(step)
        if data:
            self.checkpoint[step] = data
        self.checkpoint["last_updated"] = datetime.now().isoformat()
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2, default=str)
        
        logger.debug(f"Checkpoint saved: {step}")
    
    def _step_completed(self, step: str) -> bool:
        """Check if a step was already completed."""
        return step in self.checkpoint.get("completed_steps", [])
    
    def run(self) -> Dict[str, Any]:
        """Execute the full study pipeline with checkpointing."""
        results = {"config": self.config, "timestamp": datetime.now().isoformat()}
        
        try:
            # Step 1: Load or generate problems
            if not self._step_completed("problems"):
                logger.info("Step 1: Preparing decision problems...")
                problems = self._prepare_problems()
                self._save_checkpoint("problems", {"num_problems": len(problems)})
            else:
                logger.info("Step 1: Loading problems from checkpoint...")
                problems, _ = ProblemGenerator.load_problems(self.results_dir / "problems.json")
            results["num_problems"] = len(problems)
            
            # Step 2: Load claims for embedding
            logger.info("Step 2: Loading claims...")
            claims = self._load_claims()
            results["num_claims"] = len(claims)
            
            # Step 3: Get prompt variants
            logger.info("Step 3: Loading prompt variants...")
            variants = self.prompt_manager.get_all_variants()
            results["variants"] = [v.name for v in variants]
            
            # Step 4: Generate embeddings for all variants
            logger.info("Step 4: Generating contextualized embeddings...")
            all_embeddings = self.embedding_manager.embed_all_variants(claims, variants)
            self.embedding_manager.save_embeddings(
                all_embeddings, 
                self.results_dir / "raw_embeddings.npz"
            )
            self._save_checkpoint("embeddings")
            
            # Step 5: Apply dimension reduction
            logger.info("Step 5: Reducing embedding dimensions...")
            target_dim = self.config.get("target_dim", 32)
            reduced_embeddings = {}
            for variant_name, embs in all_embeddings.items():
                reduced_embeddings[variant_name], _ = self.embedding_manager.reduce_dimensions(
                    embs, target_dim=target_dim
                )
            
            # Step 6: Collect choices for all variants
            logger.info("Step 6: Collecting LLM choices...")
            num_reps = self.config.get("num_repetitions", 1)
            all_choices = self.choice_collector.collect_all_variants(problems, variants, num_reps)
            
            # Save raw choices
            with open(self.results_dir / "raw_choices.json", 'w') as f:
                json.dump(all_choices, f, indent=2)
            self._save_checkpoint("choices")
            
            # Step 7: Prepare Stan data packages
            logger.info("Step 7: Preparing Stan data packages...")
            stan_data_packages = {}
            for variant in variants:
                stan_data = self._create_stan_data(
                    problems=problems,
                    embeddings=reduced_embeddings[variant.name],
                    choices=all_choices[variant.name]["choices"][0],  # First repetition
                    K=self.config.get("K", 3)
                )
                stan_data_packages[variant.name] = stan_data
                
                # Save each package
                with open(self.results_dir / f"stan_data_{variant.name}.json", 'w') as f:
                    json.dump(stan_data, f, indent=2)
            
            self._save_checkpoint("stan_data")
            
            # Step 8: Fit models (optional, can be done separately)
            if self.config.get("fit_models", False):
                logger.info("Step 8: Fitting m_0 models...")
                results["model_fits"] = self._fit_models(stan_data_packages)
            
            # Step 9: Save metadata
            logger.info("Step 9: Saving results...")
            results["results_dir"] = str(self.results_dir)
            with open(self.results_dir / "run_metadata.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Study complete! Results saved to {self.results_dir}")
            return results
        
        except Exception as e:
            logger.error(f"Study failed at step: {e}", exc_info=True)
            raise
        
        # Clear checkpoint on successful completion
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
```

---

## Implementation Order (Updated)

### Phase 1: Core Infrastructure (Priority: High)
1. Create directory structure at `applications/prompt_framing_study/`
2. Implement `__init__.py` with logging setup
3. Implement `llm_client.py` (standalone, with cost tracking)
4. Implement `prompt_manager.py` with YAML config
5. Implement `validation.py`
6. Create `data/claims.json`

### Phase 2: Data Collection Pipeline (Priority: High)
7. Implement `contextualized_embedding.py`
8. Implement `problem_generator.py`
9. Implement `choice_collector.py`
10. Write unit tests for phases 1-2

### Phase 3: Study Orchestration (Priority: High)
11. Implement `cost_estimator.py`
12. Implement `study_runner.py` with checkpointing
13. Create config files
14. Implement `cli.py`
15. Test end-to-end pipeline with small pilot

### Phase 4: Analysis & Visualization (Priority: Medium)
16. Implement `robustness_analysis.py`
17. Implement `visualization.py`
18. Write `README.md` documentation

### Phase 5: Testing & Refinement (Priority: Medium)
19. Expand unit tests
20. Run full pilot study
21. Refine based on results

---

## Dependencies

### New Dependencies (add to `requirements.txt`)
```
pyyaml>=6.0
```

### Existing Dependencies (already in project)
```
openai
numpy
pandas
matplotlib
seaborn
scikit-learn
cmdstanpy
```

---

## Integration Points

### With Analysis Module (`analysis/`)
- Use `ModelEstimation` class from `model_estimation.py` for fitting m_0

### With Models (`models/`)
- Use `m_0.stan` for model fitting

### Legacy Module (`llm_rationality/`)
- **No dependency**: This module is standalone
- The legacy module may be deprecated once this module is stable

---

## Testing Strategy

### Unit Tests (`tests/test_pipeline.py`)
```python
def test_prompt_manager_loads_variants():
    """Test that prompt variants load correctly."""
    ...

def test_prompt_formatting():
    """Test that prompts are formatted correctly with claims."""
    ...

def test_embedding_generation():
    """Test contextualized embedding generation (mock API)."""
    ...

def test_dimension_reduction():
    """Test PCA dimension reduction."""
    ...

def test_stan_data_creation():
    """Test Stan data package creation matches m_0 schema."""
    ...

def test_problem_generation():
    """Test decision problem generation."""
    ...
```

### Integration Tests
```python
def test_end_to_end_pipeline():
    """Test full pipeline with minimal data (2 claims, 5 problems, 1 variant)."""
    ...
```

---

## Notes for Implementing Agent

1. **API Costs**: The study involves API calls for embeddings and choice collection. Implement caching where possible and provide cost estimates in logs.

2. **Error Handling**: Add robust error handling for API failures with retries.

3. **Reproducibility**: Use seeds consistently and save all random states.

4. **Logging**: Add informative logging throughout the pipeline for debugging.

5. **Configuration Validation**: Validate config files on load and provide helpful error messages.

6. **Standalone Design**: This module should not import from `llm_rationality`. All needed functionality is included directly.

7. **Stan Data Format**: Ensure Stan data matches the exact schema expected by `m_0.stan` (check data types, array dimensions).

8. **Temperature Setting**: The choice collector uses temperature > 0 by default. Document this clearly as it affects interpretation.