"""
Cost estimation utilities for the prompt framing study.
"""
from typing import Dict, List, Optional, Any
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
    
    # Anthropic Claude pricing per 1M tokens
    ANTHROPIC_PRICING = {
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
    
    # Rough token estimates
    AVG_CLAIM_TOKENS = 150  # Average tokens per claim description
    AVG_PROMPT_OVERHEAD = 200  # Tokens for prompt template
    AVG_RESPONSE_TOKENS = 10  # Short response expected
    
    def __init__(self):
        self.estimates = {}
    
    def estimate_embedding_cost(
        self,
        num_claims: int,
        num_variants: int,
        embedding_model: str = "text-embedding-3-small"
    ) -> Dict[str, Any]:
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
    
    def estimate_choice_collection_cost(
        self,
        num_problems: int,
        num_variants: int,
        num_repetitions: int,
        avg_alternatives: float,
        llm_model: str = "gpt-4",
        provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Estimate cost for collecting LLM choices.
        
        Args:
            num_problems: Number of decision problems
            num_variants: Number of prompt variants
            num_repetitions: Times each problem presented
            avg_alternatives: Average alternatives per problem
            llm_model: Chat model
            provider: LLM provider ("openai" or "anthropic")
            
        Returns:
            Dict with token and cost estimates
        """
        total_calls = num_problems * num_variants * num_repetitions
        
        # Input tokens: prompt overhead + (alternatives * claim tokens)
        input_per_call = self.AVG_PROMPT_OVERHEAD + (avg_alternatives * self.AVG_CLAIM_TOKENS)
        total_input = total_calls * input_per_call
        total_output = total_calls * self.AVG_RESPONSE_TOKENS
        
        if provider == "anthropic":
            pricing = self.ANTHROPIC_PRICING.get(llm_model, {"input": 3.0, "output": 15.0})
        else:
            pricing = self.CHAT_PRICING.get(llm_model, {"input": 30.0, "output": 60.0})
        
        input_cost = (total_input / 1_000_000) * pricing["input"]
        output_cost = (total_output / 1_000_000) * pricing["output"]
        
        return {
            "num_api_calls": total_calls,
            "estimated_input_tokens": int(total_input),
            "estimated_output_tokens": int(total_output),
            "estimated_cost_usd": input_cost + output_cost,
            "model": llm_model,
            "provider": provider
        }
    
    def estimate_total_study_cost(
        self,
        num_claims: int,
        num_problems: int,
        num_variants: int = 4,
        num_repetitions: int = 1,
        avg_alternatives: float = 3.0,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4",
        provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Estimate total cost for a complete study run.
        
        Returns comprehensive cost breakdown.
        """
        embedding_est = self.estimate_embedding_cost(
            num_claims, num_variants, embedding_model
        )
        
        choice_est = self.estimate_choice_collection_cost(
            num_problems, num_variants, num_repetitions, 
            avg_alternatives, llm_model, provider
        )
        
        total_cost = embedding_est["estimated_cost_usd"] + choice_est["estimated_cost_usd"]
        
        return {
            "embedding_costs": embedding_est,
            "choice_collection_costs": choice_est,
            "total_estimated_cost_usd": total_cost,
            "summary": f"Estimated total: ${total_cost:.2f} USD"
        }
    
    def print_estimate(self, estimate: Dict[str, Any]):
        """Pretty-print a cost estimate."""
        print("\n" + "=" * 50)
        print("COST ESTIMATE")
        print("=" * 50)
        
        if "embedding_costs" in estimate:
            emb = estimate["embedding_costs"]
            print(f"\nEmbeddings ({emb['model']}):")
            print(f"  - {emb['num_embeddings']:,} embeddings")
            print(f"  - ~{emb['estimated_tokens']:,} tokens")
            print(f"  - ${emb['estimated_cost_usd']:.4f}")
        
        if "choice_collection_costs" in estimate:
            ch = estimate["choice_collection_costs"]
            model_str = f"{ch.get('provider', 'openai')}/{ch['model']}"
            print(f"\nChoice Collection ({model_str}):")
            print(f"  - {ch['num_api_calls']:,} API calls")
            print(f"  - ~{ch['estimated_input_tokens']:,} input tokens")
            print(f"  - ~{ch['estimated_output_tokens']:,} output tokens")
            print(f"  - ${ch['estimated_cost_usd']:.4f}")
        
        if "total_estimated_cost_usd" in estimate:
            print(f"\n{'=' * 50}")
            print(f"TOTAL: ${estimate['total_estimated_cost_usd']:.2f} USD")
            print("=" * 50 + "\n")
    
    def format_estimate_markdown(self, estimate: Dict[str, Any]) -> str:
        """Format cost estimate as markdown."""
        lines = ["## Cost Estimate\n"]
        
        if "embedding_costs" in estimate:
            emb = estimate["embedding_costs"]
            lines.append(f"### Embeddings ({emb['model']})")
            lines.append(f"- {emb['num_embeddings']:,} embeddings")
            lines.append(f"- ~{emb['estimated_tokens']:,} tokens")
            lines.append(f"- **${emb['estimated_cost_usd']:.4f}**\n")
        
        if "choice_collection_costs" in estimate:
            ch = estimate["choice_collection_costs"]
            model_str = f"{ch.get('provider', 'openai')}/{ch['model']}"
            lines.append(f"### Choice Collection ({model_str})")
            lines.append(f"- {ch['num_api_calls']:,} API calls")
            lines.append(f"- ~{ch['estimated_input_tokens']:,} input tokens")
            lines.append(f"- ~{ch['estimated_output_tokens']:,} output tokens")
            lines.append(f"- **${ch['estimated_cost_usd']:.4f}**\n")
        
        if "total_estimated_cost_usd" in estimate:
            lines.append(f"### Total: **${estimate['total_estimated_cost_usd']:.2f} USD**")
        
        return "\n".join(lines)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CostEstimator":
        """Create estimator and compute estimate from config."""
        estimator = cls()
        
        estimate = estimator.estimate_total_study_cost(
            num_claims=config.get("num_claims", 20),
            num_problems=config.get("num_problems", 100),
            num_variants=config.get("num_variants", 4),
            num_repetitions=config.get("num_repetitions", 1),
            avg_alternatives=config.get("avg_alternatives", 3.0),
            embedding_model=config.get("embedding_model", "text-embedding-3-small"),
            llm_model=config.get("llm_model", "gpt-4"),
            provider=config.get("provider", "openai")
        )
        
        estimator.estimates["from_config"] = estimate
        return estimator
