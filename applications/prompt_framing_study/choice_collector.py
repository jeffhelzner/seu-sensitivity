"""
Choice Collector Module for Prompt Framing Study

Collects LLM choices for decision problems under different prompt variants.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import logging
from pathlib import Path

from .llm_client import OpenAIClient, create_llm_client
from .prompt_manager import PromptVariant

logger = logging.getLogger(__name__)


class ChoiceCollector:
    """
    Collect choices from LLMs for decision problems under different prompt variants.
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4",
        temperature: float = 0.7,
        provider: str = "openai",
        api_key: Optional[str] = None
    ):
        """
        Initialize the choice collector.
        
        Args:
            llm_model: Model name for choice generation
            temperature: Sampling temperature (>0 recommended for choice variability)
            provider: LLM provider ("openai" or "anthropic")
            api_key: API key (uses env var if not provided)
        """
        self.client = create_llm_client(
            provider=provider,
            model=llm_model,
            temperature=temperature,
            api_key=api_key
        )
        self.llm_model = llm_model
        self.temperature = temperature
        self.provider = provider
        
        logger.info(f"Initialized ChoiceCollector with {provider}/{llm_model}, T={temperature}")
    
    def collect_choices(
        self,
        problems: List[Dict],
        variant: PromptVariant,
        num_repetitions: int = 1,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Collect choices for all problems under a specific prompt variant.
        
        Args:
            problems: List of decision problems
            variant: Prompt variant to use
            num_repetitions: Number of times to present each problem (for T>0)
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            Dict with choices and metadata
        """
        all_choices = []
        all_responses = []
        
        total_calls = len(problems) * num_repetitions
        current = 0
        
        for rep in range(num_repetitions):
            rep_choices = []
            rep_responses = []
            
            for i, problem in enumerate(problems):
                # Get claim descriptions for this problem
                claims = problem.get("claims", [])
                if not claims and "claim_ids" in problem:
                    logger.warning(f"Problem {problem['id']} missing claim descriptions")
                    continue
                
                # Format the prompt with this variant
                prompt = variant.format_choice_prompt(claims)
                
                # Get LLM choice
                try:
                    choice = self.client.make_choice_with_prompt(
                        prompt=prompt,
                        num_alternatives=len(claims)
                    )
                    
                    rep_choices.append({
                        "problem_id": problem["id"],
                        "choice": choice,  # 0-indexed
                        "choice_1indexed": choice + 1,
                        "num_alternatives": len(claims)
                    })
                    
                    rep_responses.append({
                        "problem_id": problem["id"],
                        "prompt_length": len(prompt),
                        "success": True
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to get choice for problem {problem['id']}: {e}")
                    rep_choices.append({
                        "problem_id": problem["id"],
                        "choice": 0,  # Default to first
                        "choice_1indexed": 1,
                        "num_alternatives": len(claims),
                        "error": str(e)
                    })
                    rep_responses.append({
                        "problem_id": problem["id"],
                        "success": False,
                        "error": str(e)
                    })
                
                current += 1
                if progress_callback:
                    progress_callback(current, total_calls)
            
            all_choices.append(rep_choices)
            all_responses.append(rep_responses)
        
        # Flatten if single repetition
        if num_repetitions == 1:
            all_choices = all_choices[0]
            all_responses = all_responses[0]
        
        # Get usage stats if available
        usage = {}
        if hasattr(self.client, 'get_usage_summary'):
            usage = self.client.get_usage_summary()
        
        return {
            "variant": variant.name,
            "variant_level": variant.level,
            "llm_model": self.llm_model,
            "provider": self.provider,
            "temperature": self.temperature,
            "num_problems": len(problems),
            "num_repetitions": num_repetitions,
            "choices": all_choices,
            "responses": all_responses,
            "usage": usage,
            "timestamp": datetime.now().isoformat()
        }
    
    def collect_all_variants(
        self,
        problems: List[Dict],
        variants: List[PromptVariant],
        num_repetitions: int = 1,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Dict]:
        """
        Collect choices for all problems under all variants.
        
        Args:
            problems: List of decision problems
            variants: List of prompt variants
            num_repetitions: Number of repetitions per variant
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict mapping variant_name -> choice results
        """
        results = {}
        
        for variant in variants:
            logger.info(f"Collecting choices for variant: {variant.name}")
            results[variant.name] = self.collect_choices(
                problems, 
                variant, 
                num_repetitions,
                progress_callback
            )
        
        return results
    
    def save_choices(self, choices: Dict[str, Dict], filepath: str):
        """
        Save collected choices to JSON file.
        
        Args:
            choices: Dict of choice results by variant
            filepath: Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(choices, f, indent=2, default=str)
        
        logger.info(f"Saved choices to {filepath}")
    
    @staticmethod
    def load_choices(filepath: str) -> Dict[str, Dict]:
        """
        Load choices from JSON file.
        
        Args:
            filepath: Path to choices JSON file
            
        Returns:
            Dict of choice results by variant
        """
        with open(filepath, 'r') as f:
            choices = json.load(f)
        
        logger.info(f"Loaded choices from {filepath}")
        return choices
    
    def reset_usage(self):
        """Reset API usage tracking."""
        if hasattr(self.client, 'reset_usage'):
            self.client.reset_usage()


def extract_choice_vector(
    choices: Dict[str, Any],
    problems: List[Dict]
) -> List[int]:
    """
    Extract choice vector (y) from collected choices.
    
    Args:
        choices: Choice results dict (for one variant)
        problems: List of problems used
        
    Returns:
        List of 1-indexed choices aligned with problems
    """
    # Build lookup by problem ID
    if isinstance(choices["choices"], list) and len(choices["choices"]) > 0:
        if isinstance(choices["choices"][0], dict):
            # Single repetition format
            choice_lookup = {c["problem_id"]: c["choice_1indexed"] for c in choices["choices"]}
        else:
            # Multiple repetitions - use first repetition
            choice_lookup = {c["problem_id"]: c["choice_1indexed"] for c in choices["choices"][0]}
    else:
        choice_lookup = {}
    
    # Extract in problem order
    y = []
    for problem in problems:
        choice = choice_lookup.get(problem["id"], 1)  # Default to 1
        y.append(choice)
    
    return y
