"""
Prompt Manager Module for Prompt Framing Study

Manages prompt templates with varying rationality emphasis levels.
"""
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """Represents a single prompt variant with metadata."""
    
    name: str
    level: int  # 0=minimal, 1=baseline, 2=enhanced, 3=maximal
    template: str
    description: str
    embedding_template: str
    
    def format_choice_prompt(self, claims: List[str]) -> str:
        """
        Format the full prompt for choice collection with all claims listed.
        
        Args:
            claims: List of claim descriptions
            
        Returns:
            Formatted prompt string ready for LLM
        """
        claims_list = "\n".join([f"- Claim {i+1}: {claim}" for i, claim in enumerate(claims)])
        return self.template.format(claims_list=claims_list)
    
    def format_embedding_prompt(self, claim: str) -> str:
        """
        Format a prompt for embedding a single claim in context.
        
        Uses the variant's framing but with only one claim, allowing
        the embedding to capture how the claim is "perceived" under
        this particular framing.
        
        Args:
            claim: Single claim description
            
        Returns:
            Formatted prompt for embedding
        """
        return self.embedding_template.format(claim=claim)
    
    def __repr__(self) -> str:
        return f"PromptVariant(name='{self.name}', level={self.level})"


class PromptManager:
    """Load and manage prompt variants from configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the prompt manager.
        
        Args:
            config_path: Path to prompt_variants.yaml. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "prompt_variants.yaml"
        
        self.config_path = Path(config_path)
        self.variants: Dict[str, PromptVariant] = {}
        self._load_variants()
    
    def _load_variants(self):
        """Load prompt variants from YAML configuration."""
        import yaml
        
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            self._load_default_variants()
            return
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for variant_data in config.get("variants", []):
            variant = PromptVariant(
                name=variant_data["name"],
                level=variant_data["level"],
                template=variant_data["template"],
                description=variant_data["description"],
                embedding_template=variant_data.get("embedding_template", variant_data["template"])
            )
            self.variants[variant.name] = variant
            logger.debug(f"Loaded variant: {variant.name} (level {variant.level})")
        
        logger.info(f"Loaded {len(self.variants)} prompt variants from {self.config_path}")
    
    def _load_default_variants(self):
        """Load default prompt variants when config file is not available."""
        default_variants = [
            PromptVariant(
                name="minimal",
                level=0,
                description="Minimal framing - just pick one, no outcome structure",
                template="""You are reviewing flagged insurance claims. Choose one to send for investigation.

{claims_list}

Which claim do you choose? Respond with only the claim number (e.g., "Claim 1").""",
                embedding_template="""You are reviewing flagged insurance claims. Choose one to send for investigation.

Claim to evaluate: {claim}"""
            ),
            PromptVariant(
                name="baseline",
                level=1,
                description="Standard framing with K=3 outcome structure",
                template="""You are a claims adjuster working for a large insurance company. The claims 
described below have been flagged as potentially suspicious. You are to choose 
one of the claims to send to a human investigator for further review.

Your decision will be reviewed by two expert judges:
- Best outcome: Both judges agree your selection warrants investigation
- Middle outcome: One judge supports your decision, one doesn't  
- Worst outcome: Neither judge agrees with your decision

{claims_list}

Which claim do you choose to send to the human investigator for further review? 
Please respond with only the claim number (e.g., "Claim 1").""",
                embedding_template="""You are a claims adjuster working for a large insurance company. A claim has been 
flagged as potentially suspicious. Your decision to investigate will be reviewed 
by two expert judges who will evaluate whether the claim warrants investigation.

Claim to evaluate: {claim}"""
            ),
            PromptVariant(
                name="enhanced",
                level=2,
                description="Explicit utility values and rationality language",
                template="""You are a claims adjuster working for a large insurance company. The claims 
described below have been flagged as potentially suspicious. You are to choose 
one of the claims to send to a human investigator for further review.

Your decision will be reviewed by two expert judges. The possible outcomes are:
- Best outcome (utility = 1.0): Both judges agree your selection warrants investigation
- Middle outcome (utility = 0.5): One judge supports your decision, one doesn't
- Worst outcome (utility = 0.0): Neither judge agrees with your decision

Make your choice to maximize the expected quality of the outcome.

{claims_list}

Which claim do you choose? Respond with only the claim number (e.g., "Claim 1").""",
                embedding_template="""You are a claims adjuster making a rational decision about which claims warrant 
investigation. Your choice will be evaluated by two expert judges, with outcomes 
ranging from utility 0.0 (neither agrees) to utility 1.0 (both agree).

Claim to evaluate: {claim}"""
            ),
            PromptVariant(
                name="maximal",
                level=3,
                description="Full decision-theoretic framing with explicit EU maximization",
                template="""You are a rational decision-making agent optimizing claim investigation decisions.

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

Which claim do you choose? Respond with only the claim number (e.g., "Claim 1").""",
                embedding_template="""You are a rational decision-making agent evaluating insurance claims. Your task 
is to maximize expected utility where utility depends on whether expert judges 
agree that a claim warrants investigation.

Claim to evaluate: {claim}"""
            ),
        ]
        
        for variant in default_variants:
            self.variants[variant.name] = variant
        
        logger.info(f"Loaded {len(self.variants)} default prompt variants")
    
    def get_variant(self, name: str) -> PromptVariant:
        """
        Get a specific prompt variant by name.
        
        Args:
            name: Variant name (e.g., "minimal", "baseline", "enhanced", "maximal")
            
        Returns:
            PromptVariant object
            
        Raises:
            KeyError: If variant name not found
        """
        if name not in self.variants:
            available = list(self.variants.keys())
            raise KeyError(f"Unknown variant '{name}'. Available: {available}")
        return self.variants[name]
    
    def get_all_variants(self) -> List[PromptVariant]:
        """
        Get all prompt variants ordered by level (ascending).
        
        Returns:
            List of PromptVariant objects sorted by level
        """
        return sorted(self.variants.values(), key=lambda v: v.level)
    
    def list_variant_names(self) -> List[str]:
        """
        Return list of available variant names.
        
        Returns:
            List of variant name strings
        """
        return list(self.variants.keys())
    
    def get_variants_by_level(self, level: int) -> List[PromptVariant]:
        """
        Get all variants at a specific level.
        
        Args:
            level: Rationality emphasis level (0-3)
            
        Returns:
            List of variants at that level
        """
        return [v for v in self.variants.values() if v.level == level]
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of all variants for logging/display.
        
        Returns:
            Dict with variant summary information
        """
        return {
            "num_variants": len(self.variants),
            "variants": [
                {
                    "name": v.name,
                    "level": v.level,
                    "description": v.description
                }
                for v in self.get_all_variants()
            ]
        }
