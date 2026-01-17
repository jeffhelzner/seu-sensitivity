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
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        **kwargs
    ):
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
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", 100)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_error = e
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
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
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of API usage."""
        return {
            "model": self.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self.get_estimated_cost()
        }


class AnthropicClient(LLMClient):
    """Anthropic Claude API client with retry logic."""
    
    # Pricing per 1M tokens
    PRICING = {
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
    
    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic's messages API with retry logic."""
        last_error = None
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", 100)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    self.total_input_tokens += response.usage.input_tokens
                    self.total_output_tokens += response.usage.output_tokens
                
                return response.content[0].text.strip()
                
            except Exception as e:
                last_error = e
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
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


def create_llm_client(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: "openai" or "anthropic"
        model: Model name (uses provider default if not specified)
        **kwargs: Additional arguments passed to the client
        
    Returns:
        Configured LLM client
    """
    if provider == "openai":
        model = model or "gpt-4"
        return OpenAIClient(model=model, **kwargs)
    elif provider == "anthropic":
        model = model or "claude-3-sonnet-20240229"
        return AnthropicClient(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")
