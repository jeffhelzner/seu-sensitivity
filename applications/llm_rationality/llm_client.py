"""
LLM Client Module for Rationality Benchmarking

This module provides interfaces for interacting with various LLM providers.
"""
from typing import List, Dict, Any, Optional
import os
import re

class LLMClient:
    """Base class for LLM clients."""
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize LLM client.
        
        Args:
            model: Model name/identifier
            **kwargs: Additional model-specific parameters
        """
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def make_choice(self, context: str, alternatives: List[str], **kwargs) -> int:
        """
        Present a decision problem to the LLM and get its choice.
        
        Args:
            context: Problem context/description
            alternatives: List of alternative options
            **kwargs: Additional generation parameters
            
        Returns:
            Index of the chosen alternative (0-indexed)
        """
        # Construct a prompt that asks the LLM to make a choice
        prompt = f"{context}\n\nPlease choose one of the following options:\n"
        
        for i, alt in enumerate(alternatives):
            prompt += f"{i+1}. {alt}\n"
        
        prompt += f"\nIMPORTANT: Respond with ONLY the option number (1-{len(alternatives)})."
        
        # Get response
        response = self.generate(prompt, **kwargs)
        
        # Parse the choice - improved version
        try:
            # First try to find a pattern like "Option: 2" or "I choose 2"
            choice_patterns = [
                r'(\d+)\s*$',                     # Just a number at the end
                r'^[^\d]*?(\d+)[^\d]*$',          # Only one number in the response
                r'(?:option|choice|select|choose|pick|answer)[^\d]*?(\d+)',  # Option/Choice X
                r'(\d+)'                           # Any number as fallback
            ]
            
            for pattern in choice_patterns:
                matches = re.search(pattern, response.lower())
                if matches:
                    choice = int(matches.group(1)) - 1
                    # Validate choice is within range
                    if 0 <= choice < len(alternatives):
                        return choice
                    else:
                        print(f"Warning: Extracted choice {choice+1} out of range. Trying next pattern.")
            
            # If we get here, no valid choice was found
            print(f"Warning: Could not extract valid choice from response: {response}")
            # Just take the first number that's in range as a fallback
            numbers = re.findall(r'\d+', response)
            for num in numbers:
                choice = int(num) - 1
                if 0 <= choice < len(alternatives):
                    print(f"Using fallback extraction: choice {choice+1}")
                    return choice
            
            print(f"Warning: No valid choice found. Defaulting to 0.")
            return 0
        except Exception as e:
            print(f"Warning: Error parsing choice: {e}. Response: {response}")
            return 0  # Default to first option if parsing fails


class OpenAIClient(LLMClient):
    def __init__(self, 
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None, 
                 temperature: float = 0.0,
                 **kwargs):
        """
        Initialize OpenAI client.
        """
        super().__init__(model, **kwargs)
        self.temperature = temperature
        
        try:
            import openai
            if api_key:
                openai.api_key = api_key
            else:
                # Use environment variable
                openai.api_key = os.getenv("OPENAI_API_KEY")
                if not openai.api_key:
                    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
            self.client = openai.OpenAI()
            self.api_version = "v2"  # Assume newer version initially
            
        except ImportError:
            raise ImportError("OpenAI package not installed. Run 'pip install openai'")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI's API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            # Try the newer API structure (v1.0.0+)
            try:
                # First attempt with chat.completions API (current v1.x approach)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    **kwargs
                )
                return response.choices[0].message.content.strip()
            except (AttributeError, TypeError):
                # Fall back to older API (pre-v1.0.0)
                import openai
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    **kwargs
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            # Add retry logic for common API errors
            import time
            max_retries = 3
            for attempt in range(max_retries):
                if attempt > 0:  # Skip first attempt as it's already done
                    print(f"Error: {e}. Retrying in {2.0} seconds...")
                    time.sleep(2.0)
                    try:
                        # Try again with chat.completions API
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=self.temperature,
                                **kwargs
                            )
                            return response.choices[0].message.content.strip()
                        except (AttributeError, TypeError):
                            # Fall back to older API
                            import openai
                            response = openai.ChatCompletion.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=self.temperature,
                                **kwargs
                            )
                            return response.choices[0].message.content.strip()
                    except Exception as retry_e:
                        e = retry_e  # Update error for next iteration
                        continue
            
            # If we get here, all retries failed
            raise Exception(f"Failed to generate text after {max_retries} attempts: {e}")