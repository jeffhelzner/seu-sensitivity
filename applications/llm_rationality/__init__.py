"""
LLM Rationality Benchmarking Application

This package provides tools for assessing the "rationality" of LLMs
by estimating their sensitivity (alpha) to expected utility maximization.
"""

# Import available modules
from .llm_client import LLMClient, OpenAIClient

__all__ = [
    'LLMClient',
    'OpenAIClient',
]