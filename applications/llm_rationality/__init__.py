"""
LLM Rationality Benchmarking Application

This package provides tools for assessing the "rationality" of LLMs
by estimating their sensitivity (alpha) to expected utility maximization.
"""

from .embedding import TextFeatureGenerator
from .llm_client import LLMClient, OpenAIClient, AnthropicClient
from .benchmark import benchmark_llm_rationality, load_problems, save_results
from .visualization import plot_rationality_comparison, plot_choice_analysis

__all__ = [
    'TextFeatureGenerator',
    'LLMClient', 'OpenAIClient', 'AnthropicClient',
    'benchmark_llm_rationality', 'load_problems', 'save_results',
    'plot_rationality_comparison', 'plot_choice_analysis'
]