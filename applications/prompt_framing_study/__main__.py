"""
Entry point for running the prompt_framing_study module.

Usage:
    python -m prompt_framing_study [command] [options]
    
Or from the project root:
    python -m applications.prompt_framing_study [command] [options]
"""
from .cli import main

if __name__ == "__main__":
    main()
