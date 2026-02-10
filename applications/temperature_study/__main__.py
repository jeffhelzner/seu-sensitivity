"""
Entry point for running the temperature_study module.

Usage:
    python -m temperature_study [command] [options]

Or from the project root:
    python -m applications.temperature_study [command] [options]
"""
from .cli import main

if __name__ == "__main__":
    main()
