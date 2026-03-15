#!/usr/bin/env python
"""
Convenience entry point for the Ellsberg Study.

Usage:
    python scripts/run_ellsberg_study.py validate
    python scripts/run_ellsberg_study.py estimate-cost
    python scripts/run_ellsberg_study.py run [--skip-fitting]
    python scripts/run_ellsberg_study.py fit [--model m_02]
"""
import sys
from pathlib import Path

# Ensure the project root is on the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from applications.ellsberg_study.cli import main

if __name__ == "__main__":
    main()
