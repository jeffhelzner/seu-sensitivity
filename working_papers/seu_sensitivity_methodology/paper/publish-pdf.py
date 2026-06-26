#!/usr/bin/env python3
"""Quarto post-render hook.

Copy the freshly rendered PDF to a clean, stable, committed path one level up so
it can be linked publicly (homepage, etc.) without exposing the _output/ build
directory. Regenerated on every render, so the published copy never goes stale.
"""
import shutil
from pathlib import Path

src = Path("_output/paper.pdf")
dst = Path("../seu-sensitivity-methodology.pdf")
shutil.copyfile(src, dst)
print(f"[publish-pdf] copied {src} -> {dst}")
