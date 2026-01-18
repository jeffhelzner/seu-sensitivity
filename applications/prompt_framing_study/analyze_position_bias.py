#!/usr/bin/env python
"""Analyze position bias in prompt framing study results."""
import json
from pathlib import Path

results_dir = Path(__file__).parent / "results" / "run_20260117_184005"

with open(results_dir / "problems.json") as f:
    data = json.load(f)
problems = data["problems"]

with open(results_dir / "raw_choices.json") as f:
    choices = json.load(f)

variants = ["minimal", "baseline", "enhanced", "maximal"]

print("=" * 70)
print("POSITION BIAS ANALYSIS")
print("=" * 70)

# Overall position choice rates by variant
print("\nOVERALL POSITION CHOICE RATES:")
for variant in variants:
    pos_counts = [0, 0, 0, 0]
    for choice_data in choices[variant]["choices"]:
        pos = choice_data["choice"]
        if pos < 4:
            pos_counts[pos] += 1
    total = sum(pos_counts)
    rates = [c/total*100 for c in pos_counts]
    print(f"  {variant:10}: Pos1={rates[0]:.0f}% Pos2={rates[1]:.0f}% Pos3={rates[2]:.0f}% Pos4={rates[3]:.0f}%")

# For specific claims, check if position affects choice
print("\n" + "=" * 70)
print("CLAIM CHOICE RATE: POSITION 1 vs OTHER POSITIONS")
print("(Does the claim get chosen more when in position 1?)")
print("=" * 70)

for claim_id in ["C007", "C003", "C009", "C002", "C018"]:
    print(f"\n{claim_id}:")
    for variant in variants:
        pos1_shown = pos1_chosen = 0
        other_shown = other_chosen = 0
        
        for choice_data in choices[variant]["choices"]:
            pid = choice_data["problem_id"]
            choice_idx = choice_data["choice"]
            problem = next(p for p in problems if p["id"] == pid)
            
            if claim_id in problem["claim_ids"]:
                claim_pos = problem["claim_ids"].index(claim_id)
                if claim_pos == 0:
                    pos1_shown += 1
                    if choice_idx == 0:
                        pos1_chosen += 1
                else:
                    other_shown += 1
                    if choice_idx == claim_pos:
                        other_chosen += 1
        
        pos1_rate = pos1_chosen/pos1_shown*100 if pos1_shown > 0 else 0
        other_rate = other_chosen/other_shown*100 if other_shown > 0 else 0
        diff = pos1_rate - other_rate
        
        print(f"  {variant:10}: Pos1={pos1_chosen:2}/{pos1_shown:2} ({pos1_rate:5.1f}%)  "
              f"Other={other_chosen:2}/{other_shown:2} ({other_rate:5.1f}%)  "
              f"Diff={diff:+.1f}%")

print("\n" + "=" * 70)
print("INTERPRETATION:")
print("- If Pos1% >> Other%, there's position bias for that claim")
print("- If rates are similar across positions, choices are content-driven")
print("=" * 70)
