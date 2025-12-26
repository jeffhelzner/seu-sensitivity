#!/usr/bin/env python3
from cmdstanpy import CmdStanModel

print("Testing Stan compilation...")
try:
    model = CmdStanModel(stan_file="models/m_0.stan")
    print("✓ m_0.stan compiled successfully!")
except Exception as e:
    print(f"✗ m_0.stan failed: {e}")

try:
    model = CmdStanModel(stan_file="models/m_1_sim.stan")
    print("✓ m_1_sim.stan compiled successfully!")
except Exception as e:
    print(f"✗ m_1_sim.stan failed: {e}")
