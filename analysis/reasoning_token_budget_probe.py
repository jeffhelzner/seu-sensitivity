"""Measure the token budget o3-mini actually needs to emit a visible answer."""

import os
import sys
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
dotenv.load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import openai

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CLAIM = (
    "A homeowner filed a claim for water damage to their basement, stating that "
    "a pipe burst during a cold snap. The claim includes $15,000 for flooring "
    "replacement and $8,000 for damaged furniture. Photos show extensive water "
    "marks on walls but the pipe appears to be in good condition in provided "
    "images."
)
PROMPT = (
    f"Review the following insurance claim and assess how likely each possible "
    f"outcome is.\n\nClaim:\n{CLAIM}\n\nPossible outcomes:\n"
    "0: Neither investigator agrees the claim warrants investigation\n"
    "1: One investigator agrees the claim warrants investigation\n"
    "2: Both investigators agree the claim warrants investigation\n\n"
    "Give a brief assessment in 2-4 sentences, then end your reply with a "
    "single line in exactly this format:\nPROBABILITIES: p0, p1, p2"
)

print(f"{'budget':>8} {'reasoning':>10} {'visible':>8} {'total':>7}  text")
for budget in (400, 1000, 2000, 3000):
    r = client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "user", "content": PROMPT}],
        max_completion_tokens=budget,
    )
    usage = r.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    text = (r.choices[0].message.content or "").strip().replace("\n", " ")
    visible = usage.completion_tokens - (reasoning or 0)
    print(
        f"{budget:>8} {str(reasoning):>10} {visible:>8} "
        f"{usage.completion_tokens:>7}  {text[:70]!r}"
    )
