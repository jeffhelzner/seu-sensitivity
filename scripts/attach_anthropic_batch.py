"""Attach an operator-identified Anthropic Batch to ambiguous durable state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from applications.seu_sensitivity_study.batch_client import ProviderBatchClient
from applications.seu_sensitivity_study.config import SEUSensitivityStudyConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--operator-note", required=True)
    args = parser.parse_args()

    config = SEUSensitivityStudyConfig.from_yaml(str(args.config))
    matches = [cell for cell in config.cells if cell.cell_id == args.cell_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one configured cell for {args.cell_id!r}")
    state = ProviderBatchClient(matches[0]).attach_ambiguous_batch(
        state_path=args.state,
        batch_id=args.batch_id,
        operator_note=args.operator_note,
    )
    print(
        f"Attached {state['batch_id']} to {args.cell_id}; "
        "rerun the normal choices command to retrieve it"
    )


if __name__ == "__main__":
    main()