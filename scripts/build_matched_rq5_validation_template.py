"""Build the exact preproduction design template for matched-item RQ5 recovery."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.append(str(Path(__file__).resolve().parents[1]))

from applications.seu_sensitivity_study import (
    confirmatory_analysis,
    config,
    data_preparation,
    schemas,
)


DEFAULT_SOURCE = Path("applications/seu_sensitivity_study/results")
DEFAULT_OUTPUT = Path("results/validation_inputs/matched_rq5/stan_data_size.json")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _placeholder_choice_sets(
    pool_id: str,
    problem_set: Dict[str, Any],
    cells: Sequence[config.CellSpec],
) -> Dict[str, Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for problem in problem_set["problems"]:
        for presentation in problem["presentations"]:
            records.append(
                {
                    "problem_id": problem["id"],
                    "presentation_id": presentation["presentation_id"],
                    "menu_size": problem["menu_size"],
                    "difficulty_stratum": problem["difficulty_stratum"],
                    "family": problem["family"],
                    "chosen_position": 1,
                    "chosen_item_id": presentation["order"][0],
                    "resolution_path": "answer_token",
                    "raw_response": "ANSWER: 1",
                }
            )
    return {
        cell.cell_id: {
            "cell_id": cell.cell_id,
            "pool_id": pool_id,
            "model_name": cell.model_name,
            "prompt_condition": cell.prompt_condition,
            "choices": records,
        }
        for cell in cells
        if cell.pool_id == pool_id
    }


def _assessment_probabilities(
    source: Path,
    pool_ids: Sequence[str],
) -> Dict[str, Dict[str, List[float]]]:
    probabilities: Dict[str, Dict[str, List[float]]] = {}
    for model in config.MODELS:
        model_probabilities: Dict[str, List[float]] = {}
        for pool_id in pool_ids:
            path = source / "pools" / pool_id / "assessments" / f"{model.slug}.json"
            payload = _read_json(path)
            for record in payload["assessments"]:
                if not record.get("parse_ok") or record.get("probabilities") is None:
                    raise ValueError(
                        f"Assessment {model.name}/{pool_id}/{record['item_id']} "
                        "has no parsed probabilities"
                    )
                model_probabilities[record["item_id"]] = record["probabilities"]
        probabilities[model.name] = model_probabilities
    return probabilities


def build_template(source: Path = DEFAULT_SOURCE) -> tuple[Dict[str, Any], Dict[str, Any]]:
    pool_ids = ("venture", "hiring")
    pools = {
        pool_id: _read_json(source / "pools" / pool_id / "pool.json")
        for pool_id in pool_ids
    }
    problems = {
        pool_id: _read_json(source / "pools" / pool_id / "problems.json")
        for pool_id in pool_ids
    }
    cells = config.build_cells(pool_ids)
    design_matrix, column_names = confirmatory_analysis.matched_rq5_design(cells)

    stan_data, report = data_preparation.build_matched_rq5_stan_data(
        venture_pool=pools["venture"],
        hiring_pool=pools["hiring"],
        venture_problem_set=problems["venture"],
        hiring_problem_set=problems["hiring"],
        venture_choice_sets=_placeholder_choice_sets("venture", problems["venture"], cells),
        hiring_choice_sets=_placeholder_choice_sets("hiring", problems["hiring"], cells),
        assessment_probabilities=_assessment_probabilities(source, pool_ids),
        design_matrix=design_matrix,
        cell_ids=[cell.cell_id for cell in cells],
        cell_model_names=[cell.model_name for cell in cells],
        design_column_names=column_names,
        utility_values=[0.0, 0.5, 1.0],
        K=3,
    )
    errors = schemas.validate_stan_data(
        stan_data, model="h_m01_size_assessment_anchored"
    )
    if errors:
        raise ValueError(f"Invalid matched RQ5 validation template: {errors}")
    return stan_data, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    stan_data, report = build_template(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stan_data, indent=2) + "\n")
    report_path = args.output.with_name("assembly_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Wrote {args.output}: J={stan_data['J']}, P={stan_data['P']}, "
        f"R={stan_data['R']}, M_total={stan_data['M_total']}"
    )


if __name__ == "__main__":
    main()