import numpy as np
import pytest

from applications.seu_sensitivity_study import config, data_preparation as preparation
from applications.seu_sensitivity_study.tests.test_assessment_scale import fixed_inputs


def assembly_inputs(group="venture", presentation_id=None):
    inputs = fixed_inputs(group)
    cells = config.build_cells(["venture", "hiring"] if group == "matched_rq5" else [group])
    if group == "matched_rq5":
        from applications.seu_sensitivity_study.confirmatory_analysis import matched_rq5_design
        design, columns = matched_rq5_design(cells)
    else:
        design, columns, _ = config.SEUSensitivityStudyConfig().design_matrix_for_pool(group)
    choices = {}
    for cell in cells:
        records = [{"problem_id": problem["id"], "presentation_id": presentation["presentation_id"],
                    "menu_size": problem["menu_size"], "difficulty_stratum": problem["difficulty_stratum"],
                    "chosen_position": 1, "chosen_item_id": presentation["order"][0],
                    "resolution_path": "answer_token"}
                   for problem in inputs["problems"] if problem["id"].startswith("VEN" if cell.pool_id == "venture" else "HIR")
                   for presentation in problem["presentations"]]
        choices[cell.cell_id] = {"cell_id": cell.cell_id, "pool_id": group, "choices": records}
    return dict(pool={"pool_id": group, "items": inputs["items"]},
                problem_set={"pool_id": group, "problems": inputs["problems"]}, choice_sets=choices,
                reduced_embeddings={item["id"]: np.zeros(1) for item in inputs["items"]},
                design_matrix=design, cell_ids=[cell.cell_id for cell in cells], K=3,
                include_menu_size=True, assessment_probabilities=inputs["probabilities"],
                cell_model_names=[cell.model_name for cell in cells], utility_values=[0., .5, 1.],
                design_column_names=columns, presentation_id=presentation_id,
                include_assessment_scale_reference=True)


@pytest.mark.parametrize("missing", ["row", "presentation"])
def test_production_missing_collection_records_fail_closed(missing):
    kwargs = assembly_inputs()
    if missing == "row":
        next(iter(kwargs["choice_sets"].values()))["choices"].pop()
    else:
        for choice_set in kwargs["choice_sets"].values():
            choice_set["choices"] = [row for row in choice_set["choices"] if row["presentation_id"] == 1]
    with pytest.raises(ValueError, match="missing or unexpected collection"):
        preparation.build_stan_data(**kwargs)


def test_full_assembly_binds_both_presentations():
    data, report = preparation.build_stan_data(**assembly_inputs())
    assert data["M_total"] == 5040
    assert report["observation_metadata_version"] == 2
    assert len(report["frozen_observation_reference"]["menus"]) == 140


def validate_all(data, report):
    from applications.seu_sensitivity_study import assessment_scale, ceiling_diagnostics, confirmatory_reporting

    design, columns, cells = config.SEUSensitivityStudyConfig().design_matrix_for_pool("venture")
    return [lambda: ceiling_diagnostics.validate_observations(data, report),
            lambda: assessment_scale.validate_retained_data(report["assessment_scale_reference"], data, report, group="venture"),
            lambda: confirmatory_reporting._validate_data(data, report, cells, design, columns, group="venture", variant="primary")]


@pytest.mark.parametrize("mode", ["foreign_exclusion", "audit_count", "audit_removed", "audit_reason", "missing_audit",
                                  "retained_also_excluded", "order", "excluded_order", "mapping_missing", "mapping_duplicate"])
def test_all_validators_reject_contradictory_evidence(mode):
    kwargs = assembly_inputs()
    first = next(iter(kwargs["choice_sets"].values()))
    first["choices"][0].update(chosen_position=None, chosen_item_id=None, resolution_path="unresolved")
    data, report = preparation.build_stan_data(**kwargs)
    audit = report["na_logs"][first["cell_id"]]
    if mode == "foreign_exclusion":
        report["exclusions"][0]["cell_id"] = "not-a-study-cell"
    elif mode == "audit_count":
        audit.update(na_count=0, na_rate=0., resolved=280)
    elif mode == "audit_removed":
        audit["removed_observations"] = []
    elif mode == "audit_reason":
        report["exclusions"][0]["reason"] = "cell_na_exclusion"
    elif mode == "missing_audit":
        report.pop("na_logs")
    elif mode == "retained_also_excluded":
        report["excluded_cells"].append(report["cell_ids"][0])
    elif mode in ("order", "excluded_order"):
        row = report["observations" if mode == "order" else "exclusions"][0]
        row["item_order"].reverse()
        if row["chosen_position"] is not None:
            row["chosen_position"] = row["item_order"].index(row["chosen_item_id"]) + 1
    elif mode == "mapping_missing":
        report["frozen_observation_reference"]["menus"][0]["presentations"].pop()
    else:
        mapping = report["frozen_observation_reference"]["menus"][0]["presentations"]
        mapping[1]["presentation_id"] = 1
    for validate in validate_all(data, report):
        with pytest.raises(ValueError):
            validate()


@pytest.mark.parametrize("missing", ["row", "presentation"])
def test_consistently_recounted_dropped_rows_still_fail(missing):
    data, report = preparation.build_stan_data(**assembly_inputs())
    keep = [index for index, row in enumerate(report["observations"])
            if (index != 0 if missing == "row" else row["presentation_id"] == 1)]
    report["observations"] = [report["observations"][index] for index in keep]
    for name in ("I", "y", "cell"):
        data[name] = [data[name][index] for index in keep]
    data["M_total"] = len(keep)
    data["M_per_cell"] = [data["cell"].count(index + 1) for index in range(data["J"])]
    sizes = np.sum(data["I"], axis=1)
    data["s"] = (sizes - sizes.mean()).tolist()
    for cell_id, count in zip(report["cell_ids"], data["M_per_cell"]):
        report["na_logs"][cell_id].update(total_observations=count, resolved=count, resolution_paths={"answer_token": count})
    for validate in validate_all(data, report):
        with pytest.raises(ValueError, match="complete frozen universe"):
            validate()


def test_whole_cell_exclusion_reconciles_resolved_and_unresolved():
    kwargs = assembly_inputs()
    cell_id = kwargs["cell_ids"][0]
    for row in kwargs["choice_sets"][cell_id]["choices"][:85]:
        row.update(chosen_position=None, chosen_item_id=None, resolution_path="unresolved")
    data, report = preparation.build_stan_data(**kwargs)
    assert report["excluded_cells"] == [cell_id]
    assert len(report["exclusions"]) == 280
    assert report["na_logs"][cell_id]["na_count"] == 85
    assert data["M_total"] == 4760
    for validate in validate_all(data, report):
        validate()


@pytest.mark.parametrize("group,presentation_id,count", [("venture", 1, 2520), ("hiring", 2, 2520),
                                                       ("matched_rq5", None, 2880)])
def test_selected_universe_and_pool_membership(group, presentation_id, count):
    from applications.seu_sensitivity_study.ceiling_diagnostics import validate_observations

    data, report = preparation.build_stan_data(**assembly_inputs(group, presentation_id))
    validate_observations(data, report)
    assert data["M_total"] == count
    if presentation_id is not None:
        report["observations"][0]["presentation_id"] = 3 - presentation_id
    else:
        row = report["observations"][0]
        other = next(menu for menu in report["frozen_observation_reference"]["menus"] if menu["pool_id"] == "hiring")
        row["problem_id"] = other["id"]
    with pytest.raises(ValueError):
        validate_observations(data, report)


def test_reduced_mode_remains_available_but_not_a3_complete():
    from applications.seu_sensitivity_study.ceiling_diagnostics import validate_observations

    kwargs = assembly_inputs()
    kwargs["include_assessment_scale_reference"] = False
    for choice_set in kwargs["choice_sets"].values():
        choice_set["choices"] = choice_set["choices"][:4]
    data, report = preparation.build_stan_data(**kwargs)
    assert data["M_total"] == 72
    assert report["observation_metadata_version"] == 1
    with pytest.raises(ValueError, match="version-2"):
        validate_observations(data, report)


def test_same_composition_menu_ids_are_not_deduplicated():
    from applications.seu_sensitivity_study import assessment_scale

    inputs = fixed_inputs("venture")
    first = inputs["problems"][0]
    second = next(menu for menu in inputs["problems"][1:]
                  if menu["family"] == first["family"] and menu["menu_size"] == first["menu_size"])
    second.update(item_ids=list(first["item_ids"]), presentations=first["presentations"])
    reference = preparation.build_observation_reference(assessment_scale.build_reference(**inputs), inputs["problems"])
    menus = {menu["id"]: menu for menu in reference["menus"]}
    assert menus[first["id"]]["item_ids"] == menus[second["id"]]["item_ids"]
    assert len(menus) == 140