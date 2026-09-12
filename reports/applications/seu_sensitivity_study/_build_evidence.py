"""Export a publication-safe precollection snapshot; default to offline validation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUNDLE = HERE / "data/design_evidence.yml"
SNAPSHOT_DATE = "2026-09-12"
SOURCE_COMMIT = "2194432"
WAVE = "production-20260911-wave01"
PRODUCTION = ROOT / "applications/seu_sensitivity_study/results/production"
STAGE = PRODUCTION / "production_stages" / WAVE
FIGURES = ("assessed_eta_gaps.png", "recovery_diagnostics.png", "illustrative_softmax.png")
METRICS = ("max_rhat", "min_ess_bulk", "min_ess_tail", "min_ebfmi",
           "divergences", "treedepth_saturated_share")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Sources:
    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}
        self.headers: dict[str, str] = {}

    def text(self, path: Path) -> str:
        relative = path.resolve().relative_to(ROOT).as_posix()
        content = path.read_bytes()
        self.hashes[relative] = hashlib.sha256(content).hexdigest()
        return content.decode("utf-8")

    def json(self, path: Path):
        return json.loads(self.text(path))

    def manifest(self) -> list[dict]:
        return ([{"path": path, "sha256": value} for path, value in sorted(self.hashes.items())]
                + [{"path": path, "sha256": value, "scope": "cmdstan_header"}
                   for path, value in sorted(self.headers.items())])


def chain_header(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    lines = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            lines.append(line)
    return "".join(lines)


def sampling_metadata(sources: Sources, directory: Path, diagnostics: dict) -> dict:
    references = diagnostics.get("chain_files", [])
    if not references:
        return {"status": "unavailable", "reason": "No per-iteration chain references"}
    schedules, chain_ids = [], []
    for reference in references:
        path = directory / "chains/main" / Path(reference).name
        if not path.is_file():
            return {"status": "unavailable", "reason": "Referenced per-iteration chain header unavailable"}
        header = chain_header(path)
        sources.headers[path.relative_to(ROOT).as_posix()] = hashlib.sha256(header.encode("utf-8")).hexdigest()
        fields = dict(re.findall(r"^#\s*(\w+)\s*=\s*(.*?)\s*$", header, re.MULTILINE))
        converters = {"model": str, "num_warmup": int, "num_samples": int,
                      "delta": float, "max_depth": int, "thin": int, "id": int}
        if not converters.keys() <= fields.keys():
            return {"status": "unavailable", "reason": "Incomplete per-iteration chain header"}
        values = {key: convert(fields[key].split(" (Default)")[0]) for key, convert in converters.items()}
        chain_ids.append(values.pop("id"))
        schedules.append(values)
    require(len(set(chain_ids)) == len(chain_ids), "Duplicate chain identity")
    require(all(schedule == schedules[0] for schedule in schedules), "Inconsistent per-chain sampler schedules")
    return {"status": "available", **schedules[0], "n_chains": len(chain_ids)}


def sampling_schedules(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        sampling = row["sampling"]
        key = canonical_digest(sampling)
        groups.setdefault(key, {"sampling": sampling, "iterations": []})["iterations"].append(row["iteration"])
    return list(groups.values())


def safe_path(value: str) -> bool:
    return bool(value) and not Path(value).is_absolute() and ".." not in Path(value).parts and "\\" not in value


def publication_check(value) -> None:
    forbidden = {"raw_response", "chain_files", "api_key", "access_token", "authorization",
                 "provider_batch_id", "provider_request_id", "custom_id", "thinking_blocks"}
    if isinstance(value, dict):
        require(not forbidden.intersection(value), "Private field in publication bundle")
        for content in value.values():
            publication_check(content)
    elif isinstance(value, list):
        for content in value:
            publication_check(content)
    elif isinstance(value, str):
        require(not re.search(r"/Users/|/home/|[A-Za-z]:\\|sk-[A-Za-z0-9_-]{16,}|<thinking>|<think>", value),
                "Potential private content in publication bundle")
    elif isinstance(value, float):
        require(math.isfinite(value), "Non-finite numeric evidence")


def sampler_pass(row: dict) -> bool:
    return (all(math.isfinite(float(row[key])) for key in METRICS)
            and row["max_rhat"] < 1.01 and row["min_ess_bulk"] >= 400
            and row["min_ess_tail"] >= 400 and row["min_ebfmi"] >= 0.3
            and row["divergences"] == 0 and row["treedepth_saturated_share"] == 0)


def eta_gaps(pool: dict, model: str, size: int) -> list[float]:
    eta = {item_id: 0.5 * probabilities[1] + probabilities[2]
           for item_id, probabilities in pool["probabilities"][model].items()}
    gaps = []
    for menu in pool["menus"]:
        if menu["menu_size"] == size:
            ordered = sorted(eta[item_id] for item_id in menu["item_ids"])
            gaps.append(ordered[-1] - ordered[-2])
    return gaps


def paired_keys(pool: dict) -> list[tuple]:
    lookup = {item["id"]: item["matched_key"] for item in pool["items"]}
    return sorted(tuple(sorted(lookup[item_id] for item_id in menu["item_ids"]))
                  for menu in pool["menus"] if lookup[menu["item_ids"][0]] is not None)


def extract_recovery(sources: Sources) -> dict:
    campaigns = {}
    for name, suffix in (("venture", "venture_recovery_pilot_500"),
                         ("hiring", "hiring_recovery_pilot_500"),
                         ("matched_rq5", "matched_rq5_recovery")):
        root = ROOT / "results/parameter_recovery" / f"h_m01_size_assessment_anchored_{suffix}"
        config = sources.json(root / "config_info.json")
        rows, coverage_values = [], {}
        for iteration in range(1, 41):
            directory = root / f"iteration_{iteration}"
            diagnostics = sources.json(directory / "diagnostics.json")
            require(not diagnostics.get("invalid_diagnostic_parameters"), "Invalid saved diagnostics")
            sampling = sampling_metadata(sources, directory, diagnostics)
            archived = []
            for previous in sorted(root.glob(f"*/iteration_{iteration}/diagnostics.json")):
                archived.append(sampling_metadata(sources, previous.parent, sources.json(previous)))
            if archived:
                sampling["previous_schedules"] = archived
                known = [value for value in archived if value["status"] == "available"]
                sampling["run_role"] = (
                    "longer replacement" if sampling["status"] == "available" and known
                    and any(sampling["num_samples"] > value["num_samples"]
                            or sampling["num_warmup"] > value["num_warmup"] for value in known)
                    else "replacement; schedule comparison unavailable or not longer")
            else:
                sampling["run_role"] = "original (no superseded iteration archive)"
            truth = sources.json(directory / "true_parameters.json")
            posterior = {row[""]: row for row in csv.DictReader(io.StringIO(sources.text(directory / "posterior_summary.csv")))}
            if "min_ess_tail" not in diagnostics:
                diagnostics["min_ess_tail"] = min(float(posterior[parameter]["ESS_tail"])
                                                  for parameter in diagnostics["ess_bulk"])
            parameters = {"gamma0": truth["gamma0"], "sigma_cell": truth["sigma_cell"],
                          "gamma_size": truth["extras"]["gamma_size"]}
            parameters.update({f"gamma[{index}]": value for index, value in enumerate(truth["gamma"], 1)})
            parameters.update(truth.get("contrasts", {}))
            for parameter, actual in parameters.items():
                summary = posterior[parameter]
                lower, upper, mean = (float(summary[field]) for field in ("5%", "95%", "Mean"))
                coverage_values.setdefault(parameter, []).append((lower <= actual <= upper, mean - actual, upper - lower))
            size = posterior["gamma_size"]
            row = {"iteration": iteration, "sampling": sampling, **{key: diagnostics[key] for key in METRICS},
                   "nonfinite_proposals_total": diagnostics.get("nonfinite_proposals_total", 0),
                   "seconds": diagnostics["seconds"],
                   "gamma_size": {"truth": parameters["gamma_size"], "mean": float(size["Mean"]),
                                  "lower": float(size["5%"]), "upper": float(size["95%"])} }
            row["passes"] = sampler_pass(row)
            rows.append(row)
        coverage = {}
        for parameter, values in coverage_values.items():
            covered, errors, widths = zip(*values)
            coverage[parameter] = {"covered": sum(covered), "total": len(values),
                                   "coverage": float(np.mean(covered)), "bias": float(np.mean(errors)),
                                   "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                                   "mean_interval_width": float(np.mean(widths))}
        campaigns[name] = {
            "source_root": root.relative_to(ROOT).as_posix(),
            "last_launch_config": {key: config[key] for key in ("n_iterations", "n_mcmc_samples", "n_mcmc_warmup", "n_mcmc_chains", "adapt_delta", "max_treedepth", "J", "P", "M_total")},
            "sampling_schedules": sampling_schedules(rows),
            "iterations": rows, "passed": sum(row["passes"] for row in rows),
            "coverage": coverage,
        }
    return campaigns


def extract_contract(sources: Sources) -> tuple[list[dict], dict, list[dict]]:
    sys.path.insert(0, str(ROOT))
    from applications.seu_sensitivity_study.config import MODELS, build_cells
    from applications.seu_sensitivity_study.confirmatory_analysis import contract_manifest, matched_rq5_contract

    for filename in ("config.py", "confirmatory_analysis.py", "__init__.py", "pools.py", "schemas.py"):
        relative = Path("applications/seu_sensitivity_study") / filename
        current = sources.text(ROOT / relative)
        require(current == sources.text(STAGE / "repository" / relative), f"Contract source differs from staged revision: {filename}")
    columns = [f"model_{model.slug}" for model in MODELS[1:]] + ["prompt_seu_maximizing", "prompt_deliberative"]
    cells = build_cells(["venture"])
    design = [[float(cell.model_name == model.name) for model in MODELS[1:]] +
              [float(cell.prompt_condition == prompt) for prompt in ("seu_maximizing", "deliberative")] for cell in cells]
    contract = contract_manifest(columns, design)
    contrasts = []
    for pool_id in ("venture", "hiring"):
        for spec in contract["primary_contrasts"]:
            contrasts.append({**spec, "pool": pool_id,
                              "duplicate_sign_reversal": spec["contrast_id"] == "rq1_openai_flagship_minus_small"})
        contrasts.append({"contrast_id": "rq6_gamma_size", "research_question": "RQ6", "pool": pool_id,
                          "label": f"{pool_id}: log-alpha slope per added alternative",
                          "column_names": ["gamma_size"], "coefficients": [1.0],
                          "rope_half_width": contract["menu_size_rope_half_width"], "expected_direction": "two_sided",
                          "duplicate_sign_reversal": False})
    matched = matched_rq5_contract(build_cells(["venture", "hiring"]))
    contrasts.extend({**spec, "pool": "matched_rq5", "duplicate_sign_reversal": False} for spec in matched["contrasts"])
    models = [{"name": model.name, "slug": model.slug, "vendor": model.vendor, "tier": model.tier,
               "endpoint": model.endpoint, "request_params": model.request_params,
               "configured_temperature": model.temperature,
               "reasoning_token_reserve": model.reasoning_token_reserve,
               "substituted_for": model.substituted_for} for model in MODELS]
    return json.loads(json.dumps(contrasts)), json.loads(json.dumps(contract)), models


def extract() -> dict:
    sources = Sources()
    manifest = sources.json(STAGE / "preflight_manifest.json")
    require(manifest["wave_id"] == WAVE and manifest["git_commit"].startswith(SOURCE_COMMIT), "Wrong staged snapshot")
    config = sources.json(STAGE / "preflight_config.json")
    scientific_sources = []
    for relative in (Path("models") / f"{config['stan_model']}.stan",
                     Path("applications/seu_sensitivity_study/data_preparation.py"),
                     Path("applications/seu_sensitivity_study/PREREGISTRATION.md")):
        current = sources.text(ROOT / relative)
        staged = STAGE / "repository" / relative
        status = "not present in stage; current source hashed"
        if staged.is_file():
            require(current == sources.text(staged), f"Scientific source differs from staged revision: {relative}")
            status = "current equals frozen staged source"
        scientific_sources.append({"path": relative.as_posix(), "stage_equality": status})
    contrasts, contract, models = extract_contract(sources)
    from applications.seu_sensitivity_study.prompts import load_prompt_sets

    sources.text(ROOT / "applications/seu_sensitivity_study/prompts.py")
    pools, templates = {}, {}
    for pool_id in ("venture", "hiring"):
        directory = STAGE / "pools" / pool_id
        raw_pool = sources.json(directory / "pool.json")
        raw_menus = sources.json(directory / "problems.json")
        for filename in ("pool.json", "problems.json", "gate_report.json"):
            sources.text(PRODUCTION / "pools" / pool_id / filename)
            require(digest(PRODUCTION / "pools" / pool_id / filename) == digest(directory / filename), "Production and staged pool differ")
        pool = {"consequences": raw_pool["consequences"],
                "items": [{key: item[key] for key in ("id", "family", "text", "quality_label", "matched_key")} for item in raw_pool["items"]],
                "menus": raw_menus["problems"], "probabilities": {},
                "assessment_instruction": "neutral",
                "gate": sources.json(directory / "gate_report.json")}
        assessment_texts = {}
        for model in models:
            relative = Path("assessments") / f"{model['slug']}.json"
            raw = sources.json(directory / relative)
            sources.text(PRODUCTION / "pools" / pool_id / relative)
            require(digest(PRODUCTION / "pools" / pool_id / relative) == digest(directory / relative), "Production and staged assessments differ")
            require(raw["instruction"] == "neutral" and raw["model_name"] == model["name"], "Wrong assessment identity")
            require(all(row["parse_ok"] for row in raw["assessments"]), "Unparsed assessment in frozen source")
            pool["probabilities"][model["name"]] = {row["item_id"]: row["probabilities"] for row in raw["assessments"]}
            if model["name"] == "gpt-4o":
                assessment_texts = {row["item_id"]: row["text"] for row in raw["assessments"]}
        prompt_path = ROOT / "applications/seu_sensitivity_study/configs" / f"prompts_{pool_id}.yaml"
        sources.text(prompt_path)
        resolved = load_prompt_sets(pool_id, path=prompt_path)
        templates[pool_id] = {}
        for family, prompt in resolved.items():
            for key, value in prompt.fingerprint().items():
                require(manifest["prompt_hashes"][f"{pool_id}/{family}/{key}"] == value, "Prompt template differs from staged hash")
            templates[pool_id][family] = {"assessment_system": prompt.assessment_system,
                                        "assessment_user": prompt.assessment_user,
                                        "choice_system": prompt.choice_system, "choice_user": prompt.choice_user,
                                        "choice_instructions": prompt.choice_instructions}
        pool["example_assessments"] = assessment_texts
        pools[pool_id] = pool
    cells, examples = [], []
    matched_example_keys = None
    for pool_id, pool in pools.items():
        item_lookup = {item["id"]: item for item in pool["items"]}
        menu_lookup = {menu["id"]: menu for menu in pool["menus"]}
        for model in models:
            for prompt in ("neutral", "seu_maximizing", "deliberative"):
                cell_id = f"{model['slug']}_{prompt}_{pool_id}"
                path = STAGE / "requests" / f"{cell_id}.json"
                raw = sources.json(path)
                require(raw["request_hash"] == manifest["request_hashes"][cell_id], "Request hash differs from manifest")
                structured_hash = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                require(structured_hash == manifest["generated_hashes"][f"requests/{cell_id}.json"], "Staged request content changed")
                mapping = {row["custom_id"]: row for row in raw["mapping"]}
                require(len(mapping) == len(raw["requests"]) == 280, "Wrong request count")
                seen, matched_count = set(), 0
                settings = None
                for request in raw["requests"]:
                    location = mapping[request["custom_id"]]
                    menu = menu_lookup[location["problem_id"]]
                    order = location["item_order"]
                    expected_order = next(row["order"] for row in menu["presentations"] if row["presentation_id"] == location["presentation_id"])
                    require(order == expected_order, "Request mapping is not the frozen presentation order")
                    seen.add((menu["id"], location["presentation_id"]))
                    is_matched = item_lookup[order[0]]["matched_key"] is not None
                    matched_count += int(is_matched)
                    body = request.get("body", request.get("params"))
                    require(isinstance(body, dict), "Unknown request schema")
                    selected = {key: body[key] for key in ("model", "temperature", "max_tokens", "max_completion_tokens", "reasoning_effort", "thinking") if key in body}
                    require(settings is None or settings == selected, "Settings change within a cell")
                    settings = selected
                    if model["name"] != "gpt-4o" or prompt != "neutral" or location["presentation_id"] != 1:
                        continue
                    family = menu["family"]
                    if any(row["pool"] == pool_id and row["family"] == family for row in examples):
                        continue
                    keys = sorted(item_lookup[item_id]["matched_key"] for item_id in order) if is_matched else None
                    if is_matched and matched_example_keys is not None and keys != matched_example_keys:
                        continue
                    if is_matched:
                        matched_example_keys = keys
                    messages = [{"role": message["role"], "content": message["content"]} for message in body["messages"]]
                    require(all(isinstance(message["content"], str) for message in messages), "Non-text example request")
                    prompt_set = load_prompt_sets(pool_id)[family]
                    expected_user = prompt_set.render_choice("neutral", [pool["example_assessments"][item_id] for item_id in order])
                    require(messages[-1]["content"] == expected_user, "Example does not match item-keyed assessment order")
                    first = item_lookup[order[0]]
                    examples.append({"pool": pool_id, "family": family, "model": model["name"], "condition": prompt,
                                     "problem_id": menu["id"], "presentation_id": location["presentation_id"],
                                     "item_order": order, "matched_keys": keys,
                                     "choice_messages": messages,
                                     "assessment_item_id": first["id"],
                                     "assessment_messages": [{"role": "system", "content": prompt_set.assessment_system},
                                                             {"role": "user", "content": prompt_set.render_assessment(first["text"], pool["consequences"])}],
                                     "assessments": [{"item_id": item_id, "text": pool["example_assessments"][item_id],
                                                      "probabilities": pool["probabilities"][model["name"]][item_id]} for item_id in order]})
                require(len(seen) == 280 and matched_count == 80, "Missing presentations or wrong matched subset")
                cells.append({"cell_id": cell_id, "pool": pool_id, "model": model["name"], "prompt": prompt,
                              "request_count": len(raw["requests"]), "matched_request_count": matched_count,
                              "request_hash": raw["request_hash"], "settings": settings})
        del pool["example_assessments"]
    require(set(manifest["authorized_cell_ids"]) == {cell["cell_id"] for cell in cells}, "Manifest cell set differs")
    counts = {"cells": len(cells), "staged_choice_calls": sum(cell["request_count"] for cell in cells),
              "menus_per_pool": len(pools["venture"]["menus"]),
              "menus_per_size_per_pool": sum(menu["menu_size"] == 2 for menu in pools["venture"]["menus"]),
              "matched_keys": len({item["matched_key"] for item in pools["venture"]["items"] if item["matched_key"]}),
              "paired_menus": len(paired_keys(pools["venture"])),
              "matched_subset_calls": sum(cell["matched_request_count"] for cell in cells), "planned_fits": 15}
    fit_plan = [{"scope": scope, "variant": variant, "utility_middle": middle}
                for scope in ("venture", "hiring", "matched_rq5")
                for variant, middle in (("primary", 0.5), ("utility_low", 0.35), ("utility_high", 0.65),
                                        ("presentation_1_only", 0.5), ("presentation_2_only", 0.5))]
    data = {"schema_version": 1, "snapshot_date": SNAPSHOT_DATE, "source_commit": SOURCE_COMMIT,
            "manifest": {key: manifest[key] for key in ("wave_id", "git_commit", "created_at", "aggregate_hash", "config_hash")},
            "counts": counts,
            "count_status": "Cells and calls are observed in staged request files, not observed choices. Fifteen fits are planned, not fitted results.",
            "settings": {key: config[key] for key in ("pool_ids", "problems_per_family", "menu_sizes", "num_presentations", "presentation_mode", "K", "seed", "max_choice_tokens", "max_assessment_tokens", "collection_mode", "stan_model", "primary_utility_middle", "utility_middle_values", "reference_model", "reference_prompt")},
            "budget": {key: manifest["batch_budget"][key] for key in ("budget_ceiling_usd", "wave_reservation_usd", "reserved_total_usd")},
            "models": models, "cells": cells, "pools": pools, "examples": examples, "templates": templates,
            "contrasts": contrasts, "contract": contract, "planned_fit_variants": fit_plan,
            "recovery": extract_recovery(sources), "scientific_sources": scientific_sources,
            "limitations": [
                "Staging is not evidence of provider submission, completed choices, or spending authorization.",
                "Recovery is simulation evidence under the fitted model, not empirical production estimates or formal SBC.",
                "Coverage is recomputed from current per-iteration 5%/95% summaries and true parameters, including the latest venture iteration 16; historical E4 coverage tables may predate that rerun.",
                "Recovery schedules are read from per-iteration chain headers, including archived schedules where available; last_launch_config is not evidence of earlier settings. Missing headers are explicitly unavailable.",
                "Scientific source provenance covers the frozen Stan likelihood, data preparation, and preregistration including embedded amendments. Chain provenance hashes only initial CSV headers, not posterior draw bytes.",
                "Assessed eta gaps use model-specific neutral probabilities and utilities (0, 0.5, 1); saved embedding-axis gate gaps are different quantities.",
                "The OpenAI flagship-minus-small contrast is the exact negative of gpt-4o-mini minus gpt-4o, retained in the declared family, not independent evidence.",
                "Matched procurement and hiring stimuli share merit keys; task-specific text and assessments need not be identical.",
                "No raw provider responses, hidden thinking blocks, credentials, job identifiers, chain files, or absolute machine paths are exported.",
            ], "sources": sources.manifest()}
    return data


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(data: dict) -> None:
    if data["snapshot_date"] != SNAPSHOT_DATE or data["source_commit"] != SOURCE_COMMIT:
        raise ValueError("Unexpected snapshot identity")
    expected = {"cells": 36, "staged_choice_calls": 10080, "menus_per_pool": 140,
                "menus_per_size_per_pool": 35, "matched_keys": 24,
                "paired_menus": 40, "matched_subset_calls": 2880,
                "planned_fits": 15}
    if data["counts"] != expected:
        raise ValueError("Frozen design counts do not match the evidence")
    publication_check(data)
    require(data["manifest"]["wave_id"] == WAVE, "Wrong wave")
    require(len(data["cells"]) == 36 and sum(cell["request_count"] for cell in data["cells"]) == 10080, "Wrong cell totals")
    require(sum(cell["matched_request_count"] for cell in data["cells"]) == 2880, "Wrong matched request total")
    expected_fits = {(scope, variant, middle)
                     for scope in ("venture", "hiring", "matched_rq5")
                     for variant, middle in (("primary", 0.5), ("utility_low", 0.35),
                                             ("utility_high", 0.65), ("presentation_1_only", 0.5),
                                             ("presentation_2_only", 0.5))}
    actual_fits = [(row["scope"], row["variant"], row["utility_middle"])
                   for row in data["planned_fit_variants"]]
    require(len(actual_fits) == 15 and set(actual_fits) == expected_fits, "Wrong planned fit set")
    require(len(data["contrasts"]) == 26, "Wrong confirmatory family")
    require(Counter(spec["research_question"] for spec in data["contrasts"]) == {"RQ1": 14, "RQ2": 4, "RQ5": 6, "RQ6": 2}, "Wrong RQ family composition")
    for spec in data["contrasts"]:
        require(len(spec["coefficients"]) == len(spec["column_names"]), "Invalid contrast vector")
    for pool_id, pool in data["pools"].items():
        require(len(pool["items"]) == 60 and len({item["id"] for item in pool["items"]}) == 60, "Wrong frozen item count")
        require(len({item["matched_key"] for item in pool["items"] if item["matched_key"]}) == 24, "Wrong matched-key count")
        require(Counter(menu["menu_size"] for menu in pool["menus"]) == {2: 35, 4: 35, 6: 35, 8: 35}, "Unbalanced menu sizes")
        require(len(pool["probabilities"]) == 6, "Missing model assessments")
        for probabilities in pool["probabilities"].values():
            require(set(probabilities) == {item["id"] for item in pool["items"]}, "Assessment IDs do not match pool")
            for values in probabilities.values():
                require(len(values) == 3 and all(0 <= value <= 1 for value in values) and math.isclose(sum(values), 1.0, abs_tol=1e-6), "Invalid probability simplex")
        for menu in pool["menus"]:
            require(len(set(menu["item_ids"])) == menu["menu_size"], "Invalid menu membership")
            require(menu["presentations"][1]["order"] == list(reversed(menu["presentations"][0]["order"])), "Presentations are not reversals")
        for model in pool["probabilities"]:
            for size in (2, 4, 6, 8):
                require(len(eta_gaps(pool, model, size)) == 35, "Wrong assessed gap sample")
        require(pool["gate"]["passed"], "Saved pool gate did not pass")
        subset = [spec for spec in data["contrasts"] if spec["pool"] == pool_id]
        reverse = next(spec for spec in subset if spec["contrast_id"] == "rq1_openai_flagship_minus_small")
        reference = next(spec for spec in subset if spec["contrast_id"] == "rq1_gpt_4o_mini_minus_gpt_4o")
        require(reverse["coefficients"] == [-value for value in reference["coefficients"]] and reverse["duplicate_sign_reversal"], "OpenAI duplication not correctly flagged")
    require(paired_keys(data["pools"]["venture"]) == paired_keys(data["pools"]["hiring"]) and len(paired_keys(data["pools"]["venture"])) == 40, "Matched menus do not pair by merit key")
    require(len(data["examples"]) == 4, "Missing representative family example")
    for example in data["examples"]:
        pool = data["pools"][example["pool"]]
        require([row["item_id"] for row in example["assessments"]] == example["item_order"], "Example assessment order mismatch")
        menu = next(menu for menu in pool["menus"] if menu["id"] == example["problem_id"])
        presentation = next(row for row in menu["presentations"] if row["presentation_id"] == example["presentation_id"])
        require(example["item_order"] == presentation["order"], "Example does not match its frozen menu")
        template = data["templates"][example["pool"]][example["family"]]
        block = "\n\n".join(f"{index}. {row['text'].strip()}" for index, row in enumerate(example["assessments"], 1))
        expected_user = template["choice_user"].format(instruction=template["choice_instructions"]["neutral"].strip(),
                                                       assessments_list=block, n_max=len(example["item_order"]))
        require(example["choice_messages"] == [{"role": "system", "content": template["choice_system"].strip()},
                                               {"role": "user", "content": expected_user}], "Rendered example differs from frozen template")
        item = next(item for item in pool["items"] if item["id"] == example["assessment_item_id"])
        expected_assessment = template["assessment_user"].format(
            item_text=item["text"].strip(),
            consequence_lines="\n".join(f"  {index}. {label}" for index, label in enumerate(pool["consequences"], 1)),
            probability_format="PROBABILITIES: <p1>, <p2>, <p3>")
        require(example["assessment_messages"] == [{"role": "system", "content": template["assessment_system"]},
                                                   {"role": "user", "content": expected_assessment}], "Assessment example differs from frozen stimulus")
        for assessment in example["assessments"]:
            require(assessment["probabilities"] == pool["probabilities"][example["model"]][assessment["item_id"]], "Example q mismatch")
    require(len(data["recovery"]) == 3, "Missing recovery campaign")
    for campaign in data["recovery"].values():
        require(len(campaign["iterations"]) == 40 and campaign["passed"] == 40, "Recovery count differs from 40/40")
        require({row["iteration"] for row in campaign["iterations"]} == set(range(1, 41)), "Missing or duplicated recovery iteration")
        require(all(sampler_pass(row) and row["passes"] for row in campaign["iterations"]), "Current sampler gates do not pass")
        require(campaign["sampling_schedules"] == sampling_schedules(campaign["iterations"]), "Sampling schedule summary mismatch")
        sizes = [row["gamma_size"] for row in campaign["iterations"]]
        require(sum(row["lower"] <= row["truth"] <= row["upper"] for row in sizes) == campaign["coverage"]["gamma_size"]["covered"], "Coverage does not match retained intervals")
        for summary in campaign["coverage"].values():
            require(summary["total"] == 40 and 0 <= summary["covered"] <= 40
                    and summary["coverage"] == summary["covered"] / 40, "Inconsistent coverage aggregate")
    require(len({source["path"] for source in data["sources"]}) == len(data["sources"]), "Duplicate source path")
    for source in data["sources"]:
        require(safe_path(source["path"]) and re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is not None, "Invalid source provenance")


def numerical_figure_inputs(data: dict) -> dict:
    gaps = [{"pool": pool_id, "size": size, "model": model["name"],
             "gap": sorted(eta_gaps(pool, model["name"], size)),
             "fraction": (np.arange(1, 36) / 35).tolist()}
            for pool_id, pool in data["pools"].items() for size in (2, 4, 6, 8)
            for model in data["models"]]
    recovery = [{"campaign": name,
                 "iteration": [row["iteration"] for row in campaign["iterations"]],
                 "max_rhat": [row["max_rhat"] for row in campaign["iterations"]],
                 "min_ess_tail": [row["min_ess_tail"] for row in campaign["iterations"]],
                 "truth": [row["gamma_size"]["truth"] for row in campaign["iterations"]],
                 "mean": [row["gamma_size"]["mean"] for row in campaign["iterations"]]}
                for name, campaign in data["recovery"].items()]
    sensitivity = np.linspace(0, 30, 301)
    illustration = []
    for utilities in ([0.45, 0.55], [0.2, 0.8], [0.2, 0.4, 0.55, 0.8]):
        weights = np.exp(sensitivity[:, None] * (np.asarray(utilities) - max(utilities)))
        illustration.append({"utilities": utilities, "sensitivity": sensitivity.tolist(),
                             "probability": (weights[:, -1] / weights.sum(axis=1)).tolist()})
    return dict(zip(FIGURES, (gaps, recovery, illustration)))


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def figure_integrity(data: dict) -> dict:
    source = "\n".join(inspect.getsource(function) for function in
                       (eta_gaps, numerical_figure_inputs, plot_bundle))
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return {name: {"numerical_input_sha256": canonical_digest(values),
                   "plotting_source_sha256": source_hash}
            for name, values in numerical_figure_inputs(data).items()}


def plot_bundle(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inputs = numerical_figure_inputs(data)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.dpi": 150})
    directory = HERE / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    figure, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
    for row, (pool_id, pool) in enumerate(data["pools"].items()):
        for column, size in enumerate((2, 4, 6, 8)):
            axis = axes[row, column]
            for model, color in zip(data["models"], colors):
                curve = next(curve for curve in inputs[FIGURES[0]]
                             if (curve["pool"], curve["size"], curve["model"]) == (pool_id, size, model["name"]))
                axis.step(curve["gap"], curve["fraction"], where="post", color=color, label=model["name"])
            axis.set_title(f"{pool_id.capitalize()}, size {size}")
            axis.grid(alpha=0.18)
            if row == 1:
                axis.set_xlabel("Best minus runner-up eta")
            if column == 0:
                axis.set_ylabel("Fraction of frozen menus")
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9)
    figure.suptitle("Actual assessed utility gaps: 35 menus per size and model")
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    figure.savefig(directory / FIGURES[0])
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for index, series in enumerate(inputs[FIGURES[1]]):
        name = series["campaign"]
        for axis, metric, threshold in zip(axes[:2], ("max_rhat", "min_ess_tail"), (1.01, 400)):
            axis.scatter(series["iteration"], series[metric], s=17, alpha=0.7, color=colors[index], label=name)
            if index == 0:
                axis.axhline(threshold, color="black", linestyle="--", linewidth=1)
            axis.set_xlabel("Simulation iteration")
        axes[2].scatter(series["truth"], series["mean"], s=17, alpha=0.7, color=colors[index])
    axes[0].set_ylabel("Maximum R-hat")
    axes[1].set_ylabel("Minimum tail ESS")
    axes[1].set_yscale("log")
    limits = axes[2].get_xlim()
    axes[2].plot(limits, limits, "k--", linewidth=1)
    axes[2].set(xlabel="True menu-size slope", ylabel="Posterior mean slope")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3)
    figure.suptitle("Saved recovery: current iterations, including venture 16 rerun")
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    figure.savefig(directory / FIGURES[1])
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for curve, color in zip(inputs[FIGURES[2]], colors):
        axis.plot(curve["sensitivity"], curve["probability"], color=color, label=f"eta = {curve['utilities']}")
    axis.set(xlabel="SEU sensitivity alpha", ylabel="Probability of highest-eta alternative", ylim=(0, 1.02),
             title="Illustration only: softmax on invented utility menus")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(directory / FIGURES[2])
    plt.close(figure)


def table(headers: list[str], rows: list[list]) -> str:
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(clean(value) for value in row) + " |" for row in rows])


def recovery_tables(data: dict) -> str:
    sections = ["Sampling schedules below come from the retained per-iteration CSV headers. "
                "Original means no superseded iteration archive was found; longer replacements are compared "
                "with archived headers. The YAML's last_launch_config is not a campaign-wide schedule."]
    for name, campaign in data["recovery"].items():
        rows = []
        for group in campaign["sampling_schedules"]:
            sampling = group["sampling"]
            schedule = (f"{sampling['num_warmup']}/{sampling['num_samples']}" if sampling["status"] == "available"
                        else "unavailable per iteration")
            previous = "; ".join(f"{value['num_warmup']}/{value['num_samples']}"
                                 if value["status"] == "available" else "unavailable"
                                 for value in sampling.get("previous_schedules", [])) or "not archived"
            rows.append([", ".join(map(str, group["iterations"])), sampling["run_role"], schedule,
                         previous, sampling.get("model", "unavailable"), sampling.get("n_chains", "unavailable"),
                         sampling.get("max_depth", "unavailable"), sampling.get("delta", "unavailable")])
        sections += [f"#### {name}: sampling and full parameter recovery",
                     table(["Iterations", "Run", "Warmup/sampling per chain", "Archived warmup/sampling",
                            "Header model", "Chains", "Max depth", "Adapt delta"], rows),
                     table(["Parameter or contrast", "Covered datasets", "Coverage", "Bias", "RMSE", "Mean 90% interval width"],
                           [[f"`{parameter}`", f"{summary['covered']}/{summary['total']}",
                             f"{summary['coverage']:.3f}", f"{summary['bias']:.4f}", f"{summary['rmse']:.4f}",
                             f"{summary['mean_interval_width']:.4f}"]
                            for parameter, summary in campaign["coverage"].items()])]
    return "\n\n".join(sections)


def render_include(data: dict) -> str:
    sections = ["<!-- Generated by _build_evidence.py --refresh. Do not edit by hand. -->",
                "## Frozen Evidence Bundle {#sec-evidence-bundle}",
                f"Snapshot: **{SNAPSHOT_DATE}**. Source commit: `{SOURCE_COMMIT}`. Staged wave: `{WAVE}`. "
                "The portable [YAML bundle](data/design_evidence.yml) contains frozen item texts, neutral probability matrices, menu membership, exact examples, settings, contrast vectors, recovery summaries, and relative source SHA-256 hashes. No local raw inputs or Python execution are required to render this appendix.",
                "### Staged Design and Settings {#sec-evidence-settings}",
                data["count_status"],
                table(["Quantity", "Count", "Evidence status"], [[key.replace("_", " "), value, "planned" if key == "planned_fits" else "verified staged design"] for key, value in data["counts"].items()]),
                "There are 60 frozen items per pool and 720 parsed model-by-item neutral probability vectors across the two pools. "
                "Each of the 36 cells contains 280 staged requests; the matched subset contains 80 per cell. "
                "The manifest records a wave reservation bound of $" + f"{data['budget']['wave_reservation_usd']:.6f}" + ", not realized cost or evidence of submission.",
                table(["Arm", "Endpoint", "Actual staged choice settings"],
                      [[model["name"], model["endpoint"], "`" + json.dumps(next(cell["settings"] for cell in data["cells"] if cell["model"] == model["name"]), sort_keys=True) + "`"] for model in data["models"]]),
                "The nominal visible choice cap is 64 tokens and assessment cap is 400; actual provider completion caps above include reasoning/thinking headroom. "
                "Assessments are neutral and reused across the three choice instructions. Reversed presentations are not independent item replicates.",
                "### Frozen Stimuli and Rendered Requests {#sec-evidence-examples}",
                "Examples below are deterministic selections from the frozen GPT-4o neutral request files. "
                "Choice prompts are the actual staged strings, not rewritten examples. Assessment prompts are reconstructed from hash-verified frozen templates and item text. "
                "The procurement/hiring examples are paired by matched merit keys, not by array position."]
    for example in data["examples"]:
        pool = data["pools"][example["pool"]]
        identifier = f"{example['pool']}-{example['family']}"
        sections += [f"#### {example['pool'].capitalize()}: {example['family']} {{#sec-example-{identifier}}}",
                     f"Menu `{example['problem_id']}`, presentation {example['presentation_id']}; item order: " + ", ".join(f"`{item_id}`" for item_id in example["item_order"]) + "." +
                     (" Matched keys: " + ", ".join(example["matched_keys"]) + "." if example["matched_keys"] else ""),
                     "Consequence order: " + "; ".join(pool["consequences"]) + ".",
                     table(["Item ID", "Neutral probabilities", "Assessed eta"], [[row["item_id"], row["probabilities"], f"{0.5 * row['probabilities'][1] + row['probabilities'][2]:.4f}"] for row in example["assessments"]])]
        for item_id in example["item_order"]:
            item = next(item for item in pool["items"] if item["id"] == item_id)
            sections += [f"**Frozen stimulus `{item_id}`**", "```text\n" + item["text"] + "\n```"]
        for heading, key in (("Full neutral assessment prompt", "assessment_messages"), ("Full rendered choice prompt", "choice_messages")):
            sections.append(f"**{heading}**")
            for message in example[key]:
                sections += [f"{message['role'].capitalize()}:", "```text\n" + message["content"] + "\n```"]
    sections += ["### Assessed Utility Geometry {#sec-evidence-eta-gaps}",
                 "For each model and menu, eta is computed from that model's item-keyed neutral probabilities as $0.5q_2+q_3$. "
                 "The plotted gap is the best minus second-best eta within the actual menu. Each curve uses 35 distinct menus, once each, not both presentations. Ties have gap zero.",
                 "![Empirical distributions of actual assessed eta gaps, by pool, menu size, and model.](figures/assessed_eta_gaps.png){#fig-assessed-eta-gaps}",
                 "### Saved Gates and Recovery {#sec-evidence-recovery}",
                 "The frozen gate reports use an **embedding-derived quality axis**, with label-supervised LDA fallback; those gaps are not the assessed-eta gaps above.",
                 table(["Pool", "Saved gate", "Quality axis", "Cross-size gap maximum", "Cross-pool gap maximum"],
                       [[name, pool["gate"]["status"], pool["gate"]["quality_axis"], pool["gate"]["checks"]["eta_gap"]["cross_size"]["worst"], pool["gate"]["checks"]["eta_gap"]["cross_pool"]["worst"]] for name, pool in data["pools"].items()]),
                 "All **120/120** saved recovery datasets pass the current sampler gates: R-hat < 1.01, bulk and tail ESS >= 400, E-BFMI >= 0.3, zero divergences, and zero treedepth saturation. "
                 "Current iteration files include the authorized venture iteration-16 rerun. Coverage below is recomputed from saved central 90% intervals and simulation truths, not copied from the potentially pre-rerun E4 table.",
                 table(["Campaign", "Passes", "Worst R-hat", "Min bulk ESS", "Min tail ESS", "Min E-BFMI", "Size coverage"],
                       [[name, f"{campaign['passed']}/40", f"{max(row['max_rhat'] for row in campaign['iterations']):.5f}",
                         f"{min(row['min_ess_bulk'] for row in campaign['iterations']):.2f}", f"{min(row['min_ess_tail'] for row in campaign['iterations']):.2f}",
                         f"{min(row['min_ebfmi'] for row in campaign['iterations']):.3f}", f"{campaign['coverage']['gamma_size']['covered']}/40"] for name, campaign in data["recovery"].items()]),
                 "![Saved sampler diagnostics and menu-size parameter recovery; all points are actual current recovery metrics.](figures/recovery_diagnostics.png){#fig-saved-recovery}",
                 recovery_tables(data),
                 "These are recovery checks under the simulation model, not production effect estimates, power estimates, or a formal SBC campaign. Non-finite proposals during adaptation are retained in the YAML and are distinct from divergent transitions. Coverage at 40 datasets has coarse Monte Carlo resolution.",
                 "### Complete Confirmatory Family {#sec-evidence-contrasts}",
                 "All 26 declared decisions are listed, including the exact OpenAI sign reversal. Nonzero weights are shown below; full ordered coefficient vectors and design-column names are in the YAML. "
                 "Detection uses a central 90% interval excluding zero and a posterior median beyond the corresponding ROPE; expected direction does not change that two-sided classification rule. No multiplicity adjustment is applied. RQ3 and RQ4 remain descriptive.",
                 table(["Pool", "ID", "Label", "Nonzero coefficients", "ROPE", "Duplicate"],
                       [[spec["pool"], spec["contrast_id"], spec["label"], "; ".join(f"{name}: {weight:+g}" for name, weight in zip(spec["column_names"], spec["coefficients"]) if weight),
                         f"{spec['rope_half_width']:.6f}", "exact OpenAI sign reversal" if spec["duplicate_sign_reversal"] else ""] for spec in data["contrasts"]]),
                 "### Softmax Illustration {#sec-evidence-softmax}",
                 "![Illustrative softmax curves for invented utility menus; not production data or fitted curves.](figures/illustrative_softmax.png){#fig-illustrative-softmax}",
                 "### Frozen Prompt Templates {#sec-evidence-templates}"]
    for pool_id, families in data["templates"].items():
        for family, templates in families.items():
            sections += [f"#### {pool_id}/{family} templates", "```yaml\n" + yaml.safe_dump(templates, sort_keys=False, allow_unicode=True).rstrip() + "\n```"]
    sections += ["### Provenance and Limits {#sec-evidence-provenance}",
                 "Source SHA-256 hashes are recorded for each consumed artifact using repository-relative paths. "
                     "`--check` (also the default) validates the portable bundle, generated-file hashes, canonical numerical figure inputs, plotting source, and separate exporter integrity. "
                 "`--check-sources` additionally requires the local source artifacts; `--refresh` is the only mode that reads them to regenerate outputs.",
                     table(["Scientific source", "Frozen-stage comparison"],
                         [[source["path"], source["stage_equality"]] for source in data["scientific_sources"]]),
                 "\n".join("- " + limitation for limitation in data["limitations"])]
    return "\n\n".join(sections) + "\n"


def check_outputs(data: dict) -> None:
    require(data.get("figure_integrity") == figure_integrity(data), "Figure input or plotting source changed")
    require(data.get("exporter_integrity") == {"path": "_build_evidence.py", "sha256": digest(Path(__file__))},
            "Exporter source changed")
    require(set(data["outputs"]) == {"_evidence.qmd", *(f"figures/{name}" for name in FIGURES)}, "Missing generated outputs")
    for relative, expected_hash in data["outputs"].items():
        require(safe_path(relative) and digest(HERE / relative) == expected_hash, f"Generated output changed: {relative}")
    require((HERE / "_evidence.qmd").read_text(encoding="utf-8") == render_include(data), "Include differs from bundled evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refresh", action="store_true", help="Read local raw sources and replace the snapshot")
    mode.add_argument("--check", action="store_true", help="Validate the portable bundle only (default)")
    parser.add_argument("--check-sources", action="store_true", help="Also verify local source SHA-256 hashes")
    args = parser.parse_args()
    if args.refresh:
        data = extract()
        validate(data)
        plot_bundle(data)
        data["figure_integrity"] = figure_integrity(data)
        data["exporter_integrity"] = {"path": "_build_evidence.py", "sha256": digest(Path(__file__))}
        (HERE / "_evidence.qmd").write_text(render_include(data), encoding="utf-8")
        data["outputs"] = {relative: digest(HERE / relative) for relative in ("_evidence.qmd", *(f"figures/{name}" for name in FIGURES))}
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        BUNDLE.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    data = yaml.safe_load(BUNDLE.read_text(encoding="utf-8"))
    validate(data)
    check_outputs(data)
    if args.check_sources:
        for source in data["sources"]:
            path = ROOT / source["path"]
            actual = (hashlib.sha256(chain_header(path).encode("utf-8")).hexdigest()
                      if source.get("scope") == "cmdstan_header" else digest(path))
            if actual != source["sha256"]:
                raise ValueError(f"Source changed: {source['path']}")
    print("Evidence bundle validated (offline" + (", local hashes verified" if args.check_sources else "") + ").")


if __name__ == "__main__":
    main()