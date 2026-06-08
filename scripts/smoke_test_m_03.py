"""
Smoke test for m_03: parameterized model variants of m_0 with a Dirichlet
concentration scalar on delta.

Two checks:
1. Pipeline integration: ParameterRecovery wires `delta_concentration` into
   the inference data dict when the inference model is m_03 (and only then).
2. Baseline equivalence: with delta_concentration=1.0, m_03 recovery metrics
   match m_0 within MC noise on the same study design and seeds.

Run:
    python scripts/smoke_test_m_03.py
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.parameter_recovery import ParameterRecovery
from utils import (
    get_model_inference_hyperparams,
    get_model_sim_hyperparams,
    MODEL_PARAMETERS,
)
from utils.study_design import StudyDesign


def check_registry():
    print("\n[check_registry]")
    assert "m_03" in MODEL_PARAMETERS, "m_03 not registered in MODEL_PARAMETERS"
    assert "delta_concentration" in get_model_sim_hyperparams("m_03"), \
        "delta_concentration not in MODEL_SIM_HYPERPARAMS['m_03']"
    assert get_model_inference_hyperparams("m_03") == ["delta_concentration"], \
        "delta_concentration not in MODEL_INFERENCE_HYPERPARAMS['m_03']"
    assert get_model_inference_hyperparams("m_0") == [], \
        "m_0 should have no inference hyperparams"
    print("  registry entries OK")


def run_recovery(model_tag, n_iterations=5, alpha0=1.0, seed=2026):
    study = StudyDesign(M=25, K=3, D=5, R=15,
                       min_alts_per_problem=2, max_alts_per_problem=5,
                       feature_dist="normal", feature_params={"loc": 0, "scale": 1},
                       design_name=f"smoke_{model_tag}")
    rng = np.random.default_rng(seed)
    # Lock study design generation seed via numpy global state used inside StudyDesign
    np.random.seed(seed)
    study.generate()

    out_dir = f"/tmp/smoke_test_m_03_{model_tag}"
    if model_tag == "m_03":
        recovery = ParameterRecovery(
            inference_model_path="models/m_03.stan",
            sim_model_path="models/m_03_sim.stan",
            study_design=study,
            output_dir=out_dir,
            n_mcmc_samples=400,
            n_mcmc_chains=2,
            n_iterations=n_iterations,
            sim_hyperparams={"delta_concentration": alpha0},
            inference_hyperparams={"delta_concentration": alpha0},
        )
    else:
        recovery = ParameterRecovery(
            inference_model_path="models/m_0.stan",
            sim_model_path="models/m_0_sim.stan",
            study_design=study,
            output_dir=out_dir,
            n_mcmc_samples=400,
            n_mcmc_chains=2,
            n_iterations=n_iterations,
        )
    return recovery.run()


def aggregate_metrics(true_params_list, posterior_summaries, K=3):
    alpha_true = np.array([p["alpha"] for p in true_params_list])
    alpha_mean = np.array([s.loc["alpha", "Mean"] for s in posterior_summaries])
    alpha_rmse = float(np.sqrt(np.mean((alpha_mean - alpha_true) ** 2)))

    delta_rmses = []
    for k in range(K - 1):
        name = f"delta[{k+1}]"
        d_true = np.array([p["delta"][k] for p in true_params_list])
        d_mean = np.array([s.loc[name, "Mean"] for s in posterior_summaries])
        delta_rmses.append(float(np.sqrt(np.mean((d_mean - d_true) ** 2))))
    return {"alpha_rmse": alpha_rmse, "delta_rmses": delta_rmses}


def check_inference_data_injection():
    """Confirm the inference-stage data dict carries delta_concentration for
    m_03 and not for m_0.  We patch only the *inference* model's sample method
    so the sim stage runs normally; then we intercept the inference data dict
    before Stan is invoked."""
    print("\n[check_inference_data_injection]")
    from analysis.parameter_recovery import ParameterRecovery
    from cmdstanpy import CmdStanModel

    for inf_path, sim_path, tag, expect_dc in [
        ("models/m_03.stan", "models/m_03_sim.stan", "m_03", True),
        ("models/m_0.stan", "models/m_0_sim.stan", "m_0", False),
    ]:
        rec = ParameterRecovery(
            inference_model_path=inf_path,
            sim_model_path=sim_path,
            study_design=None,  # let it auto-create a small design
            output_dir=f"/tmp/smoke_inspect_{tag}",
            n_mcmc_samples=100,
            n_mcmc_chains=1,
            n_iterations=1,
            sim_hyperparams={"delta_concentration": 7.0} if tag == "m_03" else None,
            inference_hyperparams={"delta_concentration": 7.0} if tag == "m_03" else None,
        )
        # Provide a tiny study design so sim is fast.
        study = StudyDesign(M=10, K=3, D=3, R=8,
                           min_alts_per_problem=2, max_alts_per_problem=3)
        np.random.seed(0)
        study.generate()
        rec.study_design = study

        captured = {}
        original_inf_sample = rec.inference_model.sample

        def intercept(*args, **kwargs):
            captured["data"] = dict(kwargs.get("data", {}))
            raise RuntimeError("smoke-test-bail-after-capture")

        rec.inference_model.sample = intercept
        try:
            try:
                rec.run()
            except Exception:
                pass
        finally:
            rec.inference_model.sample = original_inf_sample

        assert "data" in captured, f"{tag}: inference sample was never called"
        inf_data = captured["data"]
        if expect_dc:
            assert inf_data.get("delta_concentration") == 7.0, \
                f"{tag} inference data missing or wrong delta_concentration: {inf_data.get('delta_concentration')}"
            print(f"  {tag}: delta_concentration=7.0 present in inference data")
        else:
            assert "delta_concentration" not in inf_data, \
                f"{tag} inference data unexpectedly has delta_concentration"
            print(f"  {tag}: delta_concentration absent from inference data")


def main():
    check_registry()
    check_inference_data_injection()

    print("\n[baseline equivalence: m_0 vs m_03 with delta_concentration=1.0]")
    print("  This compiles + runs ~5 small recovery iterations per model (~1 min each).")

    m0_truth, m0_summ = run_recovery("m_0", n_iterations=5)
    m03_truth, m03_summ = run_recovery("m_03", n_iterations=5, alpha0=1.0)

    m0_metrics = aggregate_metrics(m0_truth, m0_summ)
    m03_metrics = aggregate_metrics(m03_truth, m03_summ)
    print(f"  m_0  alpha RMSE = {m0_metrics['alpha_rmse']:.4f}, delta RMSE = {m0_metrics['delta_rmses']}")
    print(f"  m_03 alpha RMSE = {m03_metrics['alpha_rmse']:.4f}, delta RMSE = {m03_metrics['delta_rmses']}")
    # MC noise on n=5 is large; we just sanity-check that magnitudes are comparable.
    ratio = m03_metrics["alpha_rmse"] / max(m0_metrics["alpha_rmse"], 1e-9)
    print(f"  alpha RMSE ratio m_03/m_0 = {ratio:.2f} (expect roughly within [0.5, 2.0])")
    assert 0.3 < ratio < 3.0, "m_03 with delta_concentration=1 deviates too much from m_0"
    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    main()
