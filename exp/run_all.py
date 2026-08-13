"""Master runner: executes all four experiments in sequence and writes RESULTS.md."""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.filter_paths import run_exp1
from src.experiments.dp_convergence import run_exp2
from src.experiments.exercise_regions import run_exp3
from src.experiments.order_size_maps import run_exp4

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "filter_default.yaml")


def write_results_md(exp_results: dict, elapsed: dict) -> None:
    """Write a RESULTS.md summary of all experiment outcomes."""
    md_path = os.path.join(RESULTS_DIR, "RESULTS.md")

    m1 = exp_results.get("exp1", {})
    m2 = exp_results.get("exp2", {})
    m3 = exp_results.get("exp3", {})
    m4 = exp_results.get("exp4", {})

    delta_N = m2.get("delta_N", np.array([]))
    converge_iter = m2.get("converge_iter", "N/A")
    is_monotone = m2.get("is_monotone", "N/A")

    with open(md_path, "w") as fh:
        fh.write("# Experiment Results\n\n")
        fh.write(
            "Repository: qca-simulation-and-filtering-of-a-hidden-liquidity-mar\n\n"
        )
        fh.write("## Configuration\n\n")
        fh.write("| Parameter | Value |\n|---|---|\n")
        fh.write("| Horizon T | 20.0 |\n")
        fh.write("| Mark rate zeta | 2.0 |\n")
        fh.write("| Generator Psi | [[-0.2,0.2],[0.2,-0.2]] |\n")
        fh.write("| beta | 0.5 |\n")
        fh.write("| nu | 0.8 |\n")
        fh.write("| rho | 0.1 |\n")
        fh.write("| c0 | 0.1 |\n")
        fh.write("| eta | 0.05 |\n")
        fh.write("| lambda_infty | 1.0 |\n")
        fh.write("| c | 1.0 |\n")
        fh.write("| e | 3.0 |\n\n")

        fh.write("---\n\n")
        fh.write("## Experiment 1: Filter Paths\n\n")
        fh.write(
            "Simulates the hidden-liquidity Markov-modulated marked Hawkes process "
            "and runs the finite-dimensional filter for 9 (prior, intensity) combinations.\n\n"
        )
        metrics = m1.get("metrics", {})
        fh.write("| Prior | kp1 | km1 | Events | Final pi(1) | MAP accuracy |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for key, v in metrics.items():
            p, kp1, km1 = key
            fh.write(
                f"| {p:.1f} | {kp1:.0f} | {km1:.0f} | {v['n_events']} | "
                f"{v['final_posterior_regime1']:.3f} | {v['fraction_correct_MAP']:.3f} |\n"
            )
        fh.write(f"\n**Elapsed:** {elapsed.get('exp1', 0):.1f} s\n\n")

        fh.write("---\n\n")
        fh.write("## Experiment 2: DP Convergence\n\n")
        fh.write(
            "Verifies convergence of the impulse-control DP approximation g_N.\n\n"
        )
        fh.write(f"- **Monotone nondecreasing:** {is_monotone}\n")
        fh.write(f"- **Convergence iteration (delta_N < 1e-3):** {converge_iter}\n")
        if len(delta_N) > 0:
            fh.write(f"- **Final delta_N (N=20):** {float(delta_N[-1]):.6e}\n")
        fh.write("\n| Iteration | delta_N |\n|---|---|\n")
        for k, d in enumerate(delta_N):
            fh.write(f"| {k+1} | {float(d):.6e} |\n")
        fh.write(f"\n**Elapsed:** {elapsed.get('exp2', 0):.1f} s\n\n")

        fh.write("---\n\n")
        fh.write("## Experiment 3: Exercise and Continuation Regions\n\n")
        fh.write(
            "Computes exercise/continuation regions for each (kp1, km1, prior) combination.\n\n"
        )
        ex_frac = m3.get("exercise_fractions", {})
        sym_err = m3.get("symmetry_errors", {})
        fh.write("| kp1 | km1 | Prior | Exercise fraction | Symmetry error |\n")
        fh.write("|---|---|---|---|---|\n")
        for key in ex_frac:
            kp1, km1, prior = key
            fh.write(
                f"| {kp1:.0f} | {km1:.0f} | {prior:.1f} | "
                f"{ex_frac[key]:.3f} | {sym_err.get(key, float('nan')):.4f} |\n"
            )
        fh.write(f"\n**Elapsed:** {elapsed.get('exp3', 0):.1f} s\n\n")

        fh.write("---\n\n")
        fh.write("## Experiment 4: Optimal Order Size\n\n")
        fh.write(
            "Computes xi*(mu, x) policy maps for each (kp1, km1) and c in {0.7, 1.0}.\n\n"
        )
        mono_map = m4.get("monotone_in_x", {})
        fh.write("| kp1 | km1 | c | Monotone in x (>80%) |\n")
        fh.write("|---|---|---|---|\n")
        for key, v in mono_map.items():
            kp1, km1, c_val = key
            fh.write(f"| {kp1:.0f} | {km1:.0f} | {c_val:.1f} | {v} |\n")
        fh.write(f"\n**Elapsed:** {elapsed.get('exp4', 0):.1f} s\n\n")

        fh.write("---\n\n")
        fh.write("## Figures\n\n")
        fh.write(
            "All figures are saved to the `results/` directory:\n\n"
        )
        for fname in sorted(os.listdir(RESULTS_DIR)):
            if fname.endswith(".png"):
                fh.write(f"- `{fname}`\n")

    print(f"\nRESULTS.md written to {md_path}")


def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    elapsed = {}

    print("=" * 60)
    print("Running Experiment 1: Filter paths")
    print("=" * 60)
    t0 = time.time()
    m1 = run_exp1(CONFIG_PATH)
    elapsed["exp1"] = time.time() - t0

    print("\n" + "=" * 60)
    print("Running Experiment 2: DP convergence")
    print("=" * 60)
    t0 = time.time()
    m2 = run_exp2(CONFIG_PATH)
    elapsed["exp2"] = time.time() - t0

    g_sol = m2.get("g_solution")

    print("\n" + "=" * 60)
    print("Running Experiment 3: Exercise regions")
    print("=" * 60)
    t0 = time.time()
    m3 = run_exp3(CONFIG_PATH, g_solution=g_sol)
    elapsed["exp3"] = time.time() - t0

    print("\n" + "=" * 60)
    print("Running Experiment 4: Order size maps")
    print("=" * 60)
    t0 = time.time()
    m4 = run_exp4(CONFIG_PATH, g_solution=g_sol)
    elapsed["exp4"] = time.time() - t0

    all_results = {"exp1": m1, "exp2": m2, "exp3": m3, "exp4": m4}
    write_results_md(all_results, elapsed)

    print("\n" + "=" * 60)
    print("All experiments completed.")
    for k, v in elapsed.items():
        print(f"  {k}: {v:.1f} s")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    results = main()
