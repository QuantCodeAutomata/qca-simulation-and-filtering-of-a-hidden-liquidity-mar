"""Experiment 2: Convergence of the impulse-control DP approximation W_N.

Implements and verifies the recursive DP approximation for the finite-horizon
impulse-control liquidation problem.  Records delta_N = max|g_N - g_{N-1}|
for each iteration and verifies monotone nondecreasing behaviour.
"""

import os
import sys
import csv
import json
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.control.intervention import run_dp_solver
from src.utils.plotting import plot_convergence

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def run_exp2(
    config_path: str = "/workspace/project/configs/filter_default.yaml",
    n_t: int = 6,
    n_x: int = 8,
    n_kp: int = 6,
    n_km: int = 6,
    n_mu: int = 7,
) -> dict:
    """Run Experiment 2 and return convergence metrics.

    Parameters
    ----------
    config_path:
        Path to YAML configuration.
    n_t, n_x, n_kp, n_km, n_mu:
        Grid sizes.  Defaults are chosen for a fast demonstration run.
        Increase for higher accuracy (slower).

    Returns
    -------
    dict with keys:
        'delta_N': array of max differences per iteration.
        'is_monotone': bool, True if g_N is nondecreasing in N.
        'converge_iter': iteration index where delta_N first falls below 1e-3.
        'g_solution': output of run_dp_solver (contains g_iterations and grids).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Running DP solver for convergence experiment...")
    g_solution = run_dp_solver(
        config_path=config_path,
        n_t=n_t,
        n_x=n_x,
        n_kp=n_kp,
        n_km=n_km,
        n_mu=n_mu,
    )

    delta_N = g_solution["delta_N"]
    g_iters = g_solution["g_iterations"]

    # --- Check monotone nondecreasing ---
    is_monotone = True
    min_pointwise_diff = np.inf
    for k in range(1, len(g_iters)):
        diff = g_iters[k] - g_iters[k - 1]
        min_diff = float(np.min(diff))
        if min_diff < -1e-10:
            is_monotone = False
        min_pointwise_diff = min(min_pointwise_diff, min_diff)

    # --- Convergence detection ---
    threshold = 1e-3
    converge_iter = None
    for k, d in enumerate(delta_N):
        if d < threshold:
            converge_iter = k + 1  # 1-indexed
            break
    if converge_iter is None:
        converge_iter = int(len(delta_N))  # no convergence detected

    print(f"\nConvergence results:")
    print(f"  Monotone nondecreasing: {is_monotone}")
    print(f"  Min pointwise diff (g_N - g_{{N-1}}): {min_pointwise_diff:.2e}")
    print(f"  Convergence iter (delta_N < {threshold}): {converge_iter}")
    print(f"  Final delta_N: {delta_N[-1]:.6e}")

    # --- Plot convergence ---
    plot_path = os.path.join(RESULTS_DIR, "exp2_convergence.png")
    plot_convergence(delta_N, save_path=plot_path)
    import matplotlib.pyplot as plt
    plt.close("all")

    # --- Save CSV ---
    csv_path = os.path.join(RESULTS_DIR, "exp2_convergence.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["iteration", "delta_N"])
        writer.writeheader()
        for k, d in enumerate(delta_N):
            writer.writerow({"iteration": k + 1, "delta_N": float(d)})

    # --- Save summary ---
    summary = {
        "delta_N": [float(d) for d in delta_N],
        "is_monotone": bool(is_monotone),
        "converge_iter": int(converge_iter),
        "final_delta_N": float(delta_N[-1]),
        "grid": {
            "n_t": n_t, "n_x": n_x, "n_kp": n_kp, "n_km": n_km, "n_mu": n_mu
        },
    }
    json_path = os.path.join(RESULTS_DIR, "exp2_summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Exp2 done.  Plots and CSV saved to {RESULTS_DIR}")

    return {
        "delta_N": delta_N,
        "is_monotone": is_monotone,
        "converge_iter": converge_iter,
        "g_solution": g_solution,
    }


if __name__ == "__main__":
    run_exp2()
