"""Experiment 3: Exercise and continuation region geometry.

Computes and visualises the liquidation exercise/continuation regions as functions
of belief mu and inventory x, for each combination of prior and intensity setting.
Verifies regime symmetry and qualitative comparative statics.
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.control.intervention import compute_exercise_map, run_dp_solver
from src.utils.plotting import plot_exercise_region_2d

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def run_exp3(
    config_path: str = "/workspace/project/configs/filter_default.yaml",
    g_solution: dict = None,
    tol: float = 1e-5,
    n_mu_plot: int = 41,
    n_x_plot: int = 40,
) -> dict:
    """Run Experiment 3 and return region metrics.

    Parameters
    ----------
    config_path:
        Path to YAML configuration.
    g_solution:
        Pre-computed DP solution from run_dp_solver.  If None, a fresh
        small-grid solve is run.
    tol:
        Exercise tolerance: exercise where M_tilde_g >= g - tol.
    n_mu_plot:
        Number of belief points in the 2D exercise-region plot.
    n_x_plot:
        Number of inventory points in the 2D exercise-region plot.

    Returns
    -------
    dict with keys:
        'exercise_fractions': dict mapping (kp1,km1,prior) to fraction of
                              states classified as exercise.
        'symmetry_errors':    dict mapping (kp1,km1,prior) to regime-symmetry
                              check error.
        'g_solution':         DP solution used.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    model = cfg["model"]
    x_max = float(cfg["dp_solver"]["x_max"])

    if g_solution is None:
        print("No g_solution provided; running fresh small-grid DP solve...")
        g_solution = run_dp_solver(config_path=config_path, n_t=6, n_x=8,
                                   n_kp=6, n_km=6, n_mu=7)

    intensity_settings = cfg["filter"]["intensity_settings"]
    priors = cfg["filter"]["priors"]

    mu_plot = np.linspace(0.0, 1.0, n_mu_plot)
    x_plot = np.linspace(0.0, x_max, n_x_plot)

    # Use the first time grid point (t=0) for the exercise maps
    t0 = float(g_solution["t_grid"][0])

    exercise_fractions = {}
    symmetry_errors = {}

    import matplotlib.pyplot as plt

    for kp1_init, km1_init in intensity_settings:
        for prior in priors:
            key = (kp1_init, km1_init, prior)
            label = f"kp{kp1_init:.0f}_km{km1_init:.0f}_pi{prior:.1f}"
            print(f"  Exercise region: {label}")

            # Compute exercise map for (kp1, km1, t=0)
            ex_map = compute_exercise_map(
                g_solution, mu_plot, x_plot,
                kp1=kp1_init, km1=km1_init, t=t0, tol=tol
            )
            frac_exercise = float(np.mean(ex_map))
            exercise_fractions[key] = frac_exercise

            # Plot
            title = (
                f"Exercise region: "
                f"$(\\kappa^{{1,+}}, \\kappa^{{1,-}})=({kp1_init:.0f},{km1_init:.0f})$"
            )
            save_path = os.path.join(RESULTS_DIR, f"exp3_exercise_{label}.png")
            plot_exercise_region_2d(
                mu_plot, x_plot, ex_map,
                xlabel=r"Belief $\mu = \pi({\rm regime}\,1)$",
                ylabel=r"Inventory $x$",
                title=title,
                save_path=save_path,
            )
            plt.close("all")

            # --- Symmetry check ---
            # With kp2=km1, km2=kp1: the exercise region at (mu, x, kp1, km1)
            # should equal the exercise region at (1-mu, x, km1, kp1).
            ex_map_swap = compute_exercise_map(
                g_solution, 1.0 - mu_plot[::-1], x_plot,
                kp1=km1_init, km1=kp1_init, t=t0, tol=tol
            )
            sym_error = float(np.mean(np.abs(
                ex_map.astype(float) - ex_map_swap[::-1].astype(float)
            )))
            symmetry_errors[key] = sym_error
            print(f"    Fraction exercise: {frac_exercise:.3f}, "
                  f"Symmetry error: {sym_error:.4f}")

    # Save summary
    summary = {
        "exercise_fractions": {str(k): v for k, v in exercise_fractions.items()},
        "symmetry_errors": {str(k): v for k, v in symmetry_errors.items()},
        "tol": tol,
        "n_mu_plot": n_mu_plot,
        "n_x_plot": n_x_plot,
    }
    json_path = os.path.join(RESULTS_DIR, "exp3_summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Exp3 done.  Results saved to {RESULTS_DIR}")

    return {
        "exercise_fractions": exercise_fractions,
        "symmetry_errors": symmetry_errors,
        "g_solution": g_solution,
    }


if __name__ == "__main__":
    run_exp3()
