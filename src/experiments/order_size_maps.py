"""Experiment 4: Optimal order size dependence on inventory, belief, and impact.

Computes and visualises xi*(mu, x) for each combination of intensity setting
and impact parameter c in {0.7, 1.0}, reproducing the paper's policy maps.
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.control.intervention import compute_order_size_map, run_dp_solver
from src.utils.plotting import plot_order_size_heatmap

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def run_exp4(
    config_path: str = "/workspace/project/configs/filter_default.yaml",
    g_solution: dict = None,
    n_mu_plot: int = 31,
    n_x_plot: int = 30,
) -> dict:
    """Run Experiment 4 and return policy metrics.

    Parameters
    ----------
    config_path:
        Path to YAML configuration.
    g_solution:
        Pre-computed DP solution.  If None, a fresh small-grid solve is run.
    n_mu_plot:
        Belief grid resolution for 2-D plots.
    n_x_plot:
        Inventory grid resolution for 2-D plots.

    Returns
    -------
    dict with keys:
        'xi_star_maps': nested dict {(kp1,km1): {c_val: xi_star_array}}.
        'monotone_in_x': dict {(kp1,km1,c_val): bool}.
        'g_solution': DP solution used.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    model = cfg["model"]
    x_max = float(cfg["dp_solver"]["x_max"])
    c_values = cfg["dp_solver"]["c_sensitivity"]

    if g_solution is None:
        print("No g_solution provided; running fresh small-grid DP solve...")
        g_solution = run_dp_solver(config_path=config_path, n_t=6, n_x=8,
                                   n_kp=6, n_km=6, n_mu=7)

    intensity_settings = cfg["filter"]["intensity_settings"]

    mu_plot = np.linspace(0.0, 1.0, n_mu_plot)
    x_plot = np.linspace(0.0, x_max, n_x_plot)

    t0 = float(g_solution["t_grid"][0])

    xi_star_maps = {}
    monotone_in_x = {}
    summary_rows = []

    import matplotlib.pyplot as plt

    for kp1_init, km1_init in intensity_settings:
        xi_star_maps[(kp1_init, km1_init)] = {}
        for c_val in c_values:
            key = (kp1_init, km1_init, c_val)
            label = f"kp{kp1_init:.0f}_km{km1_init:.0f}_c{c_val:.1f}"
            print(f"  Order size map: {label}")

            xi_map = compute_order_size_map(
                g_solution, mu_plot, x_plot,
                kp1=kp1_init, km1=km1_init, t=t0,
                c_override=c_val
            )
            xi_star_maps[(kp1_init, km1_init)][c_val] = xi_map

            # Validation: xi* in [0, x] everywhere
            assert np.all(xi_map >= -1e-8), "xi* must be nonneg"
            assert np.all(
                xi_map <= x_plot[np.newaxis, :] + 1e-8
            ), "xi* must not exceed inventory"
            # xi*(x=0) = 0
            assert np.all(xi_map[:, 0] < 1e-8), "xi*(x=0) must be 0"

            # Check approximate monotonicity in x (fraction of mu slices monotone)
            mono_frac = float(np.mean([
                np.all(np.diff(xi_map[mi, :]) >= -1e-6)
                for mi in range(len(mu_plot))
            ]))
            monotone_in_x[key] = mono_frac > 0.8

            mean_xi = float(np.mean(xi_map))
            max_xi = float(np.max(xi_map))

            # Plot heatmap
            title = (
                f"Optimal order size: "
                f"$(\\kappa^{{1,+}}, \\kappa^{{1,-}})=({kp1_init:.0f},{km1_init:.0f})$, "
                f"$c={c_val:.1f}$"
            )
            save_path = os.path.join(RESULTS_DIR, f"exp4_xi_star_{label}.png")
            plot_order_size_heatmap(mu_plot, x_plot, xi_map, title=title,
                                    save_path=save_path)
            plt.close("all")

            summary_rows.append({
                "kp1": kp1_init, "km1": km1_init, "c": c_val,
                "mean_xi_star": mean_xi, "max_xi_star": max_xi,
                "mono_frac": mono_frac,
            })
            print(f"    mean xi*={mean_xi:.4f}, max xi*={max_xi:.4f}, "
                  f"mono_frac={mono_frac:.3f}")

    # Compare c=0.7 vs c=1.0: stronger impact -> smaller xi*
    impact_comparison = {}
    for kp1_init, km1_init in intensity_settings:
        xi_lo_c = xi_star_maps[(kp1_init, km1_init)].get(0.7, None)
        xi_hi_c = xi_star_maps[(kp1_init, km1_init)].get(1.0, None)
        if xi_lo_c is not None and xi_hi_c is not None:
            # Check the paper's claim: stronger impact => smaller optimal trade
            # In the paper's model, c is a depth parameter.
            # Larger c means more depth => less impact => larger trade.
            # So c=0.7 (less depth) => more impact => smaller trade.
            diff_mean = float(np.mean(xi_lo_c - xi_hi_c))
            impact_comparison[(kp1_init, km1_init)] = {
                "mean_diff_c0.7_minus_c1.0": diff_mean,
                "c0.7_smaller_than_c1.0": diff_mean < 0,
            }

    summary = {
        "summary_rows": summary_rows,
        "impact_comparison": {str(k): v for k, v in impact_comparison.items()},
    }
    json_path = os.path.join(RESULTS_DIR, "exp4_summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Exp4 done.  Results saved to {RESULTS_DIR}")

    return {
        "xi_star_maps": xi_star_maps,
        "monotone_in_x": monotone_in_x,
        "g_solution": g_solution,
    }


if __name__ == "__main__":
    run_exp4()
