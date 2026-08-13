"""Experiment 1: Simulation and filtering of the hidden-liquidity Hawkes process.

For each combination of prior pi_0(regime=1) in {0.9, 0.5, 0.1} and intensity
setting (kappa^{1,+}, kappa^{1,-}) in {(5,1), (3,3), (1,5)}, this script:
  1. Simulates the hidden CTMC regime path.
  2. Simulates marked buy/sell events via Ogata thinning.
  3. Runs the finite-dimensional filter.
  4. Generates diagnostic plots.
  5. Optionally simulates the price path.
  6. Saves results to /workspace/project/results/.
"""

import os
import sys
import csv
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.ctmc import simulate_ctmc
from src.model.hawkes_marked import MarkovHawkesState, simulate_marked_hawkes
from src.model.filter import run_filter
from src.model.impact import simulate_price_path
from src.utils.plotting import plot_filter_paths, plot_price_path

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def run_exp1(
    config_path: str = "/workspace/project/configs/filter_default.yaml",
) -> dict:
    """Run Experiment 1 and return metrics dict.

    Returns
    -------
    dict mapping (prior, kp1, km1) -> per-run metrics.
    """
    cfg = _load_config(config_path)
    model = cfg["model"]
    T = float(model["T"])
    zeta = float(model["zeta"])
    beta = float(model["beta"])
    nu = float(model["nu"])
    rho = float(model["rho"])
    c0 = float(model["c0"])
    eta = float(model["eta"])
    lam_inf_1 = float(model["lambda_infty_1"])
    lam_inf_2 = float(model["lambda_infty_2"])
    c = float(model["c"])
    e = float(model["e"])
    psi = np.array(model["generator"], dtype=float)

    priors = cfg["filter"]["priors"]
    intensity_settings = cfg["filter"]["intensity_settings"]
    eps = float(cfg["filter"]["eps"])

    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    lambda_infty = np.array([lam_inf_1, lam_inf_2], dtype=float)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_metrics = {}
    csv_rows = []

    for prior in priors:
        for kp1_init, km1_init in intensity_settings:
            run_label = f"prior{prior:.1f}_kp{kp1_init:.0f}_km{km1_init:.0f}"
            print(f"\n=== Exp1: {run_label} ===")

            # --- 1. Simulate hidden regime ---
            init_state = 0  # regime 1 (0-indexed)
            hidden_times, hidden_states = simulate_ctmc(
                psi, T, init_state=init_state, rng=rng
            )

            # --- 2. Set up initial Hawkes state ---
            # Regime 1: (kp1_init, km1_init); Regime 2: (km1_init, kp1_init) by symmetry
            kappa_plus_init = np.array([kp1_init, km1_init], dtype=float)
            kappa_minus_init = np.array([km1_init, kp1_init], dtype=float)
            init_hawkes = MarkovHawkesState(
                kappa_plus_init, kappa_minus_init, lambda_infty, beta, eta, zeta
            )

            # --- 3. Simulate marked buy/sell events ---
            sim_result = simulate_marked_hawkes(
                T, init_hawkes, hidden_times, hidden_states,
                lambda_infty, beta, eta, zeta, rng=rng
            )
            event_times = sim_result["event_times"]
            event_sides = sim_result["event_sides"]
            event_marks = sim_result["event_marks"]

            # Build intensity paths on a dense grid for plotting
            t_plot = np.linspace(0.0, T, 500)
            kp_path = np.zeros((len(t_plot), 2))
            km_path = np.zeros((len(t_plot), 2))

            # Reconstruct by replaying from lambda_plus_path
            lp_raw = sim_result["lambda_plus_path"]
            lm_raw = sim_result["lambda_minus_path"]
            if lp_raw:
                lp_times = np.array([r[0] for r in lp_raw])
                lp_vals = np.array([r[1:] for r in lp_raw])  # shape (n,2)
                lm_times = np.array([r[0] for r in lm_raw])
                lm_vals = np.array([r[1:] for r in lm_raw])

                for gi, t in enumerate(t_plot):
                    idx = max(0, int(np.searchsorted(lp_times, t, side="right")) - 1)
                    kp_path[gi] = lp_vals[min(idx, len(lp_vals) - 1)]
                    km_path[gi] = lm_vals[min(idx, len(lm_vals) - 1)]
            else:
                kp_path[:] = kappa_plus_init
                km_path[:] = kappa_minus_init

            # --- 4. Run filter ---
            pi0 = np.array([prior, 1.0 - prior], dtype=float)
            filter_result = run_filter(
                event_times, event_sides, event_marks,
                T, psi, kappa_plus_init, kappa_minus_init,
                lambda_infty, beta, eta, zeta, pi0, eps=eps
            )

            ft = filter_result["filter_times"]
            posterior = filter_result["posterior"]
            fkp = filter_result["kappa_plus"]
            fkm = filter_result["kappa_minus"]

            # MAP regime (0-indexed)
            map_regime = np.argmax(posterior, axis=1)
            # True hidden regime at each filter time point
            true_regime_at_ft = np.array([
                int(hidden_states[max(0, np.searchsorted(hidden_times, t, side="right") - 1)])
                for t in ft
            ])
            frac_correct = float(np.mean(map_regime == true_regime_at_ft))
            final_pi1 = float(posterior[-1, 0])
            n_events = int(len(event_times))

            print(f"  Events: {n_events}, final pi(1): {final_pi1:.3f}, "
                  f"MAP accuracy: {frac_correct:.3f}")

            # --- 5. Price path ---
            t_price, S_p, D_p, P_p = simulate_price_path(
                event_times, event_sides, event_marks, T,
                nu=nu, rho=rho, c=c, e=e
            )

            # --- 6. Plots ---
            plot_title = (
                f"$\\pi_0(1)={prior:.1f}$, "
                f"$(\\kappa^{{1,+}}, \\kappa^{{1,-}})=({kp1_init:.0f},{km1_init:.0f})$, "
                f"events={n_events}"
            )
            plot_path = os.path.join(RESULTS_DIR, f"exp1_filter_{run_label}.png")
            plot_filter_paths(
                ft, hidden_states, hidden_times, posterior,
                event_times, event_sides, fkp, fkm,
                T, title=plot_title, save_path=plot_path
            )
            import matplotlib.pyplot as plt
            plt.close("all")

            price_path_file = os.path.join(
                RESULTS_DIR, f"exp1_price_{run_label}.png"
            )
            plot_price_path(t_price, S_p, D_p, P_p, save_path=price_path_file)
            plt.close("all")

            metrics = {
                "prior": prior,
                "kp1_init": kp1_init,
                "km1_init": km1_init,
                "n_events": n_events,
                "final_posterior_regime1": final_pi1,
                "fraction_correct_MAP": frac_correct,
                "mean_kappa_plus_1": float(np.mean(fkp[:, 0])),
                "mean_kappa_minus_1": float(np.mean(fkm[:, 0])),
            }
            all_metrics[(prior, kp1_init, km1_init)] = metrics
            csv_rows.append(metrics)

    # Save CSV metrics
    csv_path = os.path.join(RESULTS_DIR, "exp1_metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nExp1 done. Metrics saved to {csv_path}")
    return {"metrics": all_metrics, "n_runs": len(csv_rows)}


if __name__ == "__main__":
    run_exp1()
