"""Runner for experiments 2, 3, and 4.

Runs the DP solver once and shares the solution across all three experiments
to avoid redundant computation.
"""

import os
import sys

sys.path.insert(0, "/workspace/project")

from src.experiments.dp_convergence import run_exp2
from src.experiments.exercise_regions import run_exp3
from src.experiments.order_size_maps import run_exp4

CONFIG = "/workspace/project/configs/filter_default.yaml"

if __name__ == "__main__":
    print("=" * 60)
    print("EXP 2: DP Convergence")
    print("=" * 60)
    result2 = run_exp2(
        config_path=CONFIG,
        n_t=6,
        n_x=8,
        n_kp=6,
        n_km=6,
        n_mu=7,
    )
    g_solution = result2["g_solution"]

    print("\n" + "=" * 60)
    print("EXP 3: Exercise / Continuation Regions")
    print("=" * 60)
    result3 = run_exp3(
        config_path=CONFIG,
        g_solution=g_solution,
        tol=1e-5,
        n_mu_plot=41,
        n_x_plot=40,
    )

    print("\n" + "=" * 60)
    print("EXP 4: Optimal Order Size Maps")
    print("=" * 60)
    result4 = run_exp4(
        config_path=CONFIG,
        g_solution=g_solution,
        n_mu_plot=31,
        n_x_plot=30,
    )

    print("\nAll experiments complete.")
    print(f"Results saved to: /workspace/project/results/")
