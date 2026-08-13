"""Visualization utilities for the hidden-liquidity Hawkes model experiments."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Optional, List


# ---------------------------------------------------------------------------
# Experiment 1: Filter paths
# ---------------------------------------------------------------------------

def plot_filter_paths(
    t_grid: np.ndarray,
    hidden_states: np.ndarray,
    hidden_times: np.ndarray,
    posterior: np.ndarray,
    event_times: np.ndarray,
    event_sides: np.ndarray,
    kappa_plus_path: np.ndarray,
    kappa_minus_path: np.ndarray,
    T: float,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Six-panel diagnostic figure for one simulation run.

    Panels (top to bottom):
      1. Hidden regime I_t (step function).
      2. Posterior pi_t(regime=1) with MAP estimate overlay.
      3. Buy (+) and sell (-) event raster.
      4. Candidate buy intensities lambda^{1,+} and lambda^{2,+}.
      5. Candidate sell intensities lambda^{1,-} and lambda^{2,-}.
      6. (shared x-axis) MAP regime vs hidden regime agreement.

    Parameters
    ----------
    t_grid:
        Time points for posterior and intensity paths (1D).
    hidden_states:
        Active regime at each jump time in hidden_times (0-indexed, 1D).
    hidden_times:
        Regime-switch times including 0.0 (1D).
    posterior:
        Posterior pi_t at each t_grid point, shape (n_t, 2).
    event_times:
        Observed jump times (1D).
    event_sides:
        +1 / -1 for each jump (1D).
    kappa_plus_path:
        Candidate buy intensities at each t_grid point, shape (n_t, 2).
    kappa_minus_path:
        Candidate sell intensities at each t_grid point, shape (n_t, 2).
    T:
        Horizon.
    title:
        Figure suptitle.
    save_path:
        If given, save figure to this path.

    Returns
    -------
    fig: matplotlib Figure object.
    """
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    # Build step-function for hidden regime over t_grid
    h_times = np.asarray(hidden_times, dtype=float)
    h_states = np.asarray(hidden_states, dtype=int)
    hidden_at_t = np.array([
        int(h_states[max(0, np.searchsorted(h_times, t, side="right") - 1)])
        for t in t_grid
    ])

    # ------------------------------------------------------------------
    # Panel 1: Hidden regime
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.step(t_grid, hidden_at_t + 1, where="post", color="steelblue", lw=1.5,
            label="Hidden regime $I_t$")
    ax.set_ylim(0.5, 2.5)
    ax.set_yticks([1, 2])
    ax.set_ylabel("Regime $I_t$")
    ax.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------------
    # Panel 2: Posterior pi_t(regime=1) and MAP
    # ------------------------------------------------------------------
    ax = axes[1]
    pi1 = posterior[:, 0]
    map_regime = np.argmax(posterior, axis=1) + 1
    ax.plot(t_grid, pi1, color="darkorange", lw=1.5, label=r"$\pi_t(1)$")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r"$\pi_t(1)$")
    ax2 = ax.twinx()
    ax2.step(t_grid, map_regime, where="post", color="purple", lw=1.0,
             alpha=0.6, label="MAP regime")
    ax2.set_ylim(0.5, 2.5)
    ax2.set_yticks([1, 2])
    ax2.set_ylabel("MAP regime", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)

    # ------------------------------------------------------------------
    # Panel 3: Event raster
    # ------------------------------------------------------------------
    ax = axes[2]
    buy_times = event_times[event_sides > 0]
    sell_times = event_times[event_sides < 0]
    ax.vlines(buy_times, 0.5, 1.0, color="green", lw=0.5, alpha=0.7, label="Buy (+)")
    ax.vlines(sell_times, 0.0, 0.5, color="red", lw=0.5, alpha=0.7, label="Sell (-)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Events")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------------
    # Panel 4: Candidate buy intensities
    # ------------------------------------------------------------------
    ax = axes[3]
    ax.plot(t_grid, kappa_plus_path[:, 0], color="green", lw=1.2,
            label=r"$\lambda^{1,+}$")
    ax.plot(t_grid, kappa_plus_path[:, 1], color="lightgreen", lw=1.2,
            ls="--", label=r"$\lambda^{2,+}$")
    ax.set_ylabel(r"Buy $\lambda^{i,+}$")
    ax.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------------
    # Panel 5: Candidate sell intensities
    # ------------------------------------------------------------------
    ax = axes[4]
    ax.plot(t_grid, kappa_minus_path[:, 0], color="red", lw=1.2,
            label=r"$\lambda^{1,-}$")
    ax.plot(t_grid, kappa_minus_path[:, 1], color="salmon", lw=1.2,
            ls="--", label=r"$\lambda^{2,-}$")
    ax.set_ylabel(r"Sell $\lambda^{i,-}$")
    ax.set_xlabel("Time $t$")
    ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.set_xlim(0.0, T)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Experiment 2: DP convergence
# ---------------------------------------------------------------------------

def plot_convergence(
    delta_N: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Line plot of max-grid |W_N - W_{N-1}| versus iteration N.

    Parameters
    ----------
    delta_N:
        1-D array of length N_iter; delta_N[k] = max|g_{k+1} - g_k|.
    save_path:
        If given, save figure here.

    Returns
    -------
    fig: matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    iters = np.arange(1, len(delta_N) + 1)
    ax.semilogy(iters, delta_N, marker="o", color="steelblue", lw=1.8,
                ms=5, label=r"$\max|g_N - g_{N-1}|$")
    ax.axvline(13, color="orange", ls="--", lw=1.2, label="N=13 (typical convergence)")
    ax.set_xlabel("Iteration $N$")
    ax.set_ylabel(r"$\delta_N = \max_{\rm grid}|g_N - g_{N-1}|$")
    ax.set_title("DP Approximation Convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Experiment 3: Exercise region
# ---------------------------------------------------------------------------

def plot_exercise_region_2d(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    exercise_map: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Filled 2-D exercise/continuation region map.

    Parameters
    ----------
    grid_x:
        Values for the x-axis (1D, length M).
    grid_y:
        Values for the y-axis (1D, length N).
    exercise_map:
        Boolean array of shape (M, N); True = exercise (sell), False = continue.
    xlabel, ylabel:
        Axis labels.
    title:
        Plot title.
    save_path:
        Save destination.

    Returns
    -------
    fig: matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = ListedColormap(["#d0e8f5", "#f5c6a0"])  # blue=continue, orange=exercise
    C = exercise_map.astype(float).T  # shape (N, M) for pcolormesh
    mesh = ax.pcolormesh(
        grid_x, grid_y, C, cmap=cmap, vmin=0, vmax=1, shading="auto"
    )
    # Legend
    cont_patch = mpatches.Patch(color="#d0e8f5", label="Continuation")
    ex_patch = mpatches.Patch(color="#f5c6a0", label="Exercise (sell)")
    ax.legend(handles=[cont_patch, ex_patch], loc="best", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Experiment 4: Optimal order size heatmap
# ---------------------------------------------------------------------------

def plot_order_size_heatmap(
    grid_mu: np.ndarray,
    grid_x: np.ndarray,
    xi_star: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of optimal order size xi*(mu, x).

    Parameters
    ----------
    grid_mu:
        Belief grid (1D, length M).
    grid_x:
        Inventory grid (1D, length N).
    xi_star:
        Optimal order size, shape (M, N).
    title:
        Plot title.
    save_path:
        Save destination.

    Returns
    -------
    fig: matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    C = xi_star.T  # shape (N, M)
    mesh = ax.pcolormesh(grid_mu, grid_x, C, cmap="viridis", shading="auto")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(r"Optimal order size $\xi^*$")
    ax.set_xlabel(r"Belief $\mu = \pi({\rm regime}\,1)$")
    ax.set_ylabel(r"Inventory $x$")
    if title:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Price path
# ---------------------------------------------------------------------------

def plot_price_path(
    t_grid: np.ndarray,
    S_path: np.ndarray,
    D_path: np.ndarray,
    P_path: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Three-panel plot of permanent (S), transient (D), and total (P=S+D) price.

    Parameters
    ----------
    t_grid:
        Time grid (1D).
    S_path:
        Permanent price component (1D).
    D_path:
        Transient deviation (1D).
    P_path:
        Total price P = S + D (1D).
    save_path:
        Save destination.

    Returns
    -------
    fig: matplotlib Figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t_grid, S_path, color="steelblue", lw=1.2)
    axes[0].set_ylabel("Permanent $S_t$")
    axes[1].plot(t_grid, D_path, color="darkorange", lw=1.2)
    axes[1].set_ylabel("Transient $D_t$")
    axes[2].plot(t_grid, P_path, color="darkgreen", lw=1.2)
    axes[2].set_ylabel("Total $P_t = S_t + D_t$")
    axes[2].set_xlabel("Time $t$")
    fig.suptitle("Price Path Components", fontsize=12)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
