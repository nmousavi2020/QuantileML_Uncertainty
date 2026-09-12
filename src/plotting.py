import os
import numpy as np
import matplotlib.pyplot as plt

def plot_errorbars(y_med, y_low, y_up, OUT_PLOT_DIR):

    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "figure.titlesize": 24
    })
    idx = np.argsort(y_med)
    y_med = y_med[idx]
    y_low = y_low[idx]
    y_up = y_up[idx]

    x = np.arange(len(y_med))

    yerr_lower = np.maximum(0, y_med - y_low)
    yerr_upper = np.maximum(0, y_up - y_med)

    uncertainty = y_up - y_low
    mean_unc = np.mean(uncertainty)

    plt.figure(figsize=(11,7))

    # Large visible error bars
    plt.errorbar(
        x, y_med,
        yerr=[yerr_lower, yerr_upper],
        fmt='o',
        markersize=7,
        markerfacecolor='red',
        markeredgecolor='red',
        ecolor='grey',
        elinewidth=1.8,
        capsize=5,
        capthick=1.8,
        alpha=0.95,
        zorder=2
    )

    # Optional connecting line
    plt.plot(x, y_med, color='blue', linewidth=1.3, alpha=0.7, zorder=1)

    plt.yscale("log")

    plt.xlabel("Sorted Sample Index")
    plt.ylabel("Mass (Gt)")
    plt.title("Predicted Mass with Uncertainty (P10–P90)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    # Bottom-right mean uncertainty label
    plt.text(
        0.98, 0.05,
        f"Mean Uncertainty (linear): {mean_unc:.2f} Gt",
        transform=plt.gca().transAxes,
        fontsize=18,
        va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="black")
    )

    plt.tight_layout()

    path = os.path.join(OUT_PLOT_DIR, "errorbar_plot.png")
    plt.savefig(os.path.join(OUT_PLOT_DIR, "errorbar_plot.png"), dpi=300)

    plt.close()

    print(f"plot is saved here: {path}")
    print(f"---------------------------------------------------------------------------------")
