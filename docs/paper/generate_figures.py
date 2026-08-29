"""Generate paper figures from saved experiment data.

Figures:
    fig1.png - Separation result: target vs single-voice vs two-voice (already exists, regenerated for consistency)
    fig2.png - Ablation comparison: all three architectures overlaid on target
    fig3.png - Parameter saturation: theta values relative to bounds for all conditions
    fig4.png - Regional MSE decomposition: grouped bars by region and architecture

Usage:
    uv run python docs/paper/generate_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Style -----------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

PAPER_DIR = Path(__file__).parent
RESULTS_DIR = PAPER_DIR.parent.parent / "results"

# Colors
C_TARGET = "#1a1a1a"        # Near-black for target
C_COUPLED = "#dc2626"       # Red for coupled single-voice
C_UNCOUPLED = "#f59e0b"     # Amber for uncoupled single-voice
C_TWO_VOICE = "#2563eb"     # Blue for two-voice
C_GAP_SHADE = "#fee2e2"     # Light red for gap region shading
C_TRANSIENT_SHADE = "#e0f2fe"  # Light blue for transient shading

# Region boundaries (ms)
R_TRANSIENT_END = 25
R_GAP_END = 50
R_SUSTAIN_ATTACK_END = 80


def _shade_regions(ax, t_max):
    """Add region shading and labels."""
    ax.axvspan(0, R_TRANSIENT_END, alpha=0.15, color="#93c5fd", zorder=0)
    ax.axvspan(R_TRANSIENT_END, R_GAP_END, alpha=0.20, color="#fca5a5", zorder=0)
    ax.axvspan(R_GAP_END, t_max, alpha=0.08, color="#bbf7d0", zorder=0)

    y_label = 1.03
    ax.text(R_TRANSIENT_END / 2, y_label, "Transient",
            ha="center", va="bottom", fontsize=9, transform=ax.get_xaxis_transform())
    ax.text((R_TRANSIENT_END + R_GAP_END) / 2, y_label, "Gap",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#dc2626",
            transform=ax.get_xaxis_transform())
    # Place "Sustain" at a visible position (not at midpoint which may be off-screen)
    sustain_x = min((R_GAP_END + t_max) / 2, R_GAP_END + 60)
    ax.text(sustain_x, y_label, "Sustain",
            ha="center", va="bottom", fontsize=9, transform=ax.get_xaxis_transform())


# --- Figure 1: Separation result (coupled vs two-voice) ---------------

def fig1_separation():
    """Target vs coupled single-voice vs two-voice."""
    d = np.load(RESULTS_DIR / "exp_04" / "figure_data.npz")
    t, target, v_single, v_two = d["t_ms"], d["T_target"], d["v_single"], d["v_two"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _shade_regions(ax, t[-1])

    ax.plot(t, target, color=C_TARGET, linewidth=2.5, label=r"Target $T(t)$", zorder=3)
    ax.plot(t, v_single, color=C_COUPLED, linewidth=1.8,
            label=r"Single-voice best fit ($\mathcal{H}_{1c}$)", zorder=2)
    ax.plot(t, v_two, color=C_TWO_VOICE, linewidth=1.8,
            label=r"Two-voice best fit ($\mathcal{H}_2$)", zorder=2)

    # Metrics annotation
    bbox = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d1d5db", alpha=0.9)
    ax.text(0.98, 0.38, (
        "Overall loss:\n"
        f"  Single:   0.0196\n"
        f"  Two:      0.00093\n"
        f"  Ratio:    21\u00d7\n\n"
        "Gap MSE:\n"
        f"  Single:   0.152\n"
        f"  Two:      0.00005\n"
        f"  Ratio:    2888\u00d7"
    ), transform=ax.transAxes, fontsize=8.5, fontfamily="monospace",
       va="top", ha="right", bbox=bbox)

    ax.set_xlim(0, 200)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"Envelope $v(t)$")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.savefig(PAPER_DIR / "fig1.png")
    print("  fig1.png saved")
    plt.close(fig)


# --- Figure 2: Ablation comparison (all three architectures) ----------

def fig2_ablation():
    """Target vs coupled vs uncoupled vs two-voice."""
    d = np.load(RESULTS_DIR / "exp_05" / "figure_data.npz")
    t = d["t_ms"]
    target = d["target"]
    coupled = d["coupled"]
    uncoupled = d["uncoupled"]
    two_voice = d["two_voice"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _shade_regions(ax, t[-1])

    ax.plot(t, target, color=C_TARGET, linewidth=2.5, label=r"Target $T(t)$", zorder=4)
    ax.plot(t, coupled, color=C_COUPLED, linewidth=1.8,
            label=r"Coupled ($\mathcal{H}_{1c}$): loss = 0.0196", zorder=3)
    ax.plot(t, uncoupled, color=C_UNCOUPLED, linewidth=1.8,
            label=r"Uncoupled ($\mathcal{H}_{1u}$): loss = 0.0071", zorder=2)
    ax.plot(t, two_voice, color=C_TWO_VOICE, linewidth=1.8,
            label=r"Two-voice ($\mathcal{H}_2$): loss = 0.00093", zorder=2)

    # Annotate gap region difference — point at uncoupled trace in the gap
    gap_t = 35.0
    gap_idx = int(gap_t)
    if gap_idx < len(uncoupled):
        ax.annotate(
            f"510\u00d7\ngap ratio",
            xy=(gap_t, uncoupled[gap_idx]),
            xytext=(gap_t + 35, 0.42),
            fontsize=9, fontweight="bold", color="#92400e",
            arrowprops=dict(arrowstyle="->", color="#92400e", lw=1.5),
            ha="center",
        )

    ax.set_xlim(0, 200)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"Envelope $v(t)$")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82), framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.2)

    # Decomposition annotation
    bbox = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d1d5db", alpha=0.9)
    ax.text(0.98, 0.38, (
        "21.2\u00d7 total =\n"
        "  2.75\u00d7 (coupling)\n"
        "  \u00d7 7.7\u00d7 (single-state)"
    ), transform=ax.transAxes, fontsize=9, fontfamily="monospace",
       va="top", ha="right", bbox=bbox)

    fig.savefig(PAPER_DIR / "fig2.png")
    print("  fig2.png saved")
    plt.close(fig)


# --- Figure 3: Parameter saturation -----------------------------------

def fig3_parameters():
    """Theta values relative to bounds for all three conditions."""
    bounds = [(-2, 2), (-3, 3), (-2, 2), (-2, 2)]

    # --- Data ---
    coupled_vals = [-1.470, 3.000, 0.273, -2.000]
    coupled_at_bound = [False, True, False, True]

    uncoupled_vals = [1.635, -3.000, -0.202, 2.000]
    uncoupled_at_bound = [False, True, False, True]

    two_grace_vals = [-0.328, 0.119, -0.338, -0.493]
    two_main_vals = [-0.839, 2.306, 0.816, 0.466]

    # Labels per panel (strength params differ between coupled/uncoupled)
    coupled_labels = [
        r"$\theta_\mathrm{total}$",
        r"$\theta_\mathrm{bal}$",
        r"$\theta_{\tau_a}$",
        r"$\theta_{\tau_r}$",
    ]
    uncoupled_labels = [
        r"$\theta_\alpha$",
        r"$\theta_\beta$",
        r"$\theta_{\tau_a}$",
        r"$\theta_{\tau_r}$",
    ]

    n = 4
    y_pos = np.arange(n)[::-1]  # top-to-bottom reading order
    bar_h = 0.50

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

    def _draw_bounds(ax):
        for i in range(n):
            lo, hi = bounds[i]
            y = y_pos[i]
            ax.plot([lo, lo], [y - 0.35, y + 0.35], color="#374151", lw=2, zorder=4)
            ax.plot([hi, hi], [y - 0.35, y + 0.35], color="#374151", lw=2, zorder=4)

    def _draw_single_panel(ax, vals, at_bound, labels, title, bar_color, bound_color,
                           bounds_label):
        ax.set_title(title, fontsize=11)
        colors = [bar_color if at_bound[i] else "#6b7280" for i in range(n)]
        edges = [bound_color if at_bound[i] else "#374151" for i in range(n)]
        ax.barh(y_pos, vals, height=bar_h, color=colors, edgecolor=edges,
                linewidth=1.5, alpha=0.85, zorder=3)
        _draw_bounds(ax)

        for i in range(n):
            val, ab, y = vals[i], at_bound[i], y_pos[i]
            txt = f"{val:+.2f}"
            color = bound_color if ab else "#374151"
            weight = "bold" if ab else "normal"
            offset = 0.20 if val >= 0 else -0.20
            ha = "left" if val >= 0 else "right"
            ax.text(val + offset, y + 0.05, txt, va="center", ha=ha,
                    fontsize=10, fontweight=weight, color=color)
            if ab:
                ax.text(val + offset, y - 0.25, "AT BOUND", va="center", ha=ha,
                        fontsize=7, fontweight="bold", color=bound_color, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=12)
        ax.set_xlim(-4.5, 4.5)
        ax.axvline(0, color="#d1d5db", linewidth=0.8, zorder=1)
        ax.grid(True, axis="x", alpha=0.15)
        ax.text(0.5, -0.07, bounds_label, transform=ax.transAxes,
                ha="center", fontsize=10, fontweight="bold", color=bar_color)

    # --- Panel A: Coupled ---
    _draw_single_panel(
        axes[0], coupled_vals, coupled_at_bound, coupled_labels,
        r"Coupled $\mathcal{H}_{1c}$" + "\nloss = 0.0196",
        C_COUPLED, "#991b1b", "2/4 at bounds",
    )

    # --- Panel B: Uncoupled ---
    _draw_single_panel(
        axes[1], uncoupled_vals, uncoupled_at_bound, uncoupled_labels,
        r"Uncoupled $\mathcal{H}_{1u}$" + "\nloss = 0.0071",
        C_UNCOUPLED, "#92400e", "2/4 at bounds",
    )

    # --- Panel C: Two-voice ---
    ax = axes[2]
    ax.set_title(r"Two-voice $\mathcal{H}_2$" + "\nloss = 0.00093", fontsize=11)
    bar_h2 = 0.22
    ax.barh(y_pos + bar_h2 / 2 + 0.03, two_grace_vals, height=bar_h2,
            color="#93c5fd", edgecolor="#1d4ed8", linewidth=1.2, alpha=0.85,
            label="Grace voice", zorder=3)
    ax.barh(y_pos - bar_h2 / 2 - 0.03, two_main_vals, height=bar_h2,
            color="#1d4ed8", edgecolor="#1e3a5f", linewidth=1.2, alpha=0.85,
            label="Main voice", zorder=3)
    _draw_bounds(ax)
    for i in range(n):
        gv, mv, y = two_grace_vals[i], two_main_vals[i], y_pos[i]
        for val, y_off, color in [(gv, bar_h2 / 2 + 0.03, "#1d4ed8"),
                                   (mv, -(bar_h2 / 2 + 0.03), "#1e3a5f")]:
            offset = 0.15 if val >= 0 else -0.15
            ax.text(val + offset, y + y_off, f"{val:+.2f}",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=9, color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coupled_labels, fontsize=12)
    ax.set_xlim(-4.5, 4.5)
    ax.axvline(0, color="#d1d5db", linewidth=0.8, zorder=1)
    ax.grid(True, axis="x", alpha=0.15)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.text(0.5, -0.07, "0/8 at bounds", transform=ax.transAxes,
            ha="center", fontsize=10, fontweight="bold", color=C_TWO_VOICE)

    fig.suptitle("Parameter Saturation Across Architectures",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "fig3.png")
    print("  fig3.png saved")
    plt.close(fig)


# --- Figure 4: Regional MSE decomposition ----------------------------

def fig4_regional_mse():
    """Grouped bar chart of MSE by region for all three architectures."""
    regions = ["Transient\n(0\u201325 ms)", "Gap\n(25\u201350 ms)",
               "Sustain attack\n(50\u201380 ms)", "Sustain plateau\n(80+ ms)"]

    # Values from experiment report
    coupled_mse = [0.077, 0.152, 0.0002, 0.007]
    uncoupled_mse = [0.028, 0.027, 0.001, 0.001]
    two_voice_mse = [0.006, 0.00005, 0.0007, 0.0006]

    x = np.arange(len(regions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_c = ax.bar(x - width, coupled_mse, width, label=r"Coupled $\mathcal{H}_{1c}$",
                    color=C_COUPLED, edgecolor="#991b1b", linewidth=0.8, alpha=0.85)
    bars_u = ax.bar(x, uncoupled_mse, width, label=r"Uncoupled $\mathcal{H}_{1u}$",
                    color=C_UNCOUPLED, edgecolor="#92400e", linewidth=0.8, alpha=0.85)
    bars_t = ax.bar(x + width, two_voice_mse, width, label=r"Two-voice $\mathcal{H}_2$",
                    color=C_TWO_VOICE, edgecolor="#1d4ed8", linewidth=0.8, alpha=0.85)

    # Log scale to show the huge gap differences
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.5)

    # Ratio annotations above gap bars
    # Coupled/Two gap
    ax.annotate(f"2888\u00d7", xy=(1 - width, coupled_mse[1]),
                xytext=(1 - width, coupled_mse[1] * 2.5),
                ha="center", fontsize=9, fontweight="bold", color=C_COUPLED)
    # Uncoupled/Two gap
    ax.annotate(f"510\u00d7", xy=(1, uncoupled_mse[1]),
                xytext=(1, uncoupled_mse[1] * 2.5),
                ha="center", fontsize=9, fontweight="bold", color=C_UNCOUPLED)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=10)
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Regional MSE Decomposition", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.2)

    # Highlight gap region background
    ax.axvspan(0.5, 1.5, alpha=0.08, color="#fca5a5", zorder=0)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "fig4.png")
    print("  fig4.png saved")
    plt.close(fig)


# --- Main -------------------------------------------------------------

if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_separation()
    fig2_ablation()
    fig3_parameters()
    fig4_regional_mse()
    print("Done.")
