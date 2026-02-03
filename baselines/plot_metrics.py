import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Data
# -------------------------
methods = [
    "CoT-Basic",
    "CoT-Self-Reflection",
    "ToT"
]

accuracy = np.array([
    0.358600583090379,
    0.25268817204301075,
    0.25268817204301075
])

tpr = np.array([
    0.5278969957081545,
    0.4034334763948498,
    0.4034334763948498
])

fpr = np.array([
    0.4721030042918455,
    0.5965665236051502,
    0.5965665236051502
])

# -------------------------
# Plot settings
# -------------------------
x = np.arange(len(methods))
bar_width = 0.6

plt.figure(figsize=(15, 4))

label_font = 14
tick_font = 12
title_font = 16
grid_alpha = 0.4

# Deep, paper-friendly colors
colors = {
    "accuracy": "#4C72B0",   # muted blue
    "tpr": "#55A868",        # muted green
    "fpr": "#C44E52"         # muted red
}

edge_color = "black"
edge_width = 1.2

# -------------------------
# Subplot 1: Accuracy
# -------------------------
plt.subplot(1, 3, 1)
plt.bar(
    x, accuracy,
    width=bar_width,
    color=colors["accuracy"],
    edgecolor=edge_color,
    linewidth=edge_width
)
plt.xticks(x, methods, rotation=20, fontsize=tick_font)
plt.yticks(fontsize=tick_font)
plt.ylim(0, 1)
plt.ylabel("Score (%)", fontsize=label_font)
plt.title("Accuracy", fontsize=title_font)
plt.grid(axis="y", linestyle="--", alpha=grid_alpha)

# -------------------------
# Subplot 2: TPR
# -------------------------
plt.subplot(1, 3, 2)
plt.bar(
    x, tpr,
    width=bar_width,
    color=colors["tpr"],
    edgecolor=edge_color,
    linewidth=edge_width
)
plt.xticks(x, methods, rotation=20, fontsize=tick_font)
plt.yticks(fontsize=tick_font)
plt.ylim(0, 1)
plt.title("True Positive Rate (TPR)", fontsize=title_font)
plt.grid(axis="y", linestyle="--", alpha=grid_alpha)

# -------------------------
# Subplot 3: FPR ONLY
# -------------------------
plt.subplot(1, 3, 3)
plt.bar(
    x, fpr,
    width=bar_width,
    color=colors["fpr"],
    edgecolor=edge_color,
    linewidth=edge_width
)
plt.xticks(x, methods, rotation=20, fontsize=tick_font)
plt.yticks(fontsize=tick_font)
plt.ylim(0, 1)
plt.title("False Positive Rate (FPR)", fontsize=title_font)
plt.grid(axis="y", linestyle="--", alpha=grid_alpha)

plt.tight_layout()

plt.savefig("metrics_comparison.jpg", dpi = 700)
plt.show()
