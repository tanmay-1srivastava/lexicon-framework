# grouped_means_paper_bars.py
# Uses the table values we already computed (no parsing).
# Groups: doctor / friends / work, plots MEAN TPR, MEAN FNR, MEAN Accuracy (proxy),
# and reports runtime MEDIAN + STD (overall + per group, printed + shown on plot).

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Per-file values (from the table)
# -----------------------------
files = [
    "doctor_visit_001",
    "doctor_visit_003",
    "friends_meeting_002",
    "friends_meeting_003",
    "work_collaboration_002",
    "work_collaboration_003",
]

# TPR/Recall (0..1)
tpr = np.array([0.8182, 0.1923, 0.5000, 0.6000, 0.3684, 0.4706])

# FNR (0..1)
fnr = np.array([0.1818, 0.8077, 0.5000, 0.4000, 0.6316, 0.5294])

# Accuracy: not identifiable without FP/TN from your summary.
# Here we use "Accuracy (proxy) = matching/total_gt" which equals TPR in your table.
acc = tpr.copy()

# Runtime seconds
runtime_s = np.array([14.7798, 10.5593, 16.5813, 18.1258, 16.0917, 8.2536])

# -----------------------------
# Grouping
# -----------------------------
groups = {
    "Doctor": np.array([i for i, f in enumerate(files) if f.startswith("doctor_visit")]),
    "Friends": np.array([i for i, f in enumerate(files) if f.startswith("friends_meeting")]),
    "Work": np.array([i for i, f in enumerate(files) if f.startswith("work_collaboration")]),
}

def mean_by_group(arr):
    out = []
    for g in groups.values():
        out.append(float(np.mean(arr[g])) if len(g) else 0.0)
    return np.array(out)

def median_std_by_group(arr):
    med = []
    std = []
    for g in groups.values():
        if len(g):
            med.append(float(np.median(arr[g])))
            std.append(float(np.std(arr[g], ddof=1)) if len(g) > 1 else 0.0)
        else:
            med.append(0.0)
            std.append(0.0)
    return np.array(med), np.array(std)

group_names = list(groups.keys())

tpr_mean = mean_by_group(tpr)
fnr_mean = mean_by_group(fnr)
acc_mean = mean_by_group(acc)

rt_median_overall = float(np.median(runtime_s))
rt_std_overall = float(np.std(runtime_s, ddof=1)) if len(runtime_s) > 1 else 0.0
rt_median_g, rt_std_g = median_std_by_group(runtime_s)

# -----------------------------
# Plot styling (paper-ish, minimal)
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

def annotate(ax, bars, fmt="{:.2f}", ypad=0.01):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width()/2,
            h + ypad,
            fmt.format(h),
            ha="center", va="bottom", fontsize=9
        )

# -----------------------------
# 1) Grouped mean metrics bar chart
# -----------------------------
x = np.arange(len(group_names))
w = 0.25

fig = plt.figure(figsize=(8.8, 5.2))
ax = fig.add_subplot(111)

b1 = ax.bar(x - w, tpr_mean, width=w, label="Mean TPR")
b2 = ax.bar(x, fnr_mean, width=w, label="Mean FNR")
b3 = ax.bar(x + w, acc_mean, width=w, label="Mean Accuracy")

ax.set_xticks(x)
ax.set_xticklabels(group_names)
ax.set_ylabel("Rate")
ax.set_ylim(0, 1.05)
ax.set_title("Mean Metrics by Scenario Group")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper right")

annotate(ax, b1, fmt="{:.3f}", ypad=0.01)
annotate(ax, b2, fmt="{:.3f}", ypad=0.01)
annotate(ax, b3, fmt="{:.3f}", ypad=0.01)


fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("baselines/grouped_mean_metrics_self_reflexion.jpg", dpi=300)
plt.close(fig)

# -----------------------------
# 2) Runtime: median+std only (overall + per group)
# -----------------------------
fig = plt.figure(figsize=(8.8, 5.2))
ax = fig.add_subplot(111)

# Plot per-group runtime median bars with std error bars
bars = ax.bar(x, rt_median_g, yerr=rt_std_g, capsize=6)
ax.set_xticks(x)
ax.set_xticklabels(group_names)
ax.set_ylabel("Seconds")
ax.set_title("Runtime by Group (Median ± Std)")
ax.grid(axis="y", alpha=0.3)

annotate(ax, bars, fmt="{:.2f}", ypad=0.15)

fig.text(
    0.01, 0.01,
    f"Overall runtime (OK-only, n={len(runtime_s)}): median={rt_median_overall:.2f}s, std={rt_std_overall:.2f}s",
    ha="left", va="bottom", fontsize=10
)

fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig("baselines/grouped_runtime_median_std_self_reflexion.png")
plt.close(fig)

# -----------------------------
# Print numbers (for paper table)
# -----------------------------
print("=== Mean metrics by group ===")
for i, g in enumerate(group_names):
    print(f"{g:8s}  mean_TPR={tpr_mean[i]:.4f}  mean_FNR={fnr_mean[i]:.4f}  mean_Acc(proxy)={acc_mean[i]:.4f}")

print("\n=== Runtime (median, std) ===")
for i, g in enumerate(group_names):
    print(f"{g:8s}  median={rt_median_g[i]:.4f}s  std={rt_std_g[i]:.4f}s")
print(f"\nOverall (OK-only) median={rt_median_overall:.4f}s  std={rt_std_overall:.4f}s")

print("\nSaved: grouped_mean_metrics.png, grouped_runtime_median_std.png")
