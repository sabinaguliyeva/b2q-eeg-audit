"""Generate figures/eeg_audit.png from results/eeg_audit_results.json."""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
res = json.loads((HERE / "results" / "eeg_audit_results.json").read_text())

INK = "#1a1a1a"
GREY = "#b8b8b8"
ACCENT = "#c2410c"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), width_ratios=[1.45, 1])
fig.patch.set_facecolor("white")

# --- left: per-subject keystroke counts ------------------------------------
per = res["per_subject"]
vals = sorted(per.values(), reverse=True)
ax1.bar(range(len(vals)), vals, color=GREY, edgecolor="none", width=0.78)
ax1.bar([len(vals) - 1], [vals[-1]], color=ACCENT, edgecolor="none", width=0.78)

ax1.set_title("Keystrokes per subject", loc="left", fontsize=11, color=INK, pad=10)
ax1.set_xlabel("subject (sorted)", fontsize=9, color=INK)
ax1.set_xticks([])
ax1.set_ylim(0, max(vals) * 1.18)
ax1.set_yticks([0, 2500, 5000, 7500, 10000])
ax1.set_yticklabels(["0", "2.5k", "5k", "7.5k", "10k"])
ax1.annotate(
    f"{vals[-1]:,}\n(4.8x less than the largest)",
    xy=(len(vals) - 1, vals[-1]),
    xytext=(len(vals) - 4.4, max(vals) * 0.62),
    fontsize=8.5, color=ACCENT, ha="left",
    arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.9),
)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
ax1.spines["left"].set_color(GREY)
ax1.spines["bottom"].set_color(GREY)
ax1.tick_params(colors=INK, labelsize=8.5)

# --- right: subject overlap across splits ----------------------------------
splits = res["splits"]
names = ["train", "val", "test"]
counts = [splits[n]["keystrokes"] for n in names]
subs = [splits[n]["subjects"] for n in names]

bars = ax2.barh(range(3), counts, color=[INK, GREY, GREY], height=0.5)
ax2.set_yticks(range(3))
ax2.set_yticklabels(names, fontsize=9.5, color=INK)
ax2.invert_yaxis()
ax2.set_title("Split sizes — and who is in them", loc="left", fontsize=11, color=INK, pad=10)
ax2.set_xlabel("keystrokes", fontsize=9, color=INK)
ax2.set_xlim(0, max(counts) * 1.52)
ax2.set_xticks([0, 50000, 100000])
ax2.set_xticklabels(["0", "50k", "100k"])

for i, (c, s) in enumerate(zip(counts, subs)):
    ax2.text(c + max(counts) * 0.03, i, f"{c:,}   ·   {s}/20 subjects",
             va="center", fontsize=8.5, color=INK)

for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.spines["left"].set_color(GREY)
ax2.spines["bottom"].set_color(GREY)
ax2.tick_params(colors=INK, labelsize=8.5)

ax2.text(
    0, 3.0,
    "every subject appears in all three splits\n"
    "→ the published EEG metric is within-subject",
    fontsize=9, color=ACCENT, va="top",
)

fig.tight_layout(rect=[0, 0.06, 1, 1])
(HERE / "figures").mkdir(exist_ok=True)
out = HERE / "figures" / "eeg_audit.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
