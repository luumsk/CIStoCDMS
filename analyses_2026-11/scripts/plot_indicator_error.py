"""Figure for the indicator-error simulation (reads ../results/indicator_error_simulation.csv).

Two panels sharing the x-axis (share of positive periventricular indicators missed, r):
  left  - mean fold AUC; right - share of identified converters no longer identified.
Line = mean over the six models, band = range over the six models. Points = the
measured small-lesion miss rates of DiceCE and CATMIL on MSLesSeg and 3D-MR-MS (Ch3),
placed at the mean over the six models. Palette: default categorical slots 1-2
(validated: CVD dE 24.7, normal dE 33.6, contrast >= 3:1).

Usage: python plot_indicator_error.py
Outputs: ../results/indicator_error_simulation.{pdf,png}
"""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
INK, MUTED, GRID, BAND = "#0b0b0b", "#5f5f5c", "#e4e3df", "#d6d5d0"
COL = {"DiceCE": "#eb6834", "CATMIL": "#2a78d6"}

df = pd.read_csv(os.path.join(RES, "indicator_error_simulation.csv"))
sweep = df[df.group == "sweep"]
meas = df[df.group == "measured"].copy()
meas["dataset"] = meas.setting.str.split("/").str[0]
meas["loss"] = meas.setting.str.split("/").str[1]
meas = meas[meas.loss.isin(COL)]

panels = [("auc_mean", "AUC (mean over 5 outer folds)", 1.0),
          ("share_converters_newly_missed", "Identified converters lost (%)", 100.0)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
for ax, (col, ylab, k) in zip(axes, panels):
    g = sweep.groupby("r_miss")[col]
    r = g.mean().index.to_numpy()
    ax.fill_between(r, g.min() * k, g.max() * k, color=BAND, lw=0, label="range over 6 models")
    ax.plot(r, g.mean() * k, color=MUTED, lw=2, label="mean over 6 models")
    for (ds, loss), m in meas.groupby(["dataset", "loss"]):
        x, y = m.r_miss.iloc[0], m[col].mean() * k
        ax.plot(x, y, "o", ms=8, color=COL[loss], mec="#fcfcfb", mew=2, zorder=3)
    # direct labels for the four measured points (positions chosen to avoid the line)
    for (ds, loss), m in meas.groupby(["dataset", "loss"]):
        x, y = m.r_miss.iloc[0], m[col].mean() * k
        up = (loss == "DiceCE") == (col == "auc_mean")
        dy = -14 if up else 9
        if ds == "3D-MR-MS":
            dy = -14 if loss == "DiceCE" else 9
        ax.annotate(f"{loss}\n{ds}", (x, y), textcoords="offset points", xytext=(0, dy), ha="center",
                    va="top" if dy < 0 else "bottom", fontsize=7, color=INK, linespacing=0.9)
    ax.set_xlabel("Positive periventricular indicators missed, r")
    ax.set_ylabel(ylab)
    ax.set_xlim(-0.03, 1.03)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=7, loc="lower left")
fig.savefig(os.path.join(RES, "indicator_error_simulation.pdf"))
fig.savefig(os.path.join(RES, "indicator_error_simulation.png"), dpi=200)
print("saved")
