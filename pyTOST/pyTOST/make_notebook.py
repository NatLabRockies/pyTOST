import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}
nb.metadata["language_info"] = {"name": "python"}

nb.cells = [
    new_markdown_cell(
        "# SAV equivalence plot\n\n"
        "Clean notebook with corrected ordering:\n"
        "**January → December → Summer → Winter → Annual → Global**."
    ),

    new_code_cell(
"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GLOBAL_PATH = "spatiotemporal_results_global_20260217.csv"
SPATIAL_PATH = "spatial_results_by_time_20260217.csv"

global_df = pd.read_csv(GLOBAL_PATH).sort_values("delta").reset_index(drop=True)
spatial_df = pd.read_csv(SPATIAL_PATH)

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
SEASON_ORDER = ["Summer (May - Oct)", "Winter (Nov-April)"]

def normalize_time_label(t):
    mapping = {
        "january": "January",
        "february": "February",
        "march": "March",
        "april": "April",
        "may": "May",
        "june": "June",
        "july": "July",
        "august": "August",
        "september": "September",
        "october": "October",
        "november": "November",
        "december": "December",
        "annual": "Annual",
        "summer (may - oct)": "Summer (May - Oct)",
        "winter (nov-april)": "Winter (Nov-April)",
    }
    key = str(t).strip()
    return mapping.get(key.lower(), key)

def time_sort_key(t):
    t = normalize_time_label(t)
    if t in MONTH_ORDER:
        return (0, MONTH_ORDER.index(t))
    if t in SEASON_ORDER:
        return (1, SEASON_ORDER.index(t))
    if t == "Annual":
        return (2, 0)
    if t == "Global":
        return (3, 0)
    return (4, str(t))

NREL_BLUE = "#0079C1"
LIGHT_GRAY = "#777777"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})"""
    ),

    new_code_cell(
"""rows = []

for _, r in global_df.iterrows():
    label = "Global" if len(global_df) == 1 else f"Global (Δ={r['delta']:g})"
    rows.append({
        "label": label,
        "mu_hat": r["mu_hat"],
        "ci_low": r["ci_low"],
        "ci_high": r["ci_high"],
        "boot_ci_low": r["boot_ci_low"],
        "boot_ci_high": r["boot_ci_high"],
        "margin": float(r["delta"]),
    })

spatial_df = spatial_df.copy()
spatial_df["time"] = spatial_df["time"].apply(normalize_time_label)
spatial_df["__k__"] = spatial_df["time"].apply(time_sort_key)
spatial_df = spatial_df.sort_values("__k__").drop(columns="__k__").reset_index(drop=True)

for _, r in spatial_df.iterrows():
    rows.append({
        "label": str(r["time"]),
        "mu_hat": r["mu_hat"],
        "ci_low": r["ci_low"],
        "ci_high": r["ci_high"],
        "boot_ci_low": r["boot_ci_low"],
        "boot_ci_high": r["boot_ci_high"],
        "margin": 1.0,
    })

df = pd.DataFrame(rows)
df["label"] = df["label"].apply(normalize_time_label)
df["__k__"] = df["label"].apply(time_sort_key)
df = df.sort_values("__k__").drop(columns="__k__").reset_index(drop=True)

df[["label", "mu_hat", "ci_low", "ci_high", "boot_ci_low", "boot_ci_high"]]"""
    ),

    new_code_cell(
"""fig_h = max(6.2, 0.38 * len(df) + 2.3)
fig, ax = plt.subplots(figsize=(8.8, fig_h))

y = np.arange(len(df))
labels = df["label"].to_numpy()

mu = df["mu_hat"].to_numpy()
lo = df["ci_low"].to_numpy()
hi = df["ci_high"].to_numpy()
blo = df["boot_ci_low"].to_numpy()
bhi = df["boot_ci_high"].to_numpy()
marg = df["margin"].to_numpy()

primary_lw = 3.0
boot_lw = 1.2
offset = 0.28
tick_halfheight = 0.14
tick_lw = 2.0

for yi, l, h, m, bl, bh in zip(y, lo, hi, mu, blo, bhi):
    ax.plot([l, h], [yi, yi], linewidth=primary_lw, color=NREL_BLUE, solid_capstyle="butt")
    ax.plot([bl, bh], [yi + offset, yi + offset], linewidth=boot_lw, color=NREL_BLUE, solid_capstyle="butt")
    ax.plot([m, m], [yi - tick_halfheight, yi + tick_halfheight], linewidth=tick_lw, color=NREL_BLUE, solid_capstyle="butt")

ax.axvline(0.0, linewidth=1.1, color=LIGHT_GRAY)
max_margin = float(np.max(marg))
ax.axvline(+max_margin, linestyle="--", linewidth=1.1, color=LIGHT_GRAY)
ax.axvline(-max_margin, linestyle="--", linewidth=1.1, color=LIGHT_GRAY)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()

ax.set_xlabel("Mean SAV difference (Monalee − Exactus)")
ax.set_title("Equivalence Results (Global + Spatial Time Slices)")

xmin = np.min([lo.min(), blo.min(), -max_margin])
xmax = np.max([hi.max(), bhi.max(), +max_margin])
pad = 0.08 * (xmax - xmin + 1e-12)
ax.set_xlim(xmin - pad, xmax + pad)

mean_marker_size = 10
legend_handles = [
    Line2D([0], [0], color=NREL_BLUE, linewidth=primary_lw, solid_capstyle="butt", label="Primary CI"),
    Line2D([0], [0], color=NREL_BLUE, linewidth=boot_lw, solid_capstyle="butt", label="Bootstrap CI"),
    Line2D([0], [0], color=NREL_BLUE, linestyle="None", marker="|",
           markersize=mean_marker_size, markeredgewidth=tick_lw, label=r"$\\hat{\\mu}$"),
    Line2D([0], [0], color=LIGHT_GRAY, linewidth=1.1, label="0 reference"),
    Line2D([0], [0], color=LIGHT_GRAY, linewidth=1.1, linestyle="--", label="Margin guide"),
]

fig.legend(
    legend_handles,
    [h.get_label() for h in legend_handles],
    loc="lower center",
    ncol=len(legend_handles),
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
    handlelength=3.0,
    columnspacing=1.8,
)

fig.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()"""
    ),
]

with open("sav_clean.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote sav_clean.ipynb")
