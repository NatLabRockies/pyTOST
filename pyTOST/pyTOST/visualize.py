"""
visualize.py
============
Minimal visualizations (matplotlib only).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_ci_comparison(summary_dict: dict):
    """
    Plot horizontal CIs for μ across methods per Δ.

    summary_dict: {method_name: DataFrame with columns [delta, mu_hat, ci_low, ci_high, equivalent]}
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    methods = list(summary_dict.keys())
    offsets = np.linspace(-0.2, 0.2, len(methods)) if methods else []
    for (k, df), off in zip(summary_dict.items(), offsets):
        for _, r in df.iterrows():
            ax.plot([r["ci_low"], r["ci_high"]], [r["delta"]+off]*2, marker='|')
    ax.axvline(0.0, linestyle='--', linewidth=1)
    ax.set_xlabel("μ (units)")
    ax.set_ylabel("Δ (equivalence margin)")
    ax.set_title("Confidence Intervals by Method")
    return fig, ax

