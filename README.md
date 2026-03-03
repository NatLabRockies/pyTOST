# pyTOST

pyTOST is a dependency-aware equivalence testing toolkit built around the **Two One-Sided Tests (TOST)** procedure. It is designed for practical “are these two measurement processes effectively the same?” questions when data may be **independent (IID)**, **clustered** (e.g., building IDs), **spatially correlated** (coordinates), **temporally correlated** (time series), or **spatio-temporal**.

The core idea is unchanged from classical TOST: for each equivalence margin Δ, estimate the mean difference μ̂, compute a valid confidence interval for μ, and declare *equivalence* if the interval lies entirely within (−Δ, +Δ). What changes is *how the confidence interval is computed* when observations are dependent.

---

## Installation

From a local clone or source directory:

```bash
pip install -e .
```

Recommended dependencies:
- numpy
- pandas
- scipy
- statsmodels
- matplotlib

Optional (for PDF reporting):
- pylatex
- a LaTeX distribution (MacTeX / TinyTeX / TeX Live / MiKTeX)

---

## Quick Start

### 1. Prepare a CSV
At minimum, your CSV must contain a column representing the **difference** to be tested (e.g., Artemis minus Exactus).

Optional columns allow pyTOST to detect and correct for dependence.

### 2. Run the pipeline
```bash
python run_tost.py \
  --input data/my_diffs.csv \
  --diff-col diff \
  --margins 1 3 5 \
  --alpha 0.05 \
  --outdir outputs/example_run
```

---

## Running `run_tost.py`

### Command-line Interface

```bash
python run_tost.py \
  --input <data.csv> \
  --diff-col <column> \
  --margins <Δ1> <Δ2> ... \
  --alpha <alpha> \
  --outdir <output_dir> \
  [--cluster-col <column>] \
  [--x-col <column> --y-col <column>] \
  [--time-col <column>] \
  [--policy auto|force_iid|force_cluster|force_spatial|force_temporal|force_spatiotemporal] \
  [--bootstrap-B <int>] \
  [--seed <int>] \
  [--report-deltas <Δ1> <Δ2> ...]
```

---

## Input CSV Format

### Required
- **diff** (or the name passed via `--diff-col`)  
  Numeric difference per observation.

Example:
```csv
diff
0.8
-0.3
1.1
```

### Optional (recommended)

| Column | Purpose |
|------|--------|
| `building_id` | Cluster identifier (use with `--cluster-col`) |
| `x`, `y` | Spatial coordinates (use with `--x-col`, `--y-col`) |
| `time` | Time index or datetime (use with `--time-col`) |

Full example:
```csv
building_id,x,y,time,diff
B01,12.3,98.1,2025-01-01T10:00:00,0.7
B01,12.6,97.9,2025-01-01T10:05:00,0.4
B02,210.2,33.5,2025-01-01T10:00:00,1.2
```

---

## Output Structure

The output directory contains:

```
outdir/
├── summaries/
│   ├── iid.csv
│   ├── cluster.csv
│   ├── spatial.csv
│   └── bootstrap.csv
├── diagnostics/
│   └── diagnostics.json
├── figures/
│   ├── ci_comparison.png
│   ├── residuals.png
│   └── acf.png
└── report/
    └── one_page_summary.pdf
```

Each summary table includes:
- equivalence margin Δ
- μ̂ (mean difference)
- CI lower and upper bounds
- equivalence decision (True / False)

---

## How pyTOST Chooses the Method (Default: `policy=auto`)

pyTOST evaluates diagnostics and selects the most appropriate TOST engine:

1. **IID TOST** – no detected dependence  
2. **Cluster-aware TOST** – intra-cluster correlation detected  
3. **Spatial TOST** – spatial autocorrelation detected  
4. **Temporal TOST** – serial correlation detected  
5. **Spatio-temporal TOST** – both spatial and temporal dependence  

Sensitivity analyses (heteroskedastic-robust and bootstrap) are always run for validation.

---

## Using pyTOST as a Library

```python
import pandas as pd
from pyTOST.workflow import run_dependency_aware_tost
from pyTOST.report_onepager import one_page_report

df = pd.read_csv("data/my_diffs.csv")

res = run_dependency_aware_tost(
    df=df,
    diff_col="diff",
    margins=[1, 3, 5],
    alpha=0.05,
    cluster_col="building_id",
    x_col="x",
    y_col="y",
    time_col="time",
    policy="auto",
    bootstrap_B=500,
    seed=42,
)

one_page_report(res, "outputs/summary.pdf")
```

---

## Interpreting Results

- If IID TOST passes but dependency-aware TOST fails, trust the dependency-aware result.
- If multiple dependency-aware methods disagree, prefer the most conservative (widest CI).
- Bootstrap intervals act as a sanity check on analytic confidence intervals.

---

## Troubleshooting

- **PDF report fails:** Ensure LaTeX (`pdflatex`) is installed and on PATH.
- **Spatial model errors:** Try `policy=force_cluster` as a conservative fallback.
- **Time parsing issues:** Convert time to ISO 8601 or numeric index.

---

## Citation

If you use pyTOST in published work, please cite the accompanying technical report and foundational references for TOST, robust inference, and spatial/temporal modeling. See `ref.bib` for canonical citations.

