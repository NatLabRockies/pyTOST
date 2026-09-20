# Synthetic Dataset for Dependence-Aware Paired-Equivalence Testing

This archive contains a synthetic, fully reproducible spatiotemporal panel that
accompanies a manuscript on dependence-aware paired-equivalence (two one-sided
tests, TOST) inference and its companion software package. It contains no
proprietary or personal data.

The dataset reproduces the key structural features exercised by the workflow:
paired observations, spatial clustering into buildings, temporal ordering, and a
balanced spatiotemporal panel, together with executable analysis materials.

## Dataset summary

| Quantity | Value |
|---|---|
| Buildings (G, clustering unit) | 9 |
| Spatial locations (M) | 25 |
| Time steps (T) | 98 |
| Total paired observations (N = M x T) | 2,450 |
| Panel completeness | complete, balanced 25 x 98 panel |
| Generation seed | 15211 |
| Equivalence margin used in the demo | Delta = 1 |
| Significance level | alpha = 0.05 |

## Data dictionary (`synthetic_stress_test_dataset.csv`, 2,450 rows)

| Column | Type | Description |
|---|---|---|
| `sample_id` | int | Row index within the panel (0 ... 2449), location-major ordering. |
| `x` | float | Projected planar east coordinate of the location. |
| `y` | float | Projected planar north coordinate of the location. |
| `time` | int | Time step, 0 ... 97 (temporal ordering within each location). |
| `A` | float | Arm A (reference) value for the paired observation. |
| `B` | float | Arm B (comparison) value for the paired observation. |
| `diff` | float | Paired difference, `B - A` (the response analyzed by TOST). |
| `cluster_id` | int | Building identifier, 9 levels (0 ... 8). Constant within a location; a genuine spatial grouping (see "Building assignment"). Use this as the clustering unit for cluster-robust and spatial engines. |
| `loc_id` | int | Spatial-location identifier, 25 levels (0 ... 24). One value per unique `(x, y)` pair, constant across the 98 time steps of a location. |

### Building assignment

The 25 spatial locations are partitioned into 9 spatially compact buildings by a
deterministic Sort-Tile-Recursive (STR) tiling of the location coordinates. Each
building contains 2-3 whole locations, and `cluster_id` is constant within a
location (every time step of a location shares the same building), so it is a
meaningful spatial cluster rather than an arbitrary label. The mean within-building
pairwise distance is roughly 2.6x smaller than the overall mean pairwise distance.

## Structure the dataset exhibits

- Paired observations: `A`, `B`, and their difference `diff = B - A`.
- Spatial clustering: locations grouped into buildings (`cluster_id`) and spatially
  correlated coordinates `(x, y)` from a Matern process.
- Temporal ordering: an AR(1)-correlated series of length 98 at each location (`time`).
- Balanced panel: every location has exactly 98 time steps.

## Contents

| File | Description |
|---|---|
| `synthetic_stress_test_dataset.csv` | The frozen synthetic panel (see data dictionary). |
| `synthetic_stress_test_meta.json` | Generator parameters, RNG seed, building-assignment note, and package versions (reproducibility record). |
| `oneshot_candidate.json` | Frozen generator parameters and seed consumed by the generation script. |
| `make_synthetic_dataset.py` | Regenerates the dataset from the recorded seed and parameters using the companion `pyTOST` package. |
| `demo_analysis.py` | Loads the dataset and runs the dependence-aware TOST engines, printing intervals and equivalence decisions. |
| `LICENSE` | Data license (CC BY 4.0). |
| `.zenodo.json` | Deposit metadata. |

## Reproducing the dataset

Install the companion `pyTOST` package (from its own archive/repository), then:

```bash
pip install pyTOST
python make_synthetic_dataset.py
```

This regenerates `synthetic_stress_test_dataset.csv` and
`synthetic_stress_test_meta.json` from seed 15211 and the parameters in
`oneshot_candidate.json`, using
`pyTOST.data_gen.synthetic_tost_data.generate_spatiotemporal`. The paired arms
and coordinates reproduce to machine precision; the `cluster_id` building labels
are assigned deterministically by the STR partition in `make_synthetic_dataset.py`.

## Example analysis

```bash
python demo_analysis.py                 # fast engines (seconds)
python demo_analysis.py --with-spatial  # also fit the Matern spatial engine (minutes)
```

Representative output at Delta = 1, alpha = 0.05 (all engines share the same
point estimate; ignoring dependence yields the narrowest interval):

```
                   engine   mu_hat   ci_low  ci_high    width  equivalent
                      IID 0.928126 0.900804 0.955448 0.054644        True
Cluster (building-robust) 0.928126 0.866391 0.989862 0.123471        True
           Temporal (HAC) 0.928126 0.892289 0.963964 0.071675        True
```

The building-robust cluster interval is roughly 2.3x wider than the naive IID
interval on this panel, illustrating why dependence-aware intervals are needed
when observations are grouped and serially correlated.

## License

The dataset and accompanying materials are released under the Creative Commons
Attribution 4.0 International license (CC BY 4.0); see `LICENSE`.

## Notes for double-blind review

Author identity, the companion software archive URL, and deposit DOIs are omitted
from this review copy and are recorded as placeholders in `.zenodo.json`. They are
to be completed for the final public deposit.
