"""Monte Carlo validation of confidence-interval coverage.

The equivalence decision rule is "the CI lies inside (-Δ, Δ)", so the correctness of
pyTOST rests on its intervals having their stated coverage. These tests check that
claim two ways:

1. **Analytically** — the IID interval is compared against the closed-form Student-t
   interval it should equal exactly.
2. **By simulation** — intervals are built over many replicate datasets with a known
   true mean, and the realized coverage is compared against nominal.

All simulations use fixed seeds, so the tests are deterministic.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyTOST import ClusterTOST, HeteroskedasticTOST, IIDTOST, TemporalTOST

ALPHA = 0.05
NOMINAL = 1 - 2 * ALPHA  # the CI-in-TOST rule uses a (1 - 2*alpha) interval
MARGINS = [1.0]


def _coverage(hits: list[bool]) -> float:
    return float(np.mean(hits))


def _binomial_tolerance(p: float, n: int, z: float = 3.5) -> float:
    """Half-width of a z-SE band for a coverage estimate from n replicates."""
    return z * float(np.sqrt(p * (1.0 - p) / n))


class TestAnalyticalAgreement:
    """The IID interval must reproduce the textbook t-interval exactly."""

    def test_iid_ci_matches_closed_form_t_interval(self):
        rng = np.random.default_rng(2024)
        values = rng.normal(0.3, 2.0, size=37)
        df = pd.DataFrame({"diff": values})

        result = IIDTOST("diff").fit(df, ALPHA, MARGINS).iloc[0]

        n = len(values)
        mean = values.mean()
        se = values.std(ddof=1) / np.sqrt(n)
        tcrit = stats.t.ppf(1 - ALPHA, n - 1)

        assert result["mu_hat"] == pytest.approx(mean)
        assert result["ci_low"] == pytest.approx(mean - tcrit * se)
        assert result["ci_high"] == pytest.approx(mean + tcrit * se)

    def test_iid_interval_has_the_nominal_confidence_level(self):
        """Interval half-width must correspond to 1 - 2*alpha, not 1 - alpha."""
        rng = np.random.default_rng(7)
        values = rng.normal(0.0, 1.0, size=50)
        df = pd.DataFrame({"diff": values})

        result = IIDTOST("diff").fit(df, ALPHA, MARGINS).iloc[0]

        half_width = (result["ci_high"] - result["ci_low"]) / 2
        se = values.std(ddof=1) / np.sqrt(len(values))
        implied_t = half_width / se
        implied_level = 1 - 2 * (1 - stats.t.cdf(implied_t, len(values) - 1))

        assert implied_level == pytest.approx(NOMINAL, abs=1e-9)

    def test_width_shrinks_at_the_root_n_rate(self):
        """Doubling n four-fold should roughly halve the interval width."""
        rng = np.random.default_rng(99)
        widths = {}
        for n in (100, 400):
            values = rng.normal(0.0, 1.0, size=n)
            row = IIDTOST("diff").fit(pd.DataFrame({"diff": values}), ALPHA, MARGINS).iloc[0]
            widths[n] = row["ci_high"] - row["ci_low"]

        assert widths[100] / widths[400] == pytest.approx(2.0, rel=0.25)


@pytest.mark.slow
class TestIIDCoverage:
    REPLICATES = 600
    N = 40
    TRUE_MU = 0.25

    def test_iid_engine_attains_nominal_coverage_on_iid_data(self):
        rng = np.random.default_rng(12345)
        hits = []

        for _ in range(self.REPLICATES):
            values = rng.normal(self.TRUE_MU, 1.0, size=self.N)
            row = IIDTOST("diff").fit(pd.DataFrame({"diff": values}), ALPHA, MARGINS).iloc[0]
            hits.append(row["ci_low"] <= self.TRUE_MU <= row["ci_high"])

        coverage = _coverage(hits)
        tol = _binomial_tolerance(NOMINAL, self.REPLICATES)
        assert coverage == pytest.approx(NOMINAL, abs=tol), (
            f"IID coverage {coverage:.3f} outside {NOMINAL:.2f} +/- {tol:.3f}"
        )


def _clustered_sample(rng, n_clusters, per_cluster, true_mu, icc_sd, noise_sd):
    """Draw a random-intercept dataset: shared cluster effect plus within noise."""
    cluster_effects = rng.normal(0.0, icc_sd, size=n_clusters)
    rows = []
    for c in range(n_clusters):
        values = true_mu + cluster_effects[c] + rng.normal(0.0, noise_sd, size=per_cluster)
        rows.append(pd.DataFrame({"diff": values, "cluster_id": f"c{c}"}))
    return pd.concat(rows, ignore_index=True)


@pytest.mark.slow
class TestClusterCoverage:
    REPLICATES = 400
    N_CLUSTERS = 30
    PER_CLUSTER = 8
    TRUE_MU = 0.25
    ICC_SD = 0.8
    NOISE_SD = 1.0

    def _simulate(self, seed):
        rng = np.random.default_rng(seed)
        cluster_hits, iid_hits = [], []

        for _ in range(self.REPLICATES):
            df = _clustered_sample(
                rng,
                self.N_CLUSTERS,
                self.PER_CLUSTER,
                self.TRUE_MU,
                self.ICC_SD,
                self.NOISE_SD,
            )

            cl = ClusterTOST("diff", "cluster_id").fit(df, ALPHA, MARGINS).iloc[0]
            cluster_hits.append(cl["ci_low"] <= self.TRUE_MU <= cl["ci_high"])

            iid = IIDTOST("diff").fit(df, ALPHA, MARGINS).iloc[0]
            iid_hits.append(iid["ci_low"] <= self.TRUE_MU <= iid["ci_high"])

        return _coverage(cluster_hits), _coverage(iid_hits)

    def test_cluster_engine_attains_near_nominal_coverage(self):
        cluster_coverage, _ = self._simulate(seed=2468)

        tol = _binomial_tolerance(NOMINAL, self.REPLICATES)
        # Cluster-robust inference is slightly conservative-to-anticonservative in
        # finite samples; allow a modest extra allowance beyond pure Monte Carlo error.
        assert cluster_coverage == pytest.approx(NOMINAL, abs=tol + 0.03), (
            f"cluster coverage {cluster_coverage:.3f} far from nominal {NOMINAL:.2f}"
        )

    def test_ignoring_clustering_undercovers(self):
        """The motivating failure: IID intervals are too narrow under clustering."""
        cluster_coverage, iid_coverage = self._simulate(seed=2468)

        assert iid_coverage < NOMINAL - 0.10, (
            f"expected marked undercoverage from the IID engine, got {iid_coverage:.3f}"
        )
        assert cluster_coverage > iid_coverage + 0.05, (
            "cluster engine should substantially improve coverage over the IID engine"
        )

    def test_cluster_intervals_are_wider_than_iid_intervals(self):
        rng = np.random.default_rng(13579)
        df = _clustered_sample(
            rng, self.N_CLUSTERS, self.PER_CLUSTER, self.TRUE_MU, self.ICC_SD, self.NOISE_SD
        )

        cl = ClusterTOST("diff", "cluster_id").fit(df, ALPHA, MARGINS).iloc[0]
        iid = IIDTOST("diff").fit(df, ALPHA, MARGINS).iloc[0]

        assert (cl["ci_high"] - cl["ci_low"]) > (iid["ci_high"] - iid["ci_low"])


def _ar1_sample(rng, n, rho, true_mu, sd=1.0):
    """Stationary AR(1) paired differences with a known mean."""
    innov = rng.normal(0.0, sd, size=n)
    values = np.empty(n)
    values[0] = innov[0] / np.sqrt(1.0 - rho**2) if rho else innov[0]
    for i in range(1, n):
        values[i] = rho * values[i - 1] + innov[i]
    return pd.DataFrame({"diff": true_mu + values, "t": np.arange(n)})


@pytest.mark.slow
class TestTemporalCoverage:
    """Coverage of the Newey-West HAC engine under serial correlation.

    These tests record measured behaviour, including a known shortfall documented in
    ``docs/review_register.md`` as RR-003. They deliberately do not assert that the
    temporal engine attains its nominal level under strong serial correlation,
    because it does not.
    """

    REPLICATES = 400
    TRUE_MU = 0.25

    def _simulate(self, n, rho, seed):
        rng = np.random.default_rng(seed)
        hac_hits, iid_hits = [], []
        for _ in range(self.REPLICATES):
            df = _ar1_sample(rng, n, rho, self.TRUE_MU)
            hac = TemporalTOST("diff", "t").fit(df, ALPHA, MARGINS).iloc[0]
            hac_hits.append(hac["ci_low"] <= self.TRUE_MU <= hac["ci_high"])
            iid = IIDTOST("diff").fit(df, ALPHA, MARGINS).iloc[0]
            iid_hits.append(iid["ci_low"] <= self.TRUE_MU <= iid["ci_high"])
        return _coverage(hac_hits), _coverage(iid_hits)

    def test_temporal_engine_is_near_nominal_on_independent_data(self):
        hac_coverage, _ = self._simulate(n=400, rho=0.0, seed=777)

        tol = _binomial_tolerance(NOMINAL, self.REPLICATES)
        assert hac_coverage == pytest.approx(NOMINAL, abs=tol), (
            f"HAC coverage {hac_coverage:.3f} outside {NOMINAL:.2f} +/- {tol:.3f} at rho=0"
        )

    def test_temporal_engine_substantially_improves_on_iid_under_ar1(self):
        """The package's core claim: modelling the dependence recovers lost coverage."""
        hac_coverage, iid_coverage = self._simulate(n=120, rho=0.7, seed=777)

        assert iid_coverage < 0.60, (
            f"expected severe IID undercoverage under AR(1), got {iid_coverage:.3f}"
        )
        assert hac_coverage > iid_coverage + 0.15, (
            f"HAC {hac_coverage:.3f} should markedly beat IID {iid_coverage:.3f}"
        )

    def test_temporal_engine_attains_nominal_coverage_under_strong_ar1(self):
        """RR-003 resolved: the corrected AR GLS path holds its level under persistence.

        The previous Newey-West default covered roughly 0.72 here. The AR(1) GLS mean
        with a bias-corrected persistence estimate and an effective-sample-size t
        critical value covers close to nominal.
        """
        glsar_coverage, _ = self._simulate(n=120, rho=0.7, seed=777)

        tol = _binomial_tolerance(NOMINAL, self.REPLICATES)
        assert glsar_coverage == pytest.approx(NOMINAL, abs=tol), (
            f"coverage {glsar_coverage:.3f} outside {NOMINAL:.2f} +/- {tol:.3f} at rho=0.7"
        )

    def test_temporal_engine_holds_up_under_near_unit_root_persistence(self):
        """rho = 0.9 is where the old HAC default collapsed to about 0.52."""
        glsar_coverage, iid_coverage = self._simulate(n=120, rho=0.9, seed=777)

        assert iid_coverage < 0.40
        assert glsar_coverage > 0.84, (
            f"coverage {glsar_coverage:.3f} should stay near nominal at rho=0.9"
        )

    def test_hac_method_remains_available_and_is_anticonservative(self):
        """The nonparametric HAC estimator is still reachable, and still undercovers.

        Keeping this documented matters: HAC is asymptotically valid for general
        dependence, so users may still want it, but at validation-study sample sizes
        it does not hold its nominal level (RR-003).
        """
        rng = np.random.default_rng(777)
        hits = []
        for _ in range(self.REPLICATES):
            df = _ar1_sample(rng, 120, 0.7, self.TRUE_MU)
            row = (
                TemporalTOST("diff", "t", method="hac")
                .fit(df, ALPHA, MARGINS)
                .iloc[0]
            )
            hits.append(row["ci_low"] <= self.TRUE_MU <= row["ci_high"])

        assert _coverage(hits) < 0.82, (
            "HAC is expected to remain anticonservative; if this improved, RR-003 and "
            "the paper's validation section must be updated"
        )


@pytest.mark.slow
class TestHeteroskedasticCoverage:
    """HC3 and ordinary OLS target the same variance for an intercept-only mean.

    Documented as RR-004. For a plain mean, Var(ybar) = sum(sigma_i^2)/n^2 and the OLS
    estimate s^2/n is already consistent for it, so HC3 is a different finite-sample
    estimator rather than a correction. The robustness that matters in this package is
    the cluster-robust wild bootstrap path, not HC3 itself.
    """

    REPLICATES = 400
    N = 120
    TRUE_MU = 0.25

    def _simulate(self, seed):
        rng = np.random.default_rng(seed)
        scales = np.linspace(0.3, 3.0, self.N)
        hc3_hits, iid_hits = [], []
        for _ in range(self.REPLICATES):
            df = pd.DataFrame({"diff": self.TRUE_MU + rng.normal(0.0, 1.0, self.N) * scales})
            hc3 = HeteroskedasticTOST("diff").fit(df, ALPHA, MARGINS).iloc[0]
            hc3_hits.append(hc3["ci_low"] <= self.TRUE_MU <= hc3["ci_high"])
            iid = IIDTOST("diff").fit(df, ALPHA, MARGINS).iloc[0]
            iid_hits.append(iid["ci_low"] <= self.TRUE_MU <= iid["ci_high"])
        return _coverage(hc3_hits), _coverage(iid_hits)

    def test_hc3_attains_at_least_nominal_coverage_under_heteroskedasticity(self):
        hc3_coverage, _ = self._simulate(seed=31)

        tol = _binomial_tolerance(NOMINAL, self.REPLICATES)
        assert hc3_coverage > NOMINAL - tol, (
            f"HC3 coverage {hc3_coverage:.3f} fell below nominal {NOMINAL:.2f}"
        )

    def test_hc3_and_iid_coverage_agree_for_an_intercept_only_mean(self):
        """RR-004: heteroskedasticity alone does not bias the ordinary mean's SE."""
        hc3_coverage, iid_coverage = self._simulate(seed=31)

        assert abs(hc3_coverage - iid_coverage) < 0.05, (
            f"HC3 {hc3_coverage:.3f} and IID {iid_coverage:.3f} should agree closely "
            "for an intercept-only model; a large gap would contradict RR-004"
        )
