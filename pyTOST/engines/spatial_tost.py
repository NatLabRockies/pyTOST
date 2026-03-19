"""
engines/spatial_tost.py
=======================

Spatial TOST (Matérn GLS via REML + LR CI)

The estimand is the population mean difference μ for an intercept-only model of
a paired difference (e.g., diff = y_B - y_A). Equivalence at margin Δ is decided
via CI-in-TOST:
    equivalent(Δ) ⇔ CI_μ ⊂ (-Δ, +Δ).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from .pubgrade_spatial_tost import fit_matern_reml, lr_ci_for_mu  # local dependency


@dataclass(frozen=True)
class SpatialConfig:
    """
    Parameters controlling the Matérn REML fit.

    Parameters
    ----------
    nu_grid
        Candidate Matérn smoothness values ν considered in the REML profile search.
    per_building_nugget
        Whether to include a nugget term (τ² I) inside each building block.
    verbose_diagnostics
        If True, print diagnostic summaries that help detect covariance/SE
        pathologies (e.g., variance collapse) and compare estimands.
    """

    nu_grid: Tuple[float, ...] = (0.5, 1.5, 2.5)
    per_building_nugget: bool = True
    verbose_diagnostics: bool = False


class SpatialTOST:
    """
    Publication-grade spatial TOST using Matérn GLS (REML) + LR CI for μ.

    Parameters
    ----------
    y : str
        Response column name (paired difference), e.g., "diff".
    cluster : str
        Building/group id column name. Required because Σ is block-diagonal by building.
    x, ycoord : str
        Spatial coordinate column names.
    config : SpatialConfig
        REML fit settings.
    """

    def __init__(
        self,
        y: str,
        cluster: str,
        x: str,
        ycoord: str,
        config: SpatialConfig | None = None,
    ):
        self.y = y
        self.cluster = cluster
        self.x = x
        self.ycoord = ycoord
        self.config = config or SpatialConfig()

    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        """
        Fit the Matérn GLS model and compute CI-in-TOST decisions across margins.

        Returns
        -------
        DataFrame
            Columns: delta, mu_hat, ci_low, ci_high, equivalent, method
        """
        for col in (self.y, self.cluster, self.x, self.ycoord):
            if col not in df.columns:
                raise ValueError(
                    f"SpatialTOST requires column {col!r}. "
                    f"Available columns: {list(df.columns)}"
                )

        dfp = df.rename(
            columns={
                self.cluster: "building_id",
                self.y: "diff",
                self.x: "x",
                self.ycoord: "y",
            }
        )

        #print(f"dfp['diff'].mean(): {dfp['diff'].mean()}")
        #print("Mean of diff passed to spatial:", dfp["diff"].mean())
        #print("Min/Max of diff:", dfp["diff"].min(), dfp["diff"].max())
        #print("First 10 diff values:", dfp["diff"].head(10).to_list())

        theta = fit_matern_reml(
            df=dfp,
            building_col="building_id",
            x_col="x",
            y_col="y",
            diff_col="diff",
            nu_grid=self.config.nu_grid,
            per_building_nugget=self.config.per_building_nugget,
        )
        #from pprint import pprint
        #pprint(f"theta: {theta}")
        mu_hat = float(theta["mu_hat"])

        if self.config.verbose_diagnostics:
            # (1) Simple means
            mean_all = float(dfp["diff"].mean())
            mean_building_eq = float(dfp.groupby("building_id")["diff"].mean().mean())
            print("[Spatial diagnostics]")
            print(f"  diff.mean()={mean_all:.6g}")
            print(f"  building-equal-weight mean={mean_building_eq:.6g}")

            # (2) Fitted covariance + conditional var(mu_hat)
            sigma2 = float(theta.get("sigma2", float("nan")))
            tau2 = float(theta.get("tau2", float("nan")))
            rho = float(theta.get("rho", float("nan")))
            var_mu = float(theta.get("var_mu", float("nan")))
            nu = float(theta.get("nu", float("nan")))
            print(f"  fitted mu_hat={mu_hat:.6g}")
            print(f"  fitted theta: sigma2={sigma2:.6g}, tau2={tau2:.6g}, rho={rho:.6g}, nu={nu:.6g}")
            print(f"  fitted var_mu (conditional)={var_mu:.6g}, se={float(var_mu**0.5):.6g}")


        ci_low, ci_high = lr_ci_for_mu(
            df=dfp,
            building_col="building_id",
            x_col="x",
            y_col="y",
            diff_col="diff",
            theta=theta,
            alpha=alpha,
        )

        if self.config.verbose_diagnostics:
            print(f"  LR CI: [{float(ci_low):.6g}, {float(ci_high):.6g}]")
            print("  ----")

        rows = []
        for d in margins:
            d = float(d)
            rows.append(
                dict(
                    delta=d,
                    mu_hat=mu_hat,
                    ci_low=float(ci_low),
                    ci_high=float(ci_high),
                    equivalent=(ci_low > -d and ci_high < d),
                    method="Matérn GLS (REML) + LR CI",
                )
            )
        return pd.DataFrame(rows)
