"""clubSandwich CR2 Satterthwaite df regression tests for cr2_mean.

Reference: clubSandwich R package 0.7.0 (Pustejovsky & Tipton 2018).
Verified numerically:
  - balanced G clusters -> df = G - 1
  - case-study cluster sizes [168,156,180,96,156,204,240] (N=1200) ->
    df = 5.6235377772
"""
import numpy as np
import pytest

from pyTOST.fewcluster import cr2_mean


def _direct_se_cr2(y, groups):
    """Independent recomputation of the CR2 SE from the closed form."""
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    N = len(y)
    mu = float(y.mean())
    e = y - mu
    uniq = np.array(sorted(set(groups.tolist())))
    s_raw = np.array([e[groups == g].sum() for g in uniq])
    n_g = np.array([(groups == g).sum() for g in uniq])
    adj = 1.0 / np.sqrt(1.0 - n_g / N)
    v = ((s_raw * adj) ** 2).sum() / N ** 2
    return float(np.sqrt(v))


def test_balanced_df_equals_G_minus_1():
    G = 7
    per = 100
    N = G * per
    rng = np.random.default_rng(0)
    y = rng.normal(size=N)
    g = np.repeat(np.arange(G), per)
    r = cr2_mean(y, g)
    assert r["G"] == G
    assert abs(r["df_bm"] - (G - 1)) < 1e-10, r["df_bm"]


def test_case_study_sizes_match_clubSandwich():
    sizes = [168, 156, 180, 96, 156, 204, 240]
    g = np.concatenate([[i] * s for i, s in enumerate(sizes)])
    rng = np.random.default_rng(1)
    y = rng.normal(size=sum(sizes))
    r = cr2_mean(y, g)
    # clubSandwich 0.7.0 reference value
    assert abs(r["df_bm"] - 5.6235377772) < 1e-6, r["df_bm"]


def test_se_cr2_unchanged_by_df_fix():
    # SE must equal a direct sqrt(sum((s_raw*adj)^2))/N recomputation
    # for both balanced and unbalanced designs.
    rng = np.random.default_rng(2)
    for sizes in ([100] * 7, [168, 156, 180, 96, 156, 204, 240], [50, 30, 40, 25]):
        y = rng.normal(size=sum(sizes))
        g = np.concatenate([[i] * s for i, s in enumerate(sizes)])
        r = cr2_mean(y, g)
        se_direct = _direct_se_cr2(y, g)
        assert r["se_cr2"] == pytest.approx(se_direct, rel=0, abs=1e-14)


def test_balanced_various_G():
    # Sanity: df = G-1 holds for a range of balanced designs.
    rng = np.random.default_rng(3)
    for G in (3, 5, 10, 20):
        per = 40
        y = rng.normal(size=G * per)
        g = np.repeat(np.arange(G), per)
        r = cr2_mean(y, g)
        assert abs(r["df_bm"] - (G - 1)) < 1e-10, (G, r["df_bm"])
