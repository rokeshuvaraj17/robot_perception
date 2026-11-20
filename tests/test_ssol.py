import itertools

import numpy as np
import pytest

from twinwings import SSOL, EpisodeSpec, LidarConfig, TwinWings, doorway_occluder, expected_ranges, run_episode
from twinwings.episode import DEFAULT_START
from twinwings.likelihoods import sector_pushforward
from twinwings.sensor import true_ranges


def _tiny():
    """3 beams, 5 range bins + no-return: |A|^3 = 216 scans, small enough to enumerate."""
    return SSOL(min_range=0.0, max_range=1.0, bin_w=0.2, sigma_hit=0.15, max_sector_deg=240, grid=12)


def _symbols(m):
    return list(m.min_range + (np.arange(m.n_bins) + 0.5) * m.bin_w) + [np.nan]


@pytest.mark.parametrize("r", [[0.35, 0.7, 0.55], [0.1, np.inf, 0.95], [np.inf, np.inf, np.inf]])
def test_ssol_sums_to_one_over_all_scans(r):
    m = _tiny()
    r_exp = np.array([r], dtype=float)
    scans = np.array(list(itertools.product(_symbols(m), repeat=3)))
    total = sum(np.exp(m.log_likelihood(z, r_exp))[0] for z in scans)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_emissions_are_normalized():
    m = _tiny()
    r_exp = np.array([[0.35, 0.7, np.inf]])
    for k in range(3):
        f = sum(np.exp(m.log_emissions(np.full(3, z), r_exp)[0][0, k]) for z in _symbols(m))
        g = sum(np.exp(m.log_emissions(np.full(3, z), r_exp)[1][0, k]) for z in _symbols(m))
        assert f == pytest.approx(1.0, abs=1e-9) and g == pytest.approx(1.0, abs=1e-9)


def test_pushforward_prior_sums_to_one_and_masks_are_short_runs():
    st, ln, lp, lp0 = sector_pushforward(60, 360, 30, rho=0.5)
    assert np.exp(lp).sum() + np.exp(lp0) == pytest.approx(1.0)
    assert ln.max() == 5 and len(st) == 60 * 5   # 30 deg sector covers at most 5 beams 6 deg apart
    assert np.exp(lp0) > 0.5                      # sectors between beams add to the empty mask


def test_rho_one_reduces_to_independent_clean_beams():
    w = TwinWings()
    cfg = LidarConfig()
    z, _ = true_ranges(w, DEFAULT_START, None, cfg)
    r = expected_ranges(w, np.array([DEFAULT_START, DEFAULT_START + [0.1, 0, 0]]), cfg)
    m = SSOL(rho=1.0 - 1e-15)
    log_f, _ = m.log_emissions(z, r)
    np.testing.assert_allclose(m.log_likelihood(z, r), log_f.sum(axis=1), atol=1e-6)


def test_occluded_run_is_paid_once_not_per_beam():
    """Box in doorway: summing over sectors beats explaining every beam as clean."""
    w = TwinWings(asym=0.30)
    cfg = LidarConfig()
    z, occluded = true_ranges(w, DEFAULT_START, doorway_occluder(w, "box", 0).shapes(0.0), cfg)
    r = expected_ranges(w, DEFAULT_START, cfg)
    m = SSOL()
    log_f, _ = m.log_emissions(z, r)
    assert occluded.sum() >= 2
    assert m.log_likelihood(z, r)[0] > log_f[0].sum() + 1.0


def test_max_combine_is_within_logM_of_sum():
    w = TwinWings()
    cfg = LidarConfig()
    z, _ = true_ranges(w, DEFAULT_START, doorway_occluder(w, "box", 0).shapes(0.0), cfg)
    r = expected_ranges(w, DEFAULT_START, cfg)
    s, mx = SSOL().log_likelihood(z, r)[0], SSOL(combine="max").log_likelihood(z, r)[0]
    assert s - np.log(301) - 1e-9 <= mx <= s + 1e-9


def test_ssol_runtime_budget():
    r = run_episode(EpisodeSpec(seed=3, duration=10), SSOL())
    assert np.percentile(r.update_ms, 95) < 100.0


def test_ssol_abstains_on_identical_wings():
    r = run_episode(EpisodeSpec(asym=0.0, occluder="box", seed=3, duration=15), SSOL())
    assert r.outcome() == "abstain"
