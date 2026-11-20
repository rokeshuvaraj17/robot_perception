import numpy as np
import pytest

from twinwings import (BeamMixture, EpisodeSpec, LidarConfig, MCL, TwinWings,
                       commit_decision, doorway_occluder, expected_ranges, run_episode)
from twinwings.episode import DEFAULT_START
from twinwings.geometry import raycast
from twinwings.sensor import true_ranges

CFG = LidarConfig()


# ------------------------------------------------------------------ geometry
def test_raycast_segment_circle_and_miss():
    seg = np.array([[2.0, -1.0, 2.0, 1.0]])
    cir = np.array([[0.0, 3.0, 0.5]])
    r = raycast([[0, 0], [0, 0], [0, 0]], [0.0, np.pi / 2, np.pi], seg, cir)
    assert r[0] == pytest.approx(2.0)
    assert r[1] == pytest.approx(2.5)
    assert np.isinf(r[2])


# --------------------------------------------------------------------- world
def test_wings_are_exact_aliases_when_asym_is_zero():
    w = TwinWings(asym=0.0)
    a, b = DEFAULT_START, w.alias(DEFAULT_START, 1)
    np.testing.assert_allclose(expected_ranges(w, a, CFG), expected_ranges(w, b, CFG))


def test_doorway_is_the_only_difference():
    w = TwinWings(asym=0.30)
    ra = expected_ranges(w, DEFAULT_START, CFG)[0]
    rb = expected_ranges(w, w.alias(DEFAULT_START, 1), CFG)[0]
    differ = np.flatnonzero(np.abs(ra - rb) > 1e-9)
    assert len(differ) > 0
    assert np.allclose(ra[differ] - rb[differ], 0.30 / np.cos(CFG.bearings()[differ]), atol=0.02)


def test_box_hides_doorway_and_looks_like_plain_wall():
    w = TwinWings(asym=0.30)
    occ = doorway_occluder(w, "box", 0)
    r, occluded = true_ranges(w, DEFAULT_START, occ.shapes(0.0), CFG)
    assert occluded.sum() >= 2
    r_wrong_wing = expected_ranges(w, w.alias(DEFAULT_START, 1), CFG)[0]
    assert np.all(np.abs(r[occluded] - r_wrong_wing[occluded]) < 0.06)


def test_free_space():
    w = TwinWings()
    assert w.is_free(DEFAULT_START[:2])[0]
    assert not w.is_free([2.05, 3.25])[0]   # inside the pillar
    assert not w.is_free([8.0, 2.0])[0]     # between the wings


# -------------------------------------------------------------- likelihood
def test_beam_mixture_prefers_true_pose():
    w = TwinWings()
    z, _ = true_ranges(w, DEFAULT_START, None, CFG)
    poses = np.array([DEFAULT_START, DEFAULT_START + [0.2, 0, 0], DEFAULT_START + [0, 0, 0.1]])
    ll = BeamMixture().log_likelihood(z, expected_ranges(w, poses, CFG))
    assert ll[0] > ll[1] and ll[0] > ll[2]


def test_no_return_handled():
    z = np.full(CFG.n_beams, np.nan)
    ll = BeamMixture().log_likelihood(z, np.full((2, CFG.n_beams), np.inf))
    assert np.all(np.isfinite(ll))


# ----------------------------------------------------------------------- MCL
def test_paired_init_is_symmetric():
    w = TwinWings()
    m = MCL(w, BeamMixture(), CFG, 500, rng=np.random.default_rng(0))
    m.init_modes(DEFAULT_START)
    a, b = m.particles[:250], m.particles[250:]
    np.testing.assert_allclose(b - a, np.tile([w.offset(1), 0, 0], (250, 1)), atol=1e-12)
    np.testing.assert_allclose(m.mode_masses(), [0.5, 0.5])


def test_commit_decision():
    t = np.arange(5.0)
    m = np.array([[.5, .5], [.6, .4], [.97, .03], [.99, .01], [1, 0]])
    assert commit_decision(m, t, 0.95) == (0, 2.0)
    assert commit_decision(m, t, 0.999) == (0, 4.0)
    assert commit_decision(m[:2], t[:2], 0.95) == (None, 60.0)


# ------------------------------------------------------------------ episodes
@pytest.mark.parametrize("occluder", [None, "box"])
def test_identical_wings_abstain(occluder):
    r = run_episode(EpisodeSpec(asym=0.0, occluder=occluder, seed=3, duration=15), BeamMixture())
    assert r.outcome() == "abstain"


def test_clean_episode_commits_correctly():
    r = run_episode(EpisodeSpec(asym=0.30, occluder=None, seed=3, duration=15), BeamMixture())
    assert r.outcome() == "correct"


def test_box_in_doorway_fools_independent_beams():
    """The failure SSOL targets: the baseline commits to the wrong wing."""
    r = run_episode(EpisodeSpec(asym=0.30, occluder="box", seed=3, duration=15), BeamMixture())
    assert r.outcome() == "wrong"


def test_runtime_budget():
    r = run_episode(EpisodeSpec(seed=3, duration=10), BeamMixture())
    assert np.percentile(r.update_ms, 95) < 100.0
