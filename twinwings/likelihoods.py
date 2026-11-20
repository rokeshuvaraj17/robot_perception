"""Scan likelihoods. Every comparator plugs into MCL through one method:

    log_likelihood(z, r_exp) -> (N,) log L(z_B | x_i)

z:     (B,)   measured ranges, np.nan = no-return
r_exp: (N, B) map ray-cast ranges for each particle, np.inf = nothing in map

Only this block changes between arms (Fig. 3 of the draft); motion model,
beams, particles, resampling and the commit rule stay identical.

Implemented so far:
  BeamMixture  - comparator 1, independent beam mixture (Thrun et al.)
  SSOL         - the proposed shared-sector likelihood (B5); combine="max"
                 gives the maximum-likelihood-sector comparator
To add next:
  independent occlusion classes (comparator 2), two-state chain
  (comparator 3), AMCL-style likelihood field with beam skipping (comparator 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

_LOG_FLOOR = 1e-300


class ScanLikelihood(Protocol):
    name: str

    def log_likelihood(self, z: np.ndarray, r_exp: np.ndarray) -> np.ndarray: ...


@dataclass
class BeamMixture:
    """Independent beam model: product over beams of a hit/short/max/random mixture.

    An occluder that blocks k beams is paid for k times (once per beam), which
    is the behaviour SSOL is designed to change.
    """
    w_hit: float = 0.80
    w_short: float = 0.10
    w_max: float = 0.05
    w_rand: float = 0.05
    sigma_hit: float = 0.05
    lam_short: float = 1.0
    max_range: float = 12.0
    name: str = "beam_mixture"

    def log_likelihood(self, z: np.ndarray, r_exp: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)[None, :]
        valid = ~np.isnan(z)
        zz = np.where(valid, z, 0.0)
        in_range = r_exp < self.max_range
        r = np.minimum(r_exp, self.max_range)

        p_hit = np.exp(-0.5 * ((zz - r) / self.sigma_hit) ** 2) / (np.sqrt(2 * np.pi) * self.sigma_hit)
        p_hit = np.where(in_range, p_hit, 0.0)
        norm_short = 1.0 - np.exp(-self.lam_short * r)
        p_short = np.where(zz < r, self.lam_short * np.exp(-self.lam_short * zz) / np.maximum(norm_short, 1e-12), 0.0)
        p_rand = 1.0 / self.max_range

        p_valid = self.w_hit * p_hit + self.w_short * p_short + self.w_rand * p_rand
        p_none = self.w_max + self.w_hit * (~in_range)
        p = np.where(valid, p_valid, p_none)
        return np.log(np.maximum(p, _LOG_FLOOR)).sum(axis=1)


# --------------------------------------------------------------------------- SSOL
def _norm_cdf(x):
    """Standard normal CDF via erf (Abramowitz-Stegun 7.1.26, |error| < 1.5e-7)."""
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


def _logsumexp_rows(a):
    m = np.max(a, axis=1, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=1, keepdims=True)))[:, 0]


def sector_pushforward(n_beams: int, grid: int = 360, max_cells: int = 30,
                       rho: float = 0.5, offset_cells: float = 0.0):
    """Push the full-grid sector prior forward to contiguous masks on the beams.

    Full grid: `grid` angular cells (wraparound). Sectors are the unique
    nonempty contiguous runs of 1..max_cells cells, M = grid * max_cells of them,
    each with prior (1 - rho) / M; the empty sector has prior rho (Eq. 5).
    A sector's mask is the set of selected beams whose bearing lies in it.
    Sectors that fall between beams add their mass to the empty mask.

    Returns (starts, lengths, log_prior, log_prior_empty) for the nonempty
    masks; every nonempty mask is a contiguous wraparound run of beams.
    """
    beam_cell = np.floor(offset_cells + np.arange(n_beams) * grid / n_beams).astype(int) % grid
    cell_to_beam = np.full(grid, -1)
    cell_to_beam[beam_cell] = np.arange(n_beams)
    p_sector = (1.0 - rho) / (grid * max_cells)
    mass = {}
    empty = rho
    for s in range(grid):
        covered = []
        for L in range(1, max_cells + 1):
            b = cell_to_beam[(s + L - 1) % grid]
            if b >= 0:
                covered.append(b)
            if not covered:
                empty += p_sector
                continue
            key = (covered[0], len(covered))  # contiguous run: (first beam, count)
            mass[key] = mass.get(key, 0.0) + p_sector
    keys = sorted(mass)
    starts = np.array([k[0] for k in keys], dtype=int)
    lengths = np.array([k[1] for k in keys], dtype=int)
    log_prior = np.log(np.array([mass[k] for k in keys]))
    return starts, lengths, log_prior, float(np.log(empty))


@dataclass
class SSOL:
    """Shared-Sector Occlusion Likelihood (B5), Eqs. (3)-(6) of the draft.

    L(z_B | x) = sum_mu pi_B(mu) prod_{b in mu} g_b(z_b | x) prod_{b not in mu} f_b(z_b | x)

    Observation alphabet A: `n_bins` range bins of width `bin_w` from
    `min_range` to `max_range`, plus a no-return symbol (nan).
      h_b: discretized hit mass N(r_b, sigma_hit) truncated to the valid bins,
           times (1 - p_none), plus p_none on no-return; all mass on no-return
           when nothing is in range along the ray.
      u:   uniform on A.
      t_b: exponential short-return mass on the valid bins below r_b,
           normalized over that support (t_b = u if the support is empty).
      f_b = (1 - eps) h_b + eps u      clean beam
      g_b = (1 - eta) t_b + eta u      occluded beam
    The sector prior is fixed (same at every pose) and sums to one, so a wrong
    pose cannot choose its own mask. Evaluated by log-sum-exp over masks with
    prefix sums of log g_b - log f_b: O(|B| + |M_B|) per particle.

    `combine="max"` replaces the sum by the best mask (comparator 10).
    """
    eps: float = 0.05
    eta: float = 0.05
    sigma_hit: float = 0.05
    lam_short: float = 1.0
    p_none: float = 0.01
    rho: float = 0.5
    max_sector_deg: float = 30.0
    grid: int = 360
    min_range: float = 0.15
    max_range: float = 12.0
    bin_w: float = 0.05
    combine: str = "sum"
    name: str = "ssol"

    def __post_init__(self):
        self.n_bins = int(np.ceil((self.max_range - self.min_range) / self.bin_w - 1e-9))
        self.log_u = -np.log(self.n_bins + 1)
        self._masks_for = None  # n_beams the masks were built for

    # ------------------------------------------------------------ prior/masks
    def _build_masks(self, n_beams: int):
        max_cells = int(round(self.max_sector_deg / 360.0 * self.grid))
        st, ln, lp, lp0 = sector_pushforward(n_beams, self.grid, max_cells, self.rho)
        self.m_start, self.m_len, self.m_logp, self.logp_empty = st, ln, lp, lp0
        self.k_max = int(ln.max())
        self._masks_for = n_beams

    def mask_prior(self, n_beams: int):
        """(starts, lengths, priors, prior_empty) as probabilities, for inspection."""
        if self._masks_for != n_beams:
            self._build_masks(n_beams)
        return self.m_start, self.m_len, np.exp(self.m_logp), np.exp(self.logp_empty)

    # -------------------------------------------------------------- emissions
    def log_emissions(self, z, r_exp):
        """Return (log f_b, log g_b), each (N, B)."""
        z = np.asarray(z, dtype=float)[None, :]
        none = np.isnan(z) | (z < self.min_range) | (z >= self.max_range)
        zb = np.clip(np.floor((np.where(none, self.min_range, z) - self.min_range) / self.bin_w),
                     0, self.n_bins - 1)
        lo = self.min_range + zb * self.bin_w
        hi = lo + self.bin_w
        r = np.asarray(r_exp, dtype=float)
        in_range = r < self.max_range
        rr = np.where(in_range, r, self.max_range)
        s = self.sigma_hit
        u = np.exp(self.log_u)

        # h_b: truncated, discretized Gaussian hit + no-return atom
        top = self.min_range + self.n_bins * self.bin_w
        z_norm = _norm_cdf((top - rr) / s) - _norm_cdf((self.min_range - rr) / s)
        hit = (_norm_cdf((hi - rr) / s) - _norm_cdf((lo - rr) / s)) / np.maximum(z_norm, 1e-300)
        h = np.where(none, np.where(in_range, self.p_none, 1.0),
                     np.where(in_range, (1.0 - self.p_none) * hit, 0.0))

        # t_b: exponential short-return mass on valid bins below r_b
        lam = self.lam_short
        n_sup = np.where(in_range, np.clip(np.ceil((rr - self.min_range) / self.bin_w), 0, self.n_bins),
                         self.n_bins)
        sup_top = self.min_range + n_sup * self.bin_w
        t_norm = np.exp(-lam * self.min_range) - np.exp(-lam * sup_top)
        t_bin = (np.exp(-lam * lo) - np.exp(-lam * hi)) / np.maximum(t_norm, 1e-300)
        t = np.where(~none & (zb < n_sup), t_bin, 0.0)
        t = np.where(n_sup == 0, u, t)

        f = (1.0 - self.eps) * h + self.eps * u
        g = (1.0 - self.eta) * t + self.eta * u
        return np.log(np.maximum(f, _LOG_FLOOR)), np.log(np.maximum(g, _LOG_FLOOR))

    # ------------------------------------------------------------- likelihood
    def _mask_terms(self, z, r_exp):
        """(log_f, terms): terms[:, 0] is the empty mask, terms[:, 1:] the M masks,
        each = log prior + sum over the mask of (log g_b - log f_b)."""
        n_beams = r_exp.shape[1]
        if self._masks_for != n_beams:
            self._build_masks(n_beams)
        log_f, log_g = self.log_emissions(z, r_exp)
        d = log_g - log_f                                         # (N, B)
        ext = np.concatenate([d, d[:, : self.k_max]], axis=1)     # wraparound
        cs = np.concatenate([np.zeros((len(d), 1)), np.cumsum(ext, axis=1)], axis=1)
        run = cs[:, self.m_start + self.m_len] - cs[:, self.m_start]   # (N, M) sum of d over each mask
        terms = np.concatenate([np.full((len(d), 1), self.logp_empty), self.m_logp[None, :] + run], axis=1)
        return log_f, terms

    def log_likelihood(self, z, r_exp):
        log_f, terms = self._mask_terms(z, r_exp)
        mix = terms.max(axis=1) if self.combine == "max" else _logsumexp_rows(terms)
        return log_f.sum(axis=1) + mix

    def occlusion_posterior(self, z, r_exp):
        """What SSOL believes about the occluder, for each pose.

        Returns (p_beam, p_any): p_beam (N, B) is the posterior probability
        that beam b lies inside the occluded sector; p_any (N,) is the
        posterior probability that the scan contains an occluder at all.
        """
        _, terms = self._mask_terms(z, r_exp)
        w = np.exp(terms - terms.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        n_beams = r_exp.shape[1]
        member = np.zeros((len(self.m_start), n_beams))
        for i, (s, k) in enumerate(zip(self.m_start, self.m_len)):
            member[i, (s + np.arange(k)) % n_beams] = 1.0
        return w[:, 1:] @ member, 1.0 - w[:, 0]
