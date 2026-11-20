"""Demo: Twin Wings layout + baseline MCL (independent beam mixture).

Runs one episode per occluder at the chosen asymmetry, prints commit outcomes
and saves two figures to out/:
  demo_layout.png    - both wings, doorway D, robot path, occluder, particles
  demo_mode_mass.png - posterior mass of the true wing over time per occluder

Usage (from ssol_sim/):  .venv/bin/python scripts/run_demo.py [--asym 0.3] [--seed 1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twinwings import SSOL, BeamMixture, EpisodeSpec, TwinWings, run_episode  # noqa: E402

LIKELIHOODS = {"beam_mixture": BeamMixture, "ssol": SSOL}
from twinwings.world import occluder_shapes  # noqa: E402

OCCLUDERS = [None, "box", "person", "chair"]
COLORS = {None: "0.4", "box": "tab:red", "person": "tab:orange", "chair": "tab:purple"}


def draw_world(ax, world, occ_xy=None, occ_kind=None):
    for x0, y0, x1, y1 in world.segments:
        ax.plot([x0, x1], [y0, y1], color="k", lw=1.6)
    dx0, dx1 = world.door_x
    ox = world.offset(world.door_wing)
    ax.annotate("D", ((dx0 + dx1) / 2 + ox, world.wing_h + world.asym + 0.15),
                ha="center", fontsize=11, weight="bold", color="tab:blue")
    for k, name in enumerate(["Wing A (true)", "Wing B (mirror)"]):
        ax.text(world.offset(k) + world.wing_w / 2, -0.55, name, ha="center", fontsize=10)
    if occ_xy is not None and occ_kind:
        segs, circles = occluder_shapes(occ_kind, occ_xy)
        for x0, y0, x1, y1 in segs:
            ax.plot([x0, x1], [y0, y1], color=COLORS[occ_kind], lw=2.5)
        for cx, cy, r in circles:
            ax.add_patch(plt.Circle((cx, cy), r, color=COLORS[occ_kind]))
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, world.offset(1) + world.wing_w + 0.5)
    ax.set_ylim(-0.9, world.wing_h + 0.8)
    ax.set_xticks([]), ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asym", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--likelihood", choices=sorted(LIKELIHOODS), default="beam_mixture")
    args = ap.parse_args()
    make_lik = LIKELIHOODS[args.likelihood]
    out = ROOT / "out"
    out.mkdir(exist_ok=True)

    results = {}
    print(f"Likelihood: {args.likelihood} | asym = {args.asym} m | tau_c = {args.tau}")
    print(f"{'occluder':9} {'outcome':8} {'t_commit':>8} {'beams hidden (mean, while standing)':>36} {'p95 update':>11}")
    for occ in OCCLUDERS:
        spec = EpisodeSpec(asym=args.asym, occluder=occ, seed=args.seed)
        r = run_episode(spec, make_lik())
        results[occ] = r
        k, tc = r.commit(args.tau)
        standing = r.times <= spec.dwell
        print(f"{str(occ):9} {r.outcome(args.tau):8} {tc:7.1f}s {r.n_occluded[standing].mean():36.1f} "
              f"{np.percentile(r.update_ms, 95):9.1f}ms")

    # ---- figure 1: layout + particles for the box episode
    world = TwinWings(asym=args.asym)
    r = results["box"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, ts, title in [(axes[0], 0.0, "t = 0 s: box hides doorway D"),
                          (axes[1], 30.0, "t = 30 s: box has walked away")]:
        i = int(np.argmin(np.abs(r.times - ts)))
        draw_world(ax, world, r.occluder_xy[i], "box")
        p, w = r.snapshots[ts]
        ax.scatter(p[:, 0], p[:, 1], s=4 + 400 * w, c="tab:green", alpha=0.5, label="particles")
        ax.plot(r.true_poses[:, 0], r.true_poses[:, 1], color="tab:blue", lw=1.2, label="robot path")
        x, y, th = r.true_poses[i]
        ax.arrow(x, y, 0.4 * np.cos(th), 0.4 * np.sin(th), width=0.06, color="tab:blue")
        ax.set_title(f"{title}   |   mass(Wing A) = {r.masses[i, 0]:.3f}", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Twin Wings, doorway asymmetry {args.asym} m, {args.likelihood} MCL", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"demo_layout_{args.likelihood}.png", dpi=140)

    # ---- figure 2: true-wing mass over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.3]})
    dwell = EpisodeSpec().dwell
    for occ, res in results.items():
        ax1.plot(res.times, res.masses[:, res.spec.true_wing], color=COLORS[occ],
                 lw=2, label=str(occ or "no occluder"))
        ax2.plot(res.times, res.n_occluded, color=COLORS[occ], lw=1.2)
    for ax in (ax1, ax2):
        ax.axvspan(0, dwell, color="0.9", zorder=0)
    ax1.axhline(args.tau, ls="--", color="k", lw=0.8)
    ax1.axhline(1 - args.tau, ls="--", color="k", lw=0.8)
    ax1.text(0.5, args.tau + 0.02, "commit to true wing", fontsize=8)
    ax1.text(0.5, 1 - args.tau - 0.07, "commit to wrong wing", fontsize=8)
    ax1.text(dwell / 2, 0.5, "occluder standing\nin doorway D", ha="center", fontsize=8, color="0.4")
    ax1.set_ylabel("posterior mass of true wing")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, loc="center right")
    ax2.set_ylabel("beams hidden")
    ax2.set_xlabel("time (s)")
    ax1.set_title(f"{args.likelihood} MCL, asymmetry {args.asym} m, 60 beams, 500 particles", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"demo_mode_mass_{args.likelihood}.png", dpi=140)
    print(f"\nSaved figures to {out}")


if __name__ == "__main__":
    main()
