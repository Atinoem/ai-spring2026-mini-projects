"""Assignment 6, Problem 4d: exact inference vs. particle filtering, without the Tk GUI.

`drive.py` shows the two trackers visually. This script measures the same comparison
quantitatively on the Lombard map: a simulated car moves according to the learned
transition model, our car stays at a fixed position, and both trackers receive the same
noisy microphone readings D_t ~ N(true distance, SONAR_STD^2). After every observation we
record each tracker's error and how many tiles its belief covers.

Usage:  python scripts/car_tracking_comparison.py
"""

from __future__ import annotations

import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CAR_DIR = REPO_ROOT / "assignments" / "06_bayesian"
os.chdir(CAR_DIR)
sys.path.insert(0, str(CAR_DIR))

import util                                   # noqa: E402
from engine.const import Const                # noqa: E402
from particle_filter import ParticleFilter    # noqa: E402
from submission import ExactInference         # noqa: E402

RUNS, STEPS = 20, 40


def tile_center(tile):
    return util.colToX(tile[1]), util.rowToY(tile[0])


def evaluate(belief, true_tile):
    """(distance of the most likely tile, expected distance, #tiles with probability > 1%)."""
    tx, ty = tile_center(true_tile)
    best, best_p, expected, support = None, -1.0, 0.0, 0
    for r in range(belief.getNumRows()):
        for c in range(belief.getNumCols()):
            p = belief.getProb(r, c)
            if p > best_p:
                best, best_p = (r, c), p
            if p > 0.01:
                support += 1
            if p > 0:
                x, y = tile_center((r, c))
                expected += p * math.hypot(x - tx, y - ty)
    bx, by = tile_center(best)
    return math.hypot(bx - tx, by - ty), expected, support


def main() -> None:
    trans = util.loadTransProb()
    tiles = {t for pair in trans for t in pair}
    num_rows = max(t[0] for t in tiles) + 1
    num_cols = max(t[1] for t in tiles) + 1
    successors = {}
    for (old, new), p in trans.items():
        successors.setdefault(old, {})[new] = p
    starts = sorted(successors)

    stats = {"exact": [], "particle": []}
    timing = {"exact": 0.0, "particle": 0.0}
    for run in range(RUNS):
        random.seed(run)
        car = random.choice(starts)
        agent_x = util.colToX(num_cols // 2)          # our car parked mid-map
        agent_y = util.rowToY(num_rows // 2)
        trackers = {"exact": ExactInference(num_rows, num_cols), "particle": ParticleFilter(num_rows, num_cols)}
        for _ in range(STEPS):
            cx, cy = tile_center(car)
            observed = math.hypot(cx - agent_x, cy - agent_y) + random.gauss(0, Const.SONAR_STD)
            for name, tracker in trackers.items():
                start = time.perf_counter()
                tracker.observe(agent_x, agent_y, observed)
                timing[name] += time.perf_counter() - start
                stats[name].append(evaluate(tracker.getBelief(), car))
                start = time.perf_counter()
                tracker.elapseTime()
                timing[name] += time.perf_counter() - start
            if car in successors:                    # the true car moves
                car = util.weightedRandomChoice(successors[car])

    steps = RUNS * STEPS
    print(f"Lombard map: {num_rows} x {num_cols} = {num_rows * num_cols} tiles, {RUNS} runs x {STEPS} steps, "
          f"{ParticleFilter.NUM_PARTICLES} particles, sonar std {Const.SONAR_STD}")
    print(f"{'tracker':<10} {'MAP-tile error':>15} {'expected error':>15} {'tiles >1%':>10} {'ms/step':>8}")
    for name, rows in stats.items():
        map_err = statistics.mean(r[0] for r in rows)
        exp_err = statistics.mean(r[1] for r in rows)
        support = statistics.mean(r[2] for r in rows)
        print(f"{name:<10} {map_err:>15.1f} {exp_err:>15.1f} {support:>10.1f} {1000 * timing[name] / steps:>8.2f}")
    print("(errors in map units; one tile is", Const.BELIEF_TILE_SIZE, "units wide)")


if __name__ == "__main__":
    main()
