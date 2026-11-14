"""Ablation study for Assignment 5, Problem 4 (better_evaluation_function).

Plays 20 games of depth-2 expectimax on smallClassic with the grader's seed ('testing') for
the submitted evaluation function and for variants that each remove or change one idea, so
the write-up (4b) can report what actually mattered.

Usage:  python scripts/pacman_eval_ablation.py
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACMAN_DIR = REPO_ROOT / "assignments" / "05_pacman"
os.chdir(PACMAN_DIR)
sys.path.insert(0, str(PACMAN_DIR))

import pacman      # noqa: E402
import submission  # noqa: E402
from util import manhattan_distance  # noqa: E402


def make_eval(maze_distances=True, hunt_scared=True, ghost_radius=1, food_terms=True):
    """The submitted evaluation function with switchable components."""
    def evaluate(state):
        if state.is_win() or state.is_lose():
            return state.get_score()
        pos = state.get_pacman_position()
        walls = state.get_walls()
        if maze_distances:
            dist, frontier = {pos: 0}, [pos]
            while frontier:
                nxt = []
                for (x, y) in frontier:
                    for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                        if not walls[n[0]][n[1]] and n not in dist:
                            dist[n] = dist[(x, y)] + 1
                            nxt.append(n)
                frontier = nxt
            d = lambda p: dist.get((int(p[0]), int(p[1])), 1000)
        else:
            d = lambda p: manhattan_distance(pos, (int(p[0]), int(p[1])))
        food = state.get_food().as_list()
        value = state.get_score()
        if food_terms:
            value += 10.0 / ((min(d(f) for f in food) if food else 0) + 1)
            value -= 4.0 * len(food)
            value -= 20.0 * len(state.get_capsules())
        for ghost in state.get_ghost_states():
            g = d(ghost.get_position())
            if hunt_scared and ghost.scaredTimer > g:
                value += 200.0 / (g + 1)
            elif g <= ghost_radius:
                value -= 500.0
            else:
                value -= 2.0 / g
        return value
    return evaluate


VARIANTS = [
    ("submitted (BFS distances, hunt scared ghosts, danger radius 1)", submission.better_evaluation_function),
    ("Manhattan instead of maze (BFS) distances", make_eval(maze_distances=False)),
    ("no scared-ghost hunting term", make_eval(hunt_scared=False)),
    ("danger radius 3 instead of 1 (more cautious)", make_eval(ghost_radius=3)),
    ("game score only (score_evaluation_function)", submission.score_evaluation_function),
]


def play(eval_fn):
    submission.better_evaluation_function = eval_fn
    submission.better = eval_fn
    random.seed("testing")
    params = "-l smallClassic -p ExpectimaxAgent -a evalFn=better -q -n 20 -c"
    with contextlib.redirect_stdout(io.StringIO()):
        games = pacman.run_games(**pacman.read_command(params.split()))
    scores = [g.state.get_score() for g in games]
    wins = [g.state.is_win() for g in games]
    win_scores = [s for s, w in zip(scores, wins) if w]
    return sum(wins), (sum(win_scores) / len(win_scores) if win_scores else float("nan")), sum(scores) / len(scores)


if __name__ == "__main__":
    print(f"{'variant':<64} {'wins':>5} {'avg win score':>14} {'avg score':>10}")
    for name, fn in VARIANTS:
        wins, avg_win, avg = play(fn)
        print(f"{name:<64} {wins:>3}/20 {avg_win:>14.1f} {avg:>10.1f}")
