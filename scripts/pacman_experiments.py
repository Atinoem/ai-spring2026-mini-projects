"""Assignment 5 (Pac-Man) checks and experiments.

1. Root minimax values vs. the handout (minimaxClassic: 9, 8, 7, -492 for depths 1-4;
   mediumClassic: 9, 18, 27 for depths 1-3). An independent plain minimax computes the value
   of every root action, and we assert that MinimaxAgent and AlphaBetaAgent pick an action
   attaining the root value.
2. Problem 5: MinimaxAgent vs. ExpectimaxAgent on trappedClassic (depth 3, 50 seeded games).

Usage:  python scripts/pacman_experiments.py
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACMAN_DIR = REPO_ROOT / "assignments" / "05_pacman"
os.chdir(PACMAN_DIR)
sys.path.insert(0, str(PACMAN_DIR))

import layout      # noqa: E402
import pacman      # noqa: E402
import submission  # noqa: E402


def reference_value(state, depth, agent, num_agents):
    """Textbook depth-limited minimax with Eval = Utility = game score."""
    actions = state.get_legal_actions(agent)
    if state.is_win() or state.is_lose() or not actions or depth == 0:
        return state.get_score()
    nxt = (agent + 1) % num_agents
    nxt_depth = depth - 1 if nxt == 0 else depth
    values = [reference_value(state.generate_successor(agent, a), nxt_depth, nxt, num_agents) for a in actions]
    return max(values) if agent == 0 else min(values)


def check_root_values() -> None:
    print("1) Root minimax values vs. the handout")
    for name, expected in [("minimaxClassic", [9, 8, 7, -492]), ("mediumClassic", [9, 18, 27])]:
        lay = layout.get_layout(name)
        state = pacman.GameState()
        state.initialize(lay, lay.get_num_ghosts())
        n = state.get_num_agents()
        for depth, exp in enumerate(expected, start=1):
            child = {a: reference_value(state.generate_successor(0, a), depth, 1, n)
                     for a in state.get_legal_actions(0)}
            root = max(child.values())
            line = f"   {name:<15} depth {depth}: root value {root:>5} (handout {exp:>5})"
            assert root == exp, f"reference minimax disagrees with handout on {name} depth {depth}"
            for cls in (submission.MinimaxAgent, submission.AlphaBetaAgent):
                start = time.perf_counter()
                action = cls(depth=str(depth)).get_action(state)
                elapsed = time.perf_counter() - start
                assert child[action] == root, f"{cls.__name__} chose a suboptimal action"
                line += f" | {cls.__name__:<14} -> {action:<5} ({elapsed:.2f}s)"
            print(line)
    print("   OK: both agents choose actions attaining the handout's root values\n")


def trapped_classic(games: int = 50) -> None:
    print(f"2) trappedClassic, depth 3, {games} seeded games per agent")
    for agent in ("MinimaxAgent", "ExpectimaxAgent"):
        random.seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            results = pacman.run_games(**pacman.read_command(
                f"-p {agent} -l trappedClassic -a depth=3 -q -n {games}".split()))
        wins = sum(g.state.is_win() for g in results)
        scores = sorted({g.state.get_score() for g in results})
        print(f"   {agent:<16} won {wins:>2}/{games}; final scores observed: {scores}")


if __name__ == "__main__":
    check_root_values()
    trapped_classic()
