"""Assignment 4 (MountainCar) experiments for Problems 3c, 4d and 5b.

Re-runs exactly the configurations of the starter's `train.py` (same MDPs, agents and
hyper-parameters, 3 trials x 1000 training episodes + 500 greedy evaluation episodes), but
  * seeds every trial, so the numbers in the report are reproducible,
  * saves the reward plots (train.plot_rewards) straight into report/figures/04_mountaincar/,
  * prints summary statistics that train.py does not report.

Usage:  python scripts/mountaincar_experiments.py [agent ...]
        agents: value-iteration, tabular, function-approximation, fa-max-speed (default: all)
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MC_DIR = REPO_ROOT / "assignments" / "04_mountaincar"
FIG_DIR = REPO_ROOT / "report" / "figures" / "04_mountaincar"
os.environ.setdefault("MPLBACKEND", "Agg")
os.chdir(MC_DIR)
sys.path.insert(0, str(MC_DIR))

import gymnasium as gym                                       # noqa: E402
from submission import (FunctionApproxQLearning, ModelBasedMonteCarlo,  # noqa: E402
                        TabularQLearning, fourier_feature_extractor)
from train import plot_rewards                                # noqa: E402
from util import ContinuousGymMdp, DiscreteGymMdp, simulate    # noqa: E402


def discrete_mdp():
    return DiscreteGymMdp("MountainCar-v0", discount=0.999, low=[-1.2, -0.07], high=[0.6, 0.07],
                          feature_bins=20, time_limit=1000)


def fa_agent(mdp):
    return FunctionApproxQLearning(36, lambda s: fourier_feature_extractor(s, max_coeff=5, scale=[1, 15]),
                                   mdp.actions, mdp.discount, exploration_prob=0.2)


def fa_max_speed_mdp():
    if "CustomMountainCar-v0" not in gym.registry:
        gym.register(id="CustomMountainCar-v0", entry_point="custom_mountain_car:CustomMountainCarEnv",
                     max_episode_steps=1000, reward_threshold=-110.0)
    return ContinuousGymMdp("CustomMountainCar-v0", max_speed=100000, discount=0.999, time_limit=1000)


# name -> (figure prefix, MDP factory, agent factory); identical to train.py
CONFIGS = {
    "value-iteration": ("mcvi", discrete_mdp,
                        lambda m: ModelBasedMonteCarlo(m.actions, m.discount, calc_val_iter_every=1e5,
                                                       exploration_prob=0.50)),
    "tabular": ("tabular", discrete_mdp,
                lambda m: TabularQLearning(m.actions, m.discount, exploration_prob=0.15)),
    "function-approximation": ("fapprox",
                               lambda: ContinuousGymMdp("MountainCar-v0", discount=0.999, time_limit=1000),
                               fa_agent),
    "fa-max-speed": ("fapprox_maxspeed100000", fa_max_speed_mdp, fa_agent),
}


def run(agent_name: str) -> None:
    prefix, make_mdp, make_agent = CONFIGS[agent_name]
    print(f"\n=== {agent_name} ===")
    print(f"{'trial':>5} {'train mean (last 100)':>22} {'eval mean':>10} {'eval std':>9} "
          f"{'eval max':>9} {'reached goal':>13} {'time':>6}")
    for trial in range(1, 4):
        random.seed(trial)
        np.random.seed(trial)
        start = time.time()
        mdp = make_mdp()
        rl = make_agent(mdp)
        train_rewards = simulate(mdp, rl, train=True, num_trials=1000)

        # Count evaluation episodes that end in the terminal (goal) state rather than at the
        # 1000-step time limit, by observing the `terminal` flag of every transition.
        goals = [0]
        transition = mdp.transition

        def tracked_transition(action):
            next_state, reward, terminal = transition(action)
            goals[0] += bool(terminal)
            return next_state, reward, terminal

        mdp.transition = tracked_transition
        eval_rewards = simulate(mdp, rl, train=False, num_trials=500)
        reached = goals[0]
        print(f"{trial:>5} {np.mean(train_rewards[-100:]):>22.1f} {np.mean(eval_rewards):>10.1f} "
              f"{np.std(eval_rewards):>9.1f} {np.max(eval_rewards):>9.1f} {reached:>9}/500 "
              f"{time.time() - start:>5.0f}s")
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plot_rewards(train_rewards, eval_rewards, str(FIG_DIR / f"{prefix}_{trial}.png"), show=False)


if __name__ == "__main__":
    for name in (sys.argv[1:] or list(CONFIGS)):
        run(name)
