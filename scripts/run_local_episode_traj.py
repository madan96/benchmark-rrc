#!/usr/bin/env python3
"""Run a single episode with a controller in simulation."""
import argparse
import json
import os
from typing import get_type_hints

from env.make_env import make_env, make_env_traj
from trifinger_simulation.tasks import move_cube
from trifinger_simulation.tasks import move_cube_on_trajectory
from mp.utils import set_seed
from combined_code import create_state_machine


def _init_env(goal_trajectory, difficulty):
    eval_config = {
        'action_space': 'torque_and_position',
        'frameskip': 3,
        'visualization': True,
        'monitor': False,
        'sim': True,
        'rank': 0
    }

    set_seed(0)
    env = make_env_traj(goal_trajectory, **eval_config)
    return env


def main():
    parser = argparse.ArgumentParser('args')
    parser.add_argument('difficulty', type=int, default=3)
    parser.add_argument('method', type=str, help="The method to run. One of 'mp-pg', 'cic-cg', 'cpc-tg'")
    parser.add_argument('--goal', default=False, action='store_true')
    parser.add_argument('--residual', default=False, action='store_true',
                        help="add to use residual policies. Only compatible with difficulties 3 and 4.")
    parser.add_argument('--bo', default=False, action='store_true',
                        help="add to use BO optimized parameters.")
    args = parser.parse_args()

    if args.goal:
        goal_trajectory = [
        [0, [0, 0, 0.08]],
        [5000, [0, 0.07, 0.08]],
        [10000, [0.07, 0.07, 0.08]],
        [15000, [0.07, 0, 0.08]],
        [20000, [0.07, -0.07, 0.08]],
        [40000, [0, -0.07, 0.08]],
        [50000, [-0.07, -0.07, 0.06]],
        [70000, [-0.07, 0, 0.08]],
        [80000, [-0.07, 0.07, 0.08]],
        [90000, [0, 0.07, 0.08]],
        [100000, [0, 0, 0.08]]
    ]
    else:
        goal_trajectory = move_cube_on_trajectory.sample_goal()
        STEP_INTERVAL = 3000
        for i in range(len(goal_trajectory)):
            goal_trajectory[i] = (STEP_INTERVAL * i, goal_trajectory[i][1])

    env = _init_env(goal_trajectory, args.difficulty)
    state_machine = create_state_machine(args.difficulty, args.method, env,
                                         args.residual, args.bo)

    #####################
    # Run state machine
    #####################
    obs = env.reset()
    state_machine.reset()

    done = False
    while not done:
        action = state_machine(obs)
        obs, _, done, _ = env.step(action)


if __name__ == "__main__":
    main()
