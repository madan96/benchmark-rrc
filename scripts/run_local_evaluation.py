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
import numpy as np


def _init_env(goal_trajectory, difficulty,visualization=True):
    eval_config = {
        'action_space': 'torque_and_position',
        'frameskip': 3,
        'visualization': visualization,
        'monitor': False,
        'sim': True,
        'rank': 0
    }

    set_seed(0)
    move_cube_on_trajectory.seed(0)
    env = make_env_traj(goal_trajectory, **eval_config)
    return env

def generate_random_goal_trajectories():
    STEP_INTERVAL = 10000
    goal_trajectory = move_cube_on_trajectory.sample_goal()
    # Note we are missing a check to validate goals as present in environment file
    for i in range(len(goal_trajectory)):
        goal_trajectory[i] = (STEP_INTERVAL * i, goal_trajectory[i][1])
    return goal_trajectory

def main():
    parser = argparse.ArgumentParser('args')
    parser.add_argument('difficulty', type=int, default=3)
    parser.add_argument('method', type=str, help="The method to run. One of 'mp-pg', 'cic-cg', 'cpc-tg'")
    parser.add_argument('--goal', default=False, action='store_true')
    parser.add_argument('--evaluate', default=False, action='store_true')
    parser.add_argument('--residual', default=False, action='store_true',
                        help="add to use residual policies. Only compatible with difficulties 3 and 4.")
    parser.add_argument('--bo', default=False, action='store_true',
                        help="add to use BO optimized parameters.")
    args = parser.parse_args()

    total_rew = 0
    
    goal_trajectory = generate_random_goal_trajectories()
    env = _init_env(goal_trajectory, args.difficulty,visualization=False)
    state_machine = create_state_machine(args.difficulty, args.method, env,
                                                args.residual, args.bo)
    if args.evaluate:
        n_episodes = 10
        rewards = []
        for e in range(n_episodes):
            goal_trajectory =  generate_random_goal_trajectories()
            env.goal = goal_trajectory
            #####################
            # Run state machine
            #####################
            obs = env.reset()
            state_machine.reset()
            env.info["trajectory"] = goal_trajectory
            done = False
            total_rew = 0
            while not done:
                action = state_machine(obs)
                obs, rew, done, info = env.step(action)
                total_rew+=rew
                if(info['time_index']>info['trajectory'][-1][0]):
                    done=True
            rewards.append(total_rew)
        rewards = np.array(rewards)
        print("Evaluation over {} episodes- median: {} mean: {}, std: {}".format(n_episodes,np.median(rewards),np.mean(rewards),np.std(rewards)))
    else:
        goal_trajectory = generate_random_goal_trajectories()
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
            obs, rew, done, info = env.step(action)
            # print(info)
            if(info['time_index']>info['trajectory'][-1][0]):
                done=True
            total_rew+=rew
        print("Total reward: ",total_rew)

if __name__ == "__main__":
    main()
