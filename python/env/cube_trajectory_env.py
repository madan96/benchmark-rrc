"""Example Gym environment for the RRC 2021 Phase 2."""
import os
import enum
import shelve
import pickle as pkl
import typing

import gym
import numpy as np

import robot_fingers
import trifinger_simulation
import trifinger_simulation.visual_objects
from trifinger_simulation import trifingerpro_limits

import trifinger_simulation.tasks.move_cube_on_trajectory as task

from .pinocchio_utils import PinocchioUtils
from mp.const import CUSTOM_LOGDIR, INIT_JOINT_CONF, CUBOID_SIZE, CUBOID_MASS


class ActionType(enum.Enum):
    """Different action types that can be used to control the robot."""

    #: Use pure torque commands.  The action is a list of torques (one per
    #: joint) in this case.
    TORQUE = enum.auto()
    #: Use joint position commands.  The action is a list of angular joint
    #: positions (one per joint) in this case.  Internally a PD controller is
    #: executed for each action to determine the torques that are applied to
    #: the robot.
    POSITION = enum.auto()
    #: Use both torque and position commands.  In this case the action is a
    #: dictionary with keys "torque" and "position" which contain the
    #: corresponding lists of values (see above).  The torques resulting from
    #: the position controller are added to the torques in the action before
    #: applying them to the robot.
    TORQUE_AND_POSITION = enum.auto()


class BaseCubeTrajectoryEnv(gym.GoalEnv):
    """Gym environment for moving cubes with TriFingerPro."""

    def __init__(
        self,
        goal_trajectory: typing.Optional[task.Trajectory] = None,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 1,
        path: str = None,
    ):
        """Initialize.

        Args:
            goal_trajectory: Goal trajectory for the cube.  If ``None`` a new
                random trajectory is sampled upon reset.
            action_type: Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size:  Number of actual control steps to be performed in one
                call of step().
        """
        # Basic initialization
        # ====================

        if goal_trajectory is not None:
            task.validate_goal(goal_trajectory)
        self.goal = goal_trajectory

        self.action_type = action_type

        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        self.step_size = step_size

        # will be initialized in reset()
        self.platform = None

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        object_state_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=trifingerpro_limits.object_position.low,
                    high=trifingerpro_limits.object_position.high,
                ),
                "orientation": gym.spaces.Box(
                    low=trifingerpro_limits.object_orientation.low,
                    high=trifingerpro_limits.object_orientation.high,
                ),
            }
        )

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict(
            {
                "robot": gym.spaces.Dict(
                    {
                        "position": robot_position_space,
                        "velocity": robot_velocity_space,
                        "torque": robot_torque_space,
                        "tip_positions": gym.spaces.Box(
                            low=np.array([trifingerpro_limits.object_position.low] * 3),
                            high=np.array([trifingerpro_limits.object_position.high] * 3),
                        ),
                        "tip_force": gym.spaces.Box(low=np.zeros(3),
                                                    high=np.ones(3))
                    }
                ),
                "action": self.action_space,
                "desired_goal": object_state_space,
                "achieved_goal": object_state_space,
            }
        )
        self.pinocchio_utils = PinocchioUtils()
        self.custom_logs = {}
        self.path = path

    def compute_reward(
        self,
        achieved_goal: task.Position,
        desired_goal: task.Position,
        info: dict,
    ) -> float:
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal: Current position of the object.
            desired_goal: Goal position of the current trajectory step.
            info: An info dictionary containing a field "time_index" which
                contains the time index of the achieved_goal.

        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true::

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
        # This is just some sanity check to verify that the given desired_goal
        # actually matches with the active goal in the trajectory.
        active_goal = np.asarray(
            task.get_active_goal(
                self.info["trajectory"], self.info["time_index"]
            )
        )
        assert np.all(active_goal == desired_goal), "{}: {} != {}".format(
            info["time_index"], active_goal, desired_goal
        )

        return -task.evaluate_state(
            info["trajectory"], info["time_index"], achieved_goal
        )

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.

        .. note::

           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.

        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        task.seed(seed)
        return [seed]

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        object_observation = camera_observation.filtered_object_pose

        active_goal = np.asarray(
            task.get_active_goal(self.info["trajectory"], t)
        )

        observation = {
            "robot": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
                "tip_positions": np.array(self.pinocchio_utils.forward_kinematics(robot_observation.position)),
                "tip_force": robot_observation.tip_force,
            },
            "action": action,
            "desired_goal": {
                "position": active_goal,
                "orientation": object_observation.orientation
            },
            "achieved_goal": {
                "position": object_observation.position,
                "orientation": object_observation.orientation,
            },
        }
        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action


class SimCubeTrajectoryEnv(BaseCubeTrajectoryEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        goal_trajectory: typing.Optional[task.Trajectory] = None,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 1,
        visualization: bool = False,
    ):
        """Initialize.

        Args:
            goal_trajectory: Goal trajectory for the cube.  If ``None`` a new
                random trajectory is sampled upon reset.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        super().__init__(
            goal_trajectory=goal_trajectory,
            action_type=action_type,
            step_size=step_size,
        )
        self.visualization = visualization
        self.difficulty = 3
        self.info = {"difficulty": 3}
        self.simulation = True
        self.frameskip = step_size

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.step_count + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > task.EPISODE_LENGTH:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            # update goal visualization
            if self.visualization:
                goal_position = task.get_active_goal(
                    self.info["trajectory"], t
                )
                self.goal_marker.set_state(goal_position, (0, 0, 0, 1))

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            self.info["time_index"] = t + 1

            # Alternatively use the observation of step t.  This is the
            # observation from the moment before action_t is applied, i.e. the
            # result of that action is not yet visible in this observation.
            #
            # When using this observation, the resulting cumulative reward
            # should match exactly the one computed during replay (with the
            # above it will differ slightly).
            #
            # self.info["time_index"] = t

            observation = self._create_observation(
                self.info["time_index"], action
            )

            reward += self.compute_reward(
                observation["achieved_goal"]["position"],
                observation["desired_goal"]["position"],
                self.info,
            )

        is_done = self.step_count >= task.EPISODE_LENGTH
        self.info['difficulty'] = 3

        return observation, reward, is_done, self.info

    def reset(self):
        """Reset the environment."""
        # hard-reset simulation
        del self.platform

        # initialize simulation
        initial_robot_position = trifingerpro_limits.robot_position.default
        # initialize cube at the centre
        initial_object_pose = task.move_cube.Pose(
            position=task.INITIAL_CUBE_POSITION
        )

        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=initial_object_pose,
        )

        # if no goal is given, sample one randomly
        if self.goal is None:
            trajectory = task.sample_goal()
        else:
            trajectory = self.goal

        # visualize the goal
        if self.visualization:
            self.goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                width=task.move_cube._CUBE_WIDTH,
                position=trajectory[0][1],
                orientation=(0, 0, 0, 1),
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )

        self.info = {"time_index": -1, "trajectory": trajectory}

        self.step_count = 0

        return self._create_observation(0, self._initial_action)

    def register_custom_log(self, name, data):
        if name in self.custom_logs:
            self.custom_logs[name].append({
                'step_count': self.step_count,
                'data': data
            })
        else:
            self.custom_logs[name] = [{
                'step_count': self.step_count,
                'data': data
            }]
    
    def save_custom_logs(self):
        print('saving custom logs...')
        custom_logdir = None
        if not(self.path is None):
            custom_logdir = self.path
        elif not os.path.isdir(CUSTOM_LOGDIR):
            print('{} does not exist. skip saving custom logs.'.format(CUSTOM_LOGDIR))
            return
        else:
            custom_logdir = CUSTOM_LOGDIR
        path = os.path.join(custom_logdir, 'custom_data')
        with shelve.open(path, writeback=True) as f:
            for key, val in self.custom_logs.items():
                f[key] = val

        # save the rewards
        path = os.path.join(custom_logdir, 'reward.pkl')
        with open(path, 'wb') as handle:
            pkl.dump(self.reward_list,handle)

        # if ran in simulation save the observation
        if (self.simulation):
            path = os.path.join(custom_logdir, 'observations.pkl')
            with open(path, 'wb') as handle:
                pkl.dump(self.observation_list,handle)

        # store the goal to a file, i.e. the last goal,...
        import json
        goal_file = os.path.join(custom_logdir, "goal.json")
        goal_info = {
            "difficulty": self.difficulty,
            "goal": json.loads(json.dumps({
        'position': self.goal["position"].tolist(),
        'orientation': self.goal["orientation"].tolist(),
        })),
            "changegoal": self.change_goal_last,
            "reachstart": self.reach_start_point,
            "reachfinish": self.reach_finish_point,
            "align_obj_error": self.align_obj_error,
            "init_align_obj_error": self.init_align_obj_error,
            "init_obj_pose": self.init_obj_pose,
        }
        with open(goal_file, "w") as fh:
            json.dump(goal_info, fh, indent=4)

class RealRobotCubeTrajectoryEnv(BaseCubeTrajectoryEnv):
    """Gym environment for moving cubes with real TriFingerPro."""



    def __init__(
        self,
        goal_trajectory: typing.Optional[task.Trajectory] = None,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 1,
        visualization: bool = False,
    ):
        """Initialize.

        Args:
            goal_trajectory: Goal trajectory for the cube.  If ``None`` a new
                random trajectory is sampled upon reset.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        super().__init__(
            goal_trajectory=goal_trajectory,
            action_type=action_type,
            step_size=step_size,
        )
        self.visualization = visualization
        self.difficulty = 3
        self.info = {"difficulty": 3}
        self.frameskip = step_size
        self.simulation = False



    def step(self, action):
        """Run one timestep of the environment's dynamics.

        Important: ``reset()`` needs to be called before doing the first step.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.info["time_index"] + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)

        reward = 0.0

        for _ in range(num_steps):
            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            self.info["time_index"] = t

            observation = self._create_observation(t, action)
            reward += self.compute_reward(
                observation["achieved_goal"]["position"],
                observation["desired_goal"]["position"],
                self.info,
            )
            # reward += self.compute_reward(
            #     observation["achieved_goal"],
            #     observation["desired_goal"],
            #     self.info,
            # )

            # make sure to not exceed the episode length
            if t >= task.EPISODE_LENGTH - 1:
                break

        is_done = t >= task.EPISODE_LENGTH
        self.info['difficulty'] = 3

        return observation, reward, is_done, self.info

    def reset(self):
        # cannot reset multiple times
        if self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )

        self.platform = robot_fingers.TriFingerPlatformWithObjectFrontend()

        # if no goal is given, sample one randomly
        if self.goal is None:
            trajectory = task.sample_goal()
        else:
            trajectory = self.goal

        self.info = {"time_index": -1, "trajectory": trajectory}

        # need to already do one step to get initial observation
        # TODO disable frameskip here?
        observation, _, _, _ = self.step(self._initial_action)

        return observation

    def register_custom_log(self, name, data):
        if name in self.custom_logs:
            self.custom_logs[name].append({
                'step_count': self.step_count,
                'data': data
            })
        else:
            self.custom_logs[name] = [{
                'step_count': self.step_count,
                'data': data
            }]
    
    def save_custom_logs(self):
        print('saving custom logs...')
        custom_logdir = None
        if not(self.path is None):
            custom_logdir = self.path
        elif not os.path.isdir(CUSTOM_LOGDIR):
            print('{} does not exist. skip saving custom logs.'.format(CUSTOM_LOGDIR))
            return
        else:
            custom_logdir = CUSTOM_LOGDIR
        path = os.path.join(custom_logdir, 'custom_data')
        with shelve.open(path, writeback=True) as f:
            for key, val in self.custom_logs.items():
                f[key] = val

        # save the rewards
        path = os.path.join(custom_logdir, 'reward.pkl')
        with open(path, 'wb') as handle:
            pkl.dump(self.reward_list,handle)

        # if ran in simulation save the observation
        if (self.simulation):
            path = os.path.join(custom_logdir, 'observations.pkl')
            with open(path, 'wb') as handle:
                pkl.dump(self.observation_list,handle)

        # store the goal to a file, i.e. the last goal,...
        import json
        goal_file = os.path.join(custom_logdir, "goal.json")
        goal_info = {
            "difficulty": self.difficulty,
            "goal": json.loads(json.dumps({
        'position': self.goal["position"].tolist(),
        'orientation': self.goal["orientation"].tolist(),
        })),
            "changegoal": self.change_goal_last,
            "reachstart": self.reach_start_point,
            "reachfinish": self.reach_finish_point,
            "align_obj_error": self.align_obj_error,
            "init_align_obj_error": self.init_align_obj_error,
            "init_obj_pose": self.init_obj_pose,
        }
        with open(goal_file, "w") as fh:
            json.dump(goal_info, fh, indent=4)
