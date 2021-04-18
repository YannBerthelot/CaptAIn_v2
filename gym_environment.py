import os
import gym
import time
import pandas as pd
from gym import spaces
import numpy as np
from environment import FlightModel
from converter import converter
import matplotlib.pyplot as plt
from numpy.linalg import norm
from configparser import ConfigParser
from render import render

parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)

DELTA_T = eval(parser.get("flight_model", "Timestep_size"))
LEVEL_TARGET = converter(eval(parser.get("task", "LEVEL_TARGET")), "feet", "m")
MAX_TIMESTEP = 200 / DELTA_T
TAKE_OFF_ALTITUDE = converter(80, "feet", "m")  # 80 feets
RUNWAY_LENGTH = 5000  # 5000m


class PlaneEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}
    """Custom Environment that follows gym interface"""

    def __init__(self, task="take-off"):
        super(PlaneEnv, self).__init__()

        self.task = task

        # Fetch flight model
        self.FlightModel = FlightModel(task=self.task)

        # get the size if state vec depending on task
        self.STATES_DIM = len(self.FlightModel.obs)

        # Define action space (pitch, thrust)
        self.action_space = spaces.Box(np.array([-1, 0]), np.array([1, +1]))

        # Define state space
        self.observation_space = spaces.Box(
            np.float32(np.zeros(self.STATES_DIM)),
            np.float32(np.ones(self.STATES_DIM)),
            dtype=np.float32,
        )
        self.episode = 0

        # objectives init
        self.take_off = False
        self.overtime = False
        self.overrun = False

        # render init
        self.viewer = None

        # measure perf
        self.time_list = []
        self.time_list_env = []
        self.time_list_ats = []
        self.env_step_time = 0
        self.step_time = 0
        self.terminal_time = 0
        self.reward_time = 0
        self.n_steps = 0
        self.time_list_dyna = []

    def terminal(self):

        if self.task == "take-off":
            self.take_off = self.FlightModel.Pos[1] > TAKE_OFF_ALTITUDE
            self.overtime = self.FlightModel.timestep > MAX_TIMESTEP - 1
            self.overrun = self.FlightModel.Pos[0] > RUNWAY_LENGTH
            if self.take_off:
                self.reason_terminal = "Take-off"
            elif self.overtime:
                self.reason_terminal = "Overtime"
            elif self.overrun:
                self.reason_terminal = "Overrun"
            if self.take_off or self.overtime or self.overrun:
                return True
            else:
                return False
        elif self.task == "level-flight":
            self.overtime = self.FlightModel.timestep > MAX_TIMESTEP - 1
            if self.overtime:
                self.reason_terminal = "Overtime"
                return True

    def compute_reward(self, obs):
        if self.task == "take-off":
            reward = self.FlightModel.lift / 100000
            if self.take_off:
                reward = 3000 - 2 * (self.FlightModel.Pos[0] / (RUNWAY_LENGTH / 10))
        elif self.task == "level-flight":
            distance = abs(LEVEL_TARGET - self.FlightModel.Pos[1])
            reward = -distance / 1000
        return reward

    def step(self, action):
        step_start = time.time()
        action_dict = {"theta": action[0], "thrust": action[1]}

        # env step
        env_step_start = time.time()
        obs = self.FlightModel.action_to_next_state_continuous(action_dict)
        self.env_step_time += time.time() - env_step_start

        # terminal
        terminal_start = time.time()
        done = self.terminal()
        self.terminal_time += time.time() - terminal_start

        # reward
        reward_start = time.time()
        reward = self.compute_reward(obs)
        self.reward_time += time.time() - reward_start
        self.n_steps += 1
        if done:
            self.episode += 1
        self.step_time += time.time() - step_start
        return np.array(obs), reward, done, {}

    def reset(self):
        print(self.episode)
        # self.time_list.append(
        #     [
        #         self.step_time,
        #         self.env_step_time,
        #         self.terminal_time,
        #         self.reward_time,
        #         self.n_steps,

        #     ]
        # self.time_list_env.append(
        #     [
        #         self.FlightModel.get_obs_time,
        #         self.FlightModel.init_state_time,
        #         self.FlightModel.action_to_next_state_continuous_time,
        #     ]
        # )
        # self.time_list_ats.append(
        #     [
        #         self.FlightModel.compute_altitude_factor_time,
        #         self.FlightModel.compute_dyna_time,
        #         self.FlightModel.compute_fuel_time,
        #         self.FlightModel.compute_mach_time,
        #         self.FlightModel.clip_theta_time,
        #         self.FlightModel.clip_thrust_time,
        #         self.FlightModel.new_theta_time,
        #         self.FlightModel.new_thrust_time,
        #         self.FlightModel.get_obs_time,
        #     ]
        # )
        # self.time_list_dyna.append(self.FlightModel.dyna_times)
        # if self.episode == 100:
        #     pd.DataFrame(
        #         np.array(self.time_list_env),
        #         columns=[
        #             "obs time",
        #             "init_state  time",
        #             "action_to_next_state_continuous time",
        #         ],
        #     ).to_csv("time_env.csv", index=False)
        self.time_list_dyna.append(self.FlightModel.dyna_times.tolist())
        if self.episode == 100:
            pd.DataFrame(
                self.time_list_dyna,
                columns=[
                    "total_time",
                    "mach_time",
                    "gamma_time",
                    "P_time",
                    "sx_time",
                    "sz_time",
                    "cx_time",
                    "cz_time",
                    "flaps_time",
                    "drag_time",
                    "newton_time",
                    "new_pos_V_time",
                    "crashed_time",
                ],
            ).to_csv("time_dyna_numba.csv", index=False)
            # pd.DataFrame(
            #     np.array(self.time_list_ats),
            #     columns=[
            #         "compute_altitude_factor_time",
            #         "compute_dyna_time",
            #         "compute_fuel_time",
            #         "compute_mach_time",
            #         "clip_theta_time",
            #         "clip_thrust_time",
            #         "new_theta_time",
            #         "new_thrust_time",
            #         "get_obs_time",
            #     ],
            # ).to_csv("time_ats.csv", index=False)
            # pd.DataFrame(
            #     np.array(self.time_list),
            #     columns=[
            #         "step time",
            #         "env step time",
            #         "terminal time",
            #         "reward time",
            #         "n_steps",
            #     ],
            # ).to_csv("time_speed.csv", index=False)

        # measure performance
        self.step_time = 0
        self.env_step_time = 0
        self.terminal_time = 0
        self.reward_time = 0
        self.n_steps = 0
        self.FlightModel.get_obs_time = 0
        self.FlightModel.init_state_time = 0
        self.FlightModel.action_to_next_state_continuous_time = 0
        self.FlightModel.dyna_times = np.zeros(13)

        # objective reset
        self.take_off = False
        self.overtime = False
        self.overrun = False

        # state reset
        self.FlightModel.init_state()

        return np.array(self.FlightModel.obs)

    def render(self, mode="human"):
        self.viewer = render(self.task, self.viewer, self.FlightModel)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None