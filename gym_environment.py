import os
import gym
import pandas as pd
from gym import spaces
import numpy as np
from environment import FlightModel
from converter import converter
import matplotlib.pyplot as plt
import pyglet
import shutil
from numpy.linalg import norm

TAKE_OFF_ALTITUDE = converter(80, "feet", "m")  # 80 feets
RUNWAY_LENGTH = 5000  # 5000m
from configparser import ConfigParser


parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)

DELTA_T = eval(parser.get("flight_model", "Timestep_size"))
LEVEL_TARGET = converter(eval(parser.get("task", "LEVEL_TARGET")), "feet", "m")
MAX_TIMESTEP = 200 / DELTA_T


class PlaneEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}
    """Custom Environment that follows gym interface"""

    def __init__(self, task="take-off"):
        super(PlaneEnv, self).__init__()
        # Fetch flight model
        self.task = task
        self.FlightModel = FlightModel(task=self.task)
        self.STATES_DIM = len(self.FlightModel.obs)

        # Define action space
        self.action_space = spaces.Box(
            np.array([-1, 0]), np.array([1, +1])
        )  # pitch, thrust
        # Define state space
        self.observation_space = spaces.Box(
            np.float32(np.zeros(self.STATES_DIM)),
            np.float32(np.ones(self.STATES_DIM)),
            dtype=np.float32,
        )
        self.take_off = False
        self.overtime = False
        self.overrun = False
        self.episode = 0
        self.viewer = None
        self.label = False
        self.sum_rewards = 0
        self.rewards = []
        self.rewards_1 = []
        self.rewards_2 = []
        shutil.rmtree("trajectories")
        os.makedirs("trajectories", exist_ok=True)

    def terminal(self):

        if self.task == "take-off":
            self.take_off = self.FlightModel.Pos[1] > TAKE_OFF_ALTITUDE
            self.overtime = self.FlightModel.timestep > MAX_TIMESTEP
            self.overrun = self.FlightModel.Pos[0] > RUNWAY_LENGTH
            if self.take_off:
                self.reason_terminal = "Take-off"
                # print(
                #     "Champagne",
                #     np.round(self.FlightModel.Pos[0], 0),
                #     np.round(self.FlightModel.V[0]),
                #     1,
                # )
            elif self.overtime:
                self.reason_terminal = "Overtime"
            elif self.overrun:
                self.reason_terminal = "Overrun"
            if self.take_off or self.overtime or self.overrun:
                return True
            else:
                return False
        elif self.task == "level-flight":
            self.overtime = self.FlightModel.timestep > MAX_TIMESTEP
            self.overspeed = self.FlightModel.Mach > 0.98
            self.over_g = norm(self.FlightModel.A) > (1.5 * 9.81)

            if self.overtime:
                self.reason_terminal = "Overtime"
                return True
            if self.overspeed:
                self.reason_terminal = "Overspeed"
                return True
            if self.over_g:
                self.reason_terminal = "Over G"
                print(norm(self.FlightModel.A))
                return True
            else:
                return False

    def compute_reward(self, obs):
        if self.task == "take-off":
            # reward_1 = 30 / (np.power(self.FlightModel.V[0], 1.0 / 3.0) + 1)
            reward_1 = self.FlightModel.Pos[0] / 1000
            reward_2 = 1 / max(1, self.FlightModel.Pos[1])
            reward = self.FlightModel.lift / 10000
            reward = reward / 10
            if self.take_off:
                reward = 3000 - 2 * (self.FlightModel.Pos[0] / (RUNWAY_LENGTH / 10))
        elif self.task == "level-flight":
            reward = -abs(LEVEL_TARGET - self.FlightModel.Pos[1]) - norm(
                self.FlightModel.A
            )
            if self.FlightModel.Mach > 0.95:
                reward += -100
            if self.over_g:
                reward += -1000

        return reward

    def step(self, action):
        action_dict = {"theta": action[0], "thrust": [action[1]]}
        obs = self.FlightModel.action_to_next_state_continuous(action_dict)
        done = self.terminal()
        reward = self.compute_reward(obs)
        self.rewards.append(reward)
        # self.rewards_1.append(reward_1)
        # self.rewards_2.append(reward_2)
        if done:
            self.sum_rewards = np.sum(self.rewards)
            if self.episode % 50 == 0:
                print("TARGET", LEVEL_TARGET)
                print(
                    f"Episode {self.episode}, State : {[np.round(x,2) for x in obs]}, Sum of rewards {np.round(self.sum_rewards,0)}, Episode length {self.FlightModel.timestep}, Result {self.reason_terminal}"
                )
                fig, ax = plt.subplots()
                plt.plot(self.FlightModel.Pos_vec[0], self.FlightModel.Pos_vec[1])
                plt.title("Trajectory")
                plt.savefig(f"trajectories/trajectory_{self.episode}.png")
                plt.close("all")

                fig, ax = plt.subplots()
                pd.Series(self.FlightModel.V_vec[0]).plot(ax=ax, label="Vx")
                pd.Series(self.FlightModel.V_vec[1]).plot(ax=ax, label="Vz")
                plt.legend()
                plt.title("Speeds over time")
                plt.savefig(f"trajectories/speeds_{self.episode}.png")

                fig, ax = plt.subplots()
                pd.Series(self.FlightModel.thrust_vec).plot(ax=ax, label="Thrust")
                pd.Series(self.FlightModel.theta_vec_act).plot(ax=ax, label="Theta")
                plt.legend()
                plt.title("Thrust and Pitch over time")
                plt.savefig(f"trajectories/tp_{self.episode}.png")

                fig, ax = plt.subplots()
                pd.Series(self.FlightModel.drag_vec[0]).plot(ax=ax, label="X")
                pd.Series(self.FlightModel.drag_vec[1]).plot(ax=ax, label="Y")
                plt.legend()
                plt.title("Drag over time")
                plt.savefig(f"trajectories/drag_{self.episode}.png")

                fig, ax = plt.subplots()
                pd.Series(self.FlightModel.S_vec[0]).plot(ax=ax, label="X")
                pd.Series(self.FlightModel.S_vec[1]).plot(ax=ax, label="Y")
                plt.legend()
                plt.title("S over time")
                plt.savefig(f"trajectories/S_{self.episode}.png")

                fig, ax = plt.subplots()
                pd.Series(self.FlightModel.C_vec[0]).plot(ax=ax, label="X")
                pd.Series(self.FlightModel.C_vec[1]).plot(ax=ax, label="Y")
                plt.legend()
                plt.title("C over time")
                plt.savefig(f"trajectories/C_{self.episode}.png")

                fig, ax = plt.subplots()
                pd.Series(self.FlightModel.lift_vec[0]).plot(ax=ax, label="X")
                pd.Series(self.FlightModel.lift_vec[1]).plot(ax=ax, label="Y")
                plt.legend()
                plt.title("lift over time")
                plt.savefig(f"trajectories/lift_{self.episode}.png")
                fig, ax = plt.subplots(3, 1, sharex="all")
                pd.Series(self.FlightModel.alpha_vec).plot(ax=ax[0], title="Alpha")
                pd.Series(self.FlightModel.gamma_vec).plot(ax=ax[1], title="Gamma")
                # print(len(pd.Series(self.FlightModel.theta_vec)))
                # print(pd.Series(self.FlightModel.theta_vec))
                pd.Series(self.FlightModel.theta_vec).plot(ax=ax[2], title="Theta")
                plt.tight_layout()
                plt.savefig(f"trajectories/angles_{self.episode}.png")
                plt.close("all")

                fig, ax = plt.subplots()
                pd.Series(self.rewards).plot(ax=ax, label="Full")
                pd.Series(self.rewards_1).plot(ax=ax, label="1")
                pd.Series(self.rewards_2).plot(ax=ax, label="2")
                plt.legend()
                plt.title("reward over time")
                plt.savefig(f"trajectories/rewards_{self.episode}.png")
                plt.close("all")

            self.episode += 1
        return np.array(obs), reward, done, {}

    def reset(self):
        self.take_off = False
        self.overtime = False
        self.overrun = False
        self.FlightModel.init_state()
        # print(self.FlightModel.theta_vec)
        self.FlightModel.init_logs()
        # print(self.FlightModel.theta_vec)
        self.sum_rewards = 0
        self.rewards = []
        self.rewards_1 = []
        self.rewards_2 = []
        return np.array(self.FlightModel.obs)

    def render(self, mode="human"):
        class DrawText:
            def __init__(self, label: pyglet.text.Label):
                self.label = label

            def render(self):
                self.label.draw()

        screen_width = 1800
        screen_height = 1000

        world_width = 500
        world_height = LEVEL_TARGET * 1.5

        scale = screen_width / world_width
        scale_y = screen_height / world_height
        carty = 100  # TOP OF CART

        cartwidth = 100.0
        cartheight = 20.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = (
                -cartwidth / 2,
                cartwidth / 2,
                cartheight / 2,
                -cartheight / 2,
            )

            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            # cart = rendering.Image("plane.png", 1.0, 1.0)

            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track = rendering.Line(
                (0, LEVEL_TARGET * scale_y + cartheight * 1.1 + (100 - cartheight)),
                (
                    screen_width,
                    LEVEL_TARGET * scale_y + cartheight * 1.1 + (100 - cartheight),
                ),
            )
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=screen_height * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(100, 100, 100, 100),
            )
            self.transform = rendering.Transform()

        # x = self.FlightModel.Pos[0]
        x = 250
        y = self.FlightModel.Pos[1]
        cartx = x * scale + cartwidth * 1.1  # MIDDLE OF CART
        carty = y * scale_y + cartheight * 1.1 + (100 - cartheight)  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.carttrans.set_rotation(self.FlightModel.theta)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None