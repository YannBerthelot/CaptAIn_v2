import gym
from gym import spaces
import numpy as np
from environment import FlightModel
from converter import converter

import pyglet

TAKE_OFF_ALTITUDE = converter(80, "feet", "m")  # 80 feets
MAX_TIMESTEP = 200
RUNWAY_LENGTH = 5000  # 5000m


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
            np.array([0, 0]), np.array([1, +1])
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

    def terminal(self):

        if self.task == "take-off":
            self.take_off = self.FlightModel.Pos[1] > TAKE_OFF_ALTITUDE
            self.overtime = self.FlightModel.timestep > MAX_TIMESTEP
            self.overrun = self.FlightModel.Pos[0] > RUNWAY_LENGTH
            if self.take_off:
                self.reason_terminal = "Take-off"
                print("Champagne", self.FlightModel.Pos[0])
            elif self.overtime:
                self.reason_terminal = "Overtime"
            elif self.overrun:
                self.reason_terminal = "Overrun"
            if self.take_off or self.overtime or self.overrun:
                return True
            else:
                return False

    def compute_reward(self, obs):
        reward = -(1 / (self.FlightModel.V[0] + 1)) + self.FlightModel.Pos[1]
        if self.take_off:
            reward = 20 - (self.FlightModel.Pos[0] / (RUNWAY_LENGTH / 10))
        return reward

    def step(self, action):
        action_dict = {"theta": action[0], "thrust": [action[1]]}
        obs = self.FlightModel.action_to_next_state_continuous(action_dict)
        done = self.terminal()
        reward = self.compute_reward(obs)
        self.sum_rewards += reward
        if done:
            if self.episode % 100 == 0:
                print(
                    f"Episode {self.episode}, State : {[np.round(x,2) for x in obs]}, Sum of rewards {np.round(self.sum_rewards,0)}, Episode length {self.FlightModel.timestep}, Result {self.reason_terminal}"
                )
            self.episode += 1
        return np.array(obs), reward, done, {}

    def reset(self):
        self.take_off = False
        self.overtime = False
        self.overrun = False
        self.FlightModel.init_state()
        self.sum_rewards = 0
        return np.array(self.FlightModel.obs)

    def render(self, mode="human"):
        class DrawText:
            def __init__(self, label: pyglet.text.Label):
                self.label = label

            def render(self):
                self.label.draw()

        screen_width = 1800
        screen_height = 400

        world_width = 5000

        scale = screen_width / world_width
        carty = 100  # TOP OF CART

        cartwidth = 50.0
        cartheight = 30.0

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

        x = self.FlightModel.Pos[0]
        cartx = x * scale + cartwidth * 1.1  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.carttrans.set_rotation(self.FlightModel.theta)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None