import os
import logging
import warnings
import time
from configparser import ConfigParser
from math import cos, sin, floor
import numpy as np

# from numpy.linalg import norm
from converter import converter
from aerodynamics import (
    compute_fuel_variation,
    compute_altitude_factor,
    compute_dyna,
    norm,
)
from utils import setup_logger
from numba import njit

# create and configure parser
parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)

SFC = float(parser.get("flight_model", "SFC"))
DELTA_T = float(parser.get("flight_model", "Timestep_size"))
RHO = float(parser.get("flight_model", "Rho"))
C_X_MIN = float(parser.get("flight_model", "C_x_min"))
C_Z_MAX = float(parser.get("flight_model", "C_z_max"))
S_WINGS = float(parser.get("flight_model", "Surface_wings"))
S_FRONT = float(parser.get("flight_model", "Surface_front"))
THRUST_MAX = float(parser.get("flight_model", "Thrust_max"))
MACH_CRITIC = float(parser.get("flight_model", "Mach_critic"))
g = float(parser.get("flight_model", "g"))
INIT_MASS = float(parser.get("flight_model", "Initial_mass"))
INIT_FUEL_MASS = float(parser.get("flight_model", "Initial_fuel_mass"))
TASK = parser.get("flight_model", "Task")
CRITICAL_ENERGY = float(parser.get("flight_model", "Critical_energy"))
LEVEL_TARGET = converter(float(parser.get("task", "LEVEL_TARGET")), "feet", "m")
DEBUG = bool(parser.get("debug", "debug"))

clip = lambda x, l, u: l if x < l else u if x > u else x


class FlightModel:
    def __init__(self, task="take-off"):

        """
        CONSTANTS
        Constants used throughout the model
        """
        # measure performance
        self.init_state_time = 0
        self.get_obs_time = 0
        self.action_to_next_state_continuous_time = 0
        self.compute_altitude_factor_time = 0
        self.new_theta_time = 0
        self.new_thrust_time = 0
        self.clip_thrust_time = 0
        self.clip_theta_time = 0
        self.compute_dyna_time = 0
        self.compute_mach_time = 0
        self.compute_fuel_time = 0
        self.dyna_times = np.zeros(13)

        # initializations
        self.task = task
        self.init_state()
        self.obs = self.get_obs()
        self.times = []

    def init_state(self):
        """
        VARIABLES
        """
        start_time = time.process_time()
        self.crashed = False
        self.timestep = 0  # init timestep
        self.m = INIT_MASS + INIT_FUEL_MASS
        self.fuel_mass = INIT_FUEL_MASS

        """
        DYNAMICS/INITIAL POSITION
        Acceleration, speed, position, theta and misc variable that will evolve at every timestep.
        """
        if self.task == "take-off":
            self.initial_altitude = 0
            self.A = [0, 0]  # Acceleration vector
            self.V = [0, 0]  # Speed Vector
            self.Pos = [0, (self.initial_altitude)]  # Position vector
            self.theta = 0  # Angle between the plane's axis and the ground
            self.thrust = 0
            self.m = self.INIT_MASS
        elif self.task == "level-flight":
            self.initial_altitude = LEVEL_TARGET + np.random.randint(-1000, 1000)
            self.A = [0, 0]  # Acceleration vector
            self.V = [245.0, 0]  # Speed Vector
            self.Pos = [0, (self.initial_altitude)]  # Position vector
            self.theta = 0  # Angle between the plane's axis and the ground
            self.thrust = THRUST_MAX * 0.7 * compute_altitude_factor(self.Pos[1])
        self.Mach = norm(np.array(self.V)) / 343
        self.thrust_modified = 0  # Thrust after the influence of altitude factor
        self.lift = 0
        self.init_state_time += time.process_time() - start_time

    def get_obs(self):

        """
        OBSERVATIONS
        States vec for RL stocking position and velocity
        """
        start_time = time.process_time()
        if self.task == "take-off":
            obs = [
                floor(self.Pos[0]),
                floor(self.Pos[1]),
                self.V[0],
                self.V[1],
            ]
        elif self.task == "level-flight":
            obs = [
                self.Pos[1],
                self.V[0],
                self.V[1],
                self.A[0],
                self.A[1],
                self.thrust,
                self.theta,
            ]
        self.get_obs_time += time.process_time() - start_time
        return obs

    def action_to_next_state_continuous(self, action):

        """
        Compute the dynamics of the the plane over a given numbero f episodes based on thrust and theta values
        Variables : Thrust in N, theta in degrees, number of episodes (no unit)
        This will be used by the RL environment.
        """
        start_time = time.process_time()
        self.altitude_factor = compute_altitude_factor(self.Pos[1])
        self.compute_altitude_factor_time += time.process_time() - start_time

        start_new_theta = time.process_time()
        # convert the pitch angle to radians
        new_theta = action["theta"] * 20 * 0.0174533  # convert to radians
        self.new_theta_time += time.process_time() - start_new_theta

        # Apply the atitude factor to the thrust
        start_new_thrust = time.process_time()
        thrust_modified = (action["thrust"]) * self.altitude_factor * THRUST_MAX
        self.new_thrust_time += time.process_time() - start_new_thrust

        # Compute new thrust and pitch (theta) value based on previous value and agent input
        # thrust
        start_clip_thrust = time.process_time()
        delta_thrust = clip(
            thrust_modified - self.thrust,
            -0.1 * THRUST_MAX,
            0.1 * THRUST_MAX,
        )
        self.thrust += delta_thrust
        self.clip_thrust_time += time.process_time() - start_clip_thrust

        # theta
        start_clip_theta = time.process_time()
        delta_theta = clip(
            new_theta - self.theta, -0.017453292519943295, 0.017453292519943295
        )
        self.theta += delta_theta
        self.clip_theta_time += time.process_time() - start_clip_theta

        # Compute the dynamics for the episode
        # compute new A, V, Pos
        (self.A, self.V, self.Pos, self.lift, self.crashed, dyna_times,) = compute_dyna(
            self.thrust,
            self.theta,
            np.array(self.A),
            np.array(self.V),
            np.array(self.Pos),
            self.m,
            self.altitude_factor,
        )
        self.dyna_times += np.array(dyna_times)
        self.compute_dyna_time += dyna_times[0]

        # compute new mach number
        start_compute_mach = time.process_time()
        self.Mach = norm(np.array(self.V)) / 343
        self.compute_mach_time += time.process_time() - start_compute_mach

        # Fuel
        # Prevent to consume more fuel than there's left
        start_compute_fuel = time.process_time()
        fuel_variation = -min(compute_fuel_variation(self.thrust), self.fuel_mass)
        self.fuel_mass += fuel_variation
        self.m += fuel_variation
        self.compute_fuel_time += time.process_time() - start_compute_fuel

        # update the observation/state vector
        self.obs = self.get_obs()

        # keep track of time
        self.timestep += 1
        self.action_to_next_state_continuous_time += time.process_time() - start_time
        return self.obs


if __name__ == "__main__":
    import time

    n_loops = int(1e6)
    model = FlightModel()
    start = time.process_time()
    for i in range(n_loops):
        model.action_to_next_state_continuous({"theta": 0.3, "thrust": [0.9]})
    print("Python", time.process_time() - start)
