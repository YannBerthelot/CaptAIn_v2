import os
import logging
import warnings

from configparser import ConfigParser
from math import cos, sin, floor
import numpy as np
from numpy.linalg import norm
from converter import converter
from aerodynamics import (
    compute_fuel_variation,
    compute_altitude_factor,
    compute_dyna,
)
from utils import setup_logger


# create and configure parser
parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)

PRECISION = 4
CACHE_SIZE = 100
SFC = float(parser.get("flight_model", "SFC"))
DELTA_T = float(parser.get("flight_model", "Timestep_size"))
RHO = float(parser.get("flight_model", "Rho"))
C_X_MIN = float(parser.get("flight_model", "C_x_min"))
C_Z_MAX = float(parser.get("flight_model", "C_z_max"))
S_WINGS = float(parser.get("flight_model", "Surface_wings"))
S_FRONT = float(parser.get("flight_model", "Surface_front"))
g = 9.81
TASK = parser.get("flight_model", "Task")
CRITICAL_ENERGY = float(parser.get("flight_model", "Critical_energy"))
LEVEL_TARGET = converter(float(parser.get("task", "LEVEL_TARGET")), "feet", "m")
DEBUG = bool(parser.get("debug", "debug"))

# create and configure logger
os.makedirs("logs", exist_ok=True)
print("debug", DEBUG)
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOG_FORMAT = "%(levelno)s %(asctime)s %(funcName)s - %(message)s"
logger = setup_logger(
    "environment_logger",
    "logs/environment.log",
    level=level,
    format=LOG_FORMAT,
)


class FlightModel:
    def __init__(self, task="take-off"):

        self.task = task
        """
        CONSTANTS
        Constants used throughout the model
        """
        self.g = 9.81  # gravity vector in m.s-2
        self.init_mass = 54412.0
        self.init_fuel_mass = 23860 / 1.25
        self.fuel_mass = self.init_fuel_mass  # Fuel mass at take-off in kg
        self.m = self.init_mass + self.init_fuel_mass  # mass in kg
        self.RHO = 1.225  # air density in kg.m-3
        self.S_front = 12.6  # Frontal surface in m2
        self.S_wings = 122.6  # Wings surface in m2
        self.C_x_min = 0.095  # Drag coefficient
        self.C_z_max = 0.9  # Lift coefficient
        self.THRUST_MAX = 120000 * 2  # Max thrust in Newtons
        self.DELTA_T = 1  # Timestep size in seconds
        self.V_R = 77  # VR for takeoff (R is for Rotate)
        self.MAX_SPEED = 250  # Max speed bearable by the plane
        self.flaps_factor = 1.5  # Lift improvement due to flaps
        self.SFC = 17.5 / 1000  # Specific Fuel Consumption in kg/(N.s)

        self.M_critic = 0.78  # Critical Mach Number
        self.critical_energy = (
            1323000  # Maximal acceptable kinetic energy at landing in Joules
        )
        # initializations
        self.init_logs()
        self.init_state()
        self.get_obs()

    def init_logs(self):
        """
        LISTS FOR PLOT:
        Lists initialization to store values in order to monitor them through graphs
        """
        self.lift_vec = [[], []]  # Store lift values for each axis
        self.P_vec = []  # Store P values
        self.T_vec = [[], []]  # Store thrust values for both axis
        self.drag_vec = [[], []]  # Store drag values
        self.A_vec = [[], []]  # Store Acceleration values
        self.V_vec = [[], []]  # Store Speed values
        self.Pos_vec = [[], []]  # Store position values
        self.alt_factor_vec = []  # Store altitude factor values
        self.C_vec = [[], []]  # Store the coefficient values
        self.S_vec = [[], []]  # Store the reference surface values
        self.Fuel_vec = []
        self.thrust_vec = []
        # Store angle alpha values (angle between the speed vector and the plane)
        self.alpha_vec = []
        # Store angla gamma values (angle bweteen the ground and speed vector)
        self.gamma_vec = []
        # Store angle theta values (angle between the plane's axis and the ground)
        self.theta_vec = []
        self.theta_vec_act = []

        """
        KPIS:
        Useful information to print at the end of an episode
        """
        self.max_alt = 0  # Record maximal altitude reached
        self.max_A = 0  # Record maximal acceleration
        self.min_A = 0  # Record minimal acceleration
        self.max_V = 0  # Record maximal speed
        self.min_V = 0  # Record minimal speed

        """
        ACTIONS:
        Action vec for RL stocking thrust and theta values
        """

        # Represent the actions : couples of thrust and theta.
        self.action_vec = [
            [thrust, theta] for thrust in range(5, 11) for theta in range(0, 15)
        ]
        self.thrust_act_vec = range(5, 11)
        self.theta_act_vec = range(0, 15)

    def init_state(self):
        """
        VARIABLES
        """
        self.V_R_ok = False  # Has VR been reached
        self.crashed = False
        self.timestep = 0  # init timestep
        self.m = self.init_mass + self.init_fuel_mass

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
            self.m = self.init_mass
        elif self.task == "level-flight":
            self.initial_altitude = LEVEL_TARGET
            self.A = [0, 0]  # Acceleration vector
            self.V = [245, 0]  # Speed Vector
            self.Pos = [0, (self.initial_altitude)]  # Position vector
            self.theta = 0  # Angle between the plane's axis and the ground
            self.thrust = self.THRUST_MAX * 0.7 * compute_altitude_factor(self.Pos[1])
        self.Mach = norm(self.V) / 343
        self.thrust_modified = 0  # Thrust after the influence of altitude factor
        self.lift = 0

    def get_obs(self):

        """
        OBSERVATIONS
        States vec for RL stocking position and velocity
        """
        if self.task == "take-off":
            self.obs = [
                floor(self.Pos[0]),
                floor(self.Pos[1]),
                self.V[0],
                self.V[1],
            ]
        elif self.task == "level-flight":
            self.obs = [
                self.Pos[1],
                self.V[0],
                self.V[1],
                self.A[0],
                self.A[1],
                self.thrust,
                self.theta,
            ]
        logger.debug(f"{self.obs}")
        return self.obs

    def print_kpis(self):
        """
        Print interesting values : max alt, max and min acceleration, max and min speed.
        """
        print("max alt", self.max_alt)
        print("max A", self.max_A)
        print("min A", self.min_A)
        print("max V", self.max_V)
        print("min V", self.min_V)
        print("max x", max(self.Pos_vec[0]))

    def action_to_next_state_continuous(self, action):

        """
        Compute the dynamics of the the plane over a given numbero f episodes based on thrust and theta values
        Variables : Thrust in N, theta in degrees, number of episodes (no unit)
        This will be used by the RL environment.
        """
        self.altitude_factor = compute_altitude_factor(self.Pos[1])
        new_theta = np.radians(
            action["theta"] * 20
        )  # convert the pitch angle to radians

        thrust_modified = (action["thrust"][0]) * self.altitude_factor * self.THRUST_MAX
        # Apply the atitude factor to the thrust
        # logger.debug(f"iterate over timesteps")

        delta_thrust = np.clip(
            thrust_modified - self.thrust,
            -0.1 * self.THRUST_MAX,
            0.1 * self.THRUST_MAX,
        )
        # print("old theta", np.degrees(self.theta))
        self.thrust += delta_thrust
        # print("EAZAAZ", new_theta - self.theta)
        delta_theta = np.clip(new_theta - self.theta, np.radians(-1), np.radians(1))
        # print("delta theta", np.degrees(delta_theta))
        self.theta += delta_theta

        # print("new theta", np.degrees(self.theta))
        # Compute the dynamics for the episode

        # self.theta_vec_act.append(np.degrees(self.theta) * 100 / 90)
        # self.thrust_vec.append(self.thrust * 100 / self.THRUST_MAX)

        # compute new A, V, Pos
        # logger.debug(f"compute dyna")
        self.A, self.V, self.Pos, self.lift, self.crashed = compute_dyna(
            self.thrust,
            self.theta,
            np.array(self.A),
            np.array(self.V),
            np.array(self.Pos),
            self.m,
            self.altitude_factor,
        )
        self.Mach = norm(self.V) / 343
        # Fuel
        # logger.debug(f"fuel")
        fuel_variation = compute_fuel_variation(self.thrust)
        self.fuel_mass += -min(fuel_variation, self.fuel_mass)
        # self.m += -min(fuel_variation, self.fuel_mass)
        if self.m < self.init_mass:
            err = ValueError(
                f"Consumed more fuel than available m : {self.m}, fuel cons {fuel_variation}, available fuel : {self.fuel_mass}"
            )
            logger.error(err)
            raise err
        # self.Fuel_vec.append(self.fuel_mass)

        self.get_obs()
        self.timestep += 1
        return self.obs


if __name__ == "__main__":
    import time

    n_loops = int(1e6)
    model = FlightModel()
    start = time.time()
    for i in range(n_loops):
        model.action_to_next_state_continuous({"theta": 0.3, "thrust": [0.9]})
    print("Python", time.time() - start)
