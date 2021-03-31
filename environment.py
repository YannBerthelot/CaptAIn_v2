import os
import math
from math import cos, sin, ceil, floor
import numpy as np
from numpy import arcsin
from numpy.linalg import norm
from converter import converter
from aerodynamics import (
    compute_gamma,
    compute_Cx,
    compute_Cz,
    compute_Sx,
    compute_Sz,
    compute_fuel_variation,
    compute_drag,
    compute_altitude_factor,
    compute_alpha,
    compute_next_position,
)

from functools import lru_cache
from configparser import ConfigParser


parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)

PRECISION = 4
CACHE_SIZE = 100
SFC = eval(parser.get("flight_model", "SFC"))
DELTA_T = eval(parser.get("flight_model", "Timestep_size"))
RHO = eval(parser.get("flight_model", "Rho"))
C_X_MIN = eval(parser.get("flight_model", "C_x_min"))
C_Z_MAX = eval(parser.get("flight_model", "C_z_max"))
S_WINGS = eval(parser.get("flight_model", "Surface_wings"))
S_FRONT = eval(parser.get("flight_model", "Surface_front"))
g = 9.81
TASK = parser.get("flight_model", "Task")
CRITICAL_ENERGY = eval(parser.get("flight_model", "Critical_energy"))
LEVEL_TARGET = converter(eval(parser.get("task", "LEVEL_TARGET")), "feet", "m")


class FlightModel:
    def __init__(self, task="take-off"):

        self.task = task
        """
        CONSTANTS
        Constants used throughout the model
        """
        self.g = 9.81  # gravity vector in m.s-2
        self.init_mass = 73500
        self.m = 73500  # mass in kg
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
        self.fuel_mass = 23860 / 1.25  # Fuel mass at take-off in kg
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
        self.thrust_act_vec = [thrust for thrust in range(5, 11)]
        self.theta_act_vec = [theta for theta in range(0, 15)]

    def init_state(self):
        """
        VARIABLES
        """
        self.V_R_ok = False  # Has VR been reached
        self.crashed = False
        self.timestep = 0  # init timestep

        """
        DYNAMICS/INITIAL POSITION
        Acceleration, speed, position, theta and misc variable that will evolve at every timestep.
        """
        if self.task == "take-off":
            self.initial_altitude = 0
            self.A = [(0), (0)]  # Acceleration vector
            self.V = [(0), (0)]  # Speed Vector
            self.Pos = [(0), (self.initial_altitude)]  # Position vector
            self.theta = 0  # Angle between the plane's axis and the ground
            self.thrust = 0
            self.m = self.init_mass
            self.lift = 0
        elif self.task == "level-flight":
            self.initial_altitude = LEVEL_TARGET
            self.A = [(0), (0)]  # Acceleration vector
            self.V = [(245), (0)]  # Speed Vector
            self.Pos = [(0), (self.initial_altitude)]  # Position vector
            self.theta = 0  # Angle between the plane's axis and the ground
            self.thrust = self.THRUST_MAX * 0.7 * compute_altitude_factor(self.Pos[1])
        self.thrust_modified = 0  # Thrust after the influence of altitude factor
        self.Mach = norm(self.V) / 343  # Mach number

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
            self.obs = [self.Pos[1], self.V[0], self.V[1], self.A[0], self.A[1]]
        return self.obs

    def compute_dyna(self, thrust):
        """
        Compute the dynamcis : Acceleration, Speed and Position
        Speed(t+1) = Speed(t) + Acceleration(t) * Delta_t
        Position(t+1) = Position(t) + Speed(t) * Delta_t
        """
        # Update acceleration, speed and position
        old_A = self.A
        old_V = self.V

        self.A, self.V[1], self.Pos[1], self.crashed = self.compute_acceleration(
            thrust,
            self.V[0],
            self.V[1],
            self.theta,
            self.m,
            self.Mach,
            self.Pos[1],
            self.altitude_factor,
        )
        # print("old A", old_A, "new A", self.A)
        # print(
        #     "self.V[0], self.V[1], self.A[0], self.A[1]",
        #     self.V[0],
        #     self.V[1],
        #     self.A[0],
        #     self.A[1],
        # )
        self.V = compute_next_position(self.V[0], self.V[1], self.A[0], self.A[1])
        # if self.V[0] < 0:
        # print(
        #     "crashed",
        #     self.crashed,
        #     "thrust",
        #     thrust,
        #     "theta",
        #     self.theta,
        #     "V",
        #     self.V[0],
        #     self.V[1],
        #     "A",
        #     self.A[0],
        #     self.A[1],
        #     "old V",
        #     old_V,
        #     "old A",
        #     old_A,
        # )
        # exit()
        self.Mach = self.V[0] / 343

        self.Pos = compute_next_position(self.Pos[0], self.Pos[1], self.V[0], self.V[1])
        if self.V[0] > self.MAX_SPEED:
            self.V[0] = self.MAX_SPEED

        # Update plot lists
        # self.A_vec[0].append(self.A[0])
        # self.A_vec[1].append(self.A[1])
        self.V_vec[0].append(self.V[0])
        self.V_vec[1].append(self.V[1])
        self.Pos_vec[0].append(self.Pos[0])
        self.Pos_vec[1].append(self.Pos[1])

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

    def compute_acceleration(
        self, thrust, V_x, V_z, theta, m, Mach, altitude, altitude_factor
    ):
        """
        Compute the acceleration for a timestep based on the thrust by using Newton's second law : F = m.a <=> a = F/m with F the resultant of all forces
        applied on the oject, m its mass and a the acceleration o fthe object.
        Variables used:
        - P [Weight] in kg
        - V in m/s
        - gamma in rad
        - alpha in rad
        - S_x  in m^2
        - S_y in m^2
        - C_x (no units)
        - C_z (no units)
        On the vertical axis (z):
        F_z = Lift_z(alpha) * cos(theta) + Thrust * sin(theta) - Drag_z(alpha) * sin(gamma)  - P

        On the horizontal axis(x):
        F_x = Thrust_x  * cos(theta) - Drag_x(alpha) * cos(gamma) - Lift_x(alpha) * sin(theta)
        """
        # Compute the magnitude of the speed vector
        norm_V = norm([V_x, V_z])
        # print("norm_V", norm_V, V_x, V_z)
        # Compute gamma based on speed
        gamma = compute_gamma(V_z, norm_V)

        # Compute alpha based on gamma and theta
        # print("THETA", np.degrees(theta))
        alpha = compute_alpha(theta, gamma)

        # Compute P
        P = m * g
        # print("MASS", m)
        # Compute Drag magnitude
        S_x = compute_Sx(alpha)
        S_z = compute_Sz(alpha)
        C_x = compute_Cx(alpha, Mach)
        C_z = compute_Cz(alpha, Mach)

        if altitude > 122:
            flaps_factor = 1
        else:
            flaps_factor = 1.7

        drag = compute_drag(S_x, norm_V, C_x, altitude_factor) * flaps_factor

        # Compute lift magnitude
        self.lift = compute_drag(S_z, norm_V, C_z, altitude_factor) * flaps_factor

        # Newton's second law
        # Z-Axis
        # Project onto Z-axis
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        lift_z = cos_theta * self.lift
        drag_z = -sin(gamma) * drag
        thrust_z = sin_theta * thrust
        # Compute the sum
        F_z = lift_z + drag_z + thrust_z - P
        # print("Z lift", lift_z, "drag", drag_z, "thrust_z", thrust_z, "P", P)
        # X-Axis
        # Project on X-axis
        lift_x = -sin_theta * self.lift
        drag_x = -abs(cos(gamma) * drag)
        thrust_x = cos_theta * thrust

        # Compute the sum
        F_x = lift_x + drag_x + thrust_x
        # print("X lift", lift_x, "drag", drag_x, "thrust_x", thrust_x)
        # print(
        #     "theta",
        #     np.degrees(theta),
        #     "Vx",
        #     V_x,
        #     "drag",
        #     drag,
        #     "lift",
        #     lift,
        #     "F_x",
        #     F_x,
        #     "drag_x",
        #     drag_x,
        #     "lift_x",
        #     lift_x,
        # )
        # Check if we are on the ground, if so prevent from going underground by setting  vertical position and vertical speed to 0.
        crashed = False
        if TASK == "take-off":
            if (altitude <= 0) and (F_z <= 0):
                F_z = 0
                energy = 0.5 * m * altitude ** 2
                if energy > CRITICAL_ENERGY:
                    crashed = True
                V_z = 0
                altitude = 0

        # Compute Acceleration using a = F/m
        A = [F_x / m, F_z / m]

        # Append all the interesting values to their respective lists for monitoring

        self.lift_vec[0].append(lift_x)
        self.lift_vec[1].append(lift_z)
        # self.P_vec.append(P)
        # self.T_vec[1].append(thrust_z)
        # self.T_vec[0].append(thrust_x)
        self.drag_vec[0].append(drag_x)
        self.drag_vec[1].append(drag_z)
        self.alpha_vec.append(np.degrees(alpha))
        self.gamma_vec.append(np.degrees(gamma))
        self.theta_vec.append(np.degrees(self.theta))
        # print(self.theta_vec)
        self.S_vec[0].append(S_x)
        self.S_vec[1].append(S_z)
        self.C_vec[0].append(C_x)
        self.C_vec[1].append(C_z)

        return A, V_z, altitude, crashed

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
        thrust_modified = int(
            (action["thrust"][0]) * self.altitude_factor * self.THRUST_MAX
        )  # Apply the atitude factor to the thrust

        for i in range(1):
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

            self.theta_vec_act.append(np.degrees(self.theta) * 100 / 90)
            self.thrust_vec.append(self.thrust * 100 / self.THRUST_MAX)

            self.compute_dyna(self.thrust)

            # Fuel
            fuel_variation = compute_fuel_variation(self.thrust)
            self.fuel_mass += -fuel_variation
            self.m += -fuel_variation
            self.Fuel_vec.append(self.fuel_mass)

            self.get_obs()
            self.timestep += 1
        return self.obs


if __name__ == "__main__":
    model = FlightModel()
    for i in range(3):
        model.action_to_next_state_continuous({"theta": 0.3, "thrust": [0.9]})
