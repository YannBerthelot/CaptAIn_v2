import math
from math import cos, sin, ceil, floor
import numpy as np
from numpy import arcsin
from numpy.linalg import norm
from converter import converter
from aerodynamics import gamma

from functools import cache
from configparser import ConfigParser

parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)

PRECISION = 4


class FlightModel:
    def __init__(self, task="take-off"):

        self.task = task
        """
        CONSTANTS
        Constants used throughout the model
        """
        self.g = 9.81  # gravity vector in m.s-2
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

        elif self.task == "level-flight":
            self.initial_altitude = 10000
            self.A = [(0), (0)]  # Acceleration vector
            self.V = [(245), (0)]  # Speed Vector
            self.Pos = [(0), (self.initial_altitude)]  # Position vector
            self.theta = 0  # Angle between the plane's axis and the ground
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

    def fuel_consumption(self):
        """
        Compute the fuel mass variation at each timestep based on the thrust.
        Update the remaining fuell mass and plane mass accordingly
        """
        fuel_variation = self.SFC * self.DELTA_T * self.thrust_modified / 1000
        self.fuel_mass += -fuel_variation
        self.m += -fuel_variation
        self.Fuel_vec.append(self.fuel_mass)

    def drag(self, S, V, C):
        """
        Compute the drag using:
        S : Surface peripendicular to the drag direction
        V : Speed colinear with the drag direction
        C : Coefficient of drag/lift regarding the drag direction
        RHO : air density
        F = 1/2 * S * C * V^2
        """
        return 0.5 * self.RHO * self.altitude_factor() * S * C * np.power(V, 2)

    
            return maximal - 0.8 * (self.Mach - M_d)

    def C_x(self, alpha):
        """
        Compute the drag coefficient at M = 0 depending on alpha (the higher alpha the higher the drag)
        """
        alpha = alpha + np.radians(0)
        C_x = (np.degrees(alpha) * 0.02) ** 2 + self.C_x_min
        return self.Mach_Cx(np.round(C_x, PRECISION),np.round(self.Mach, PRECISION))

    def C_z(self, alpha):
        """
        Compute the lift coefficient at M=0 depending on alpha (the higher the alpha, the higher the lift until stall)
        """
        alpha = alpha + np.radians(5)
        # return self.C_z_max
        sign = np.sign(np.degrees(alpha))
        if abs(np.degrees(alpha)) < 15:
            # Quadratic evolution  from C_z = 0 for 0 degrees and reaching a max value of C_z = 1.5 for 15 degrees
            C_z = sign * abs((np.degrees(alpha) / 15) * self.C_z_max)
        elif abs(np.degrees(alpha)) < 20:
            # Quadratic evolution  from C_z = 1.5 for 15 degrees to C_2 ~ 1.2 for 20 degrees.
            C_z = sign * abs((1 - ((abs(np.degrees(alpha)) - 15) / 15)) * self.C_z_max)
        else:
            ##if alpha > 20 degrees : Stall => C_z = 0
            C_z = 0
        C_z = self.Mach_Cz(np.round(C_z, 4), np.round(self.Mach, 4))
        return C_z

    def alpha(self, gamma):
        """
        Compute alpha (the angle between the plane's axis and the speed vector).
        alpha = theta - gamma
        """
        alpha = self.theta - gamma
        return np.round(alpha, PRECISION)

    def S_x(self, alpha):
        """
        update the value of the surface orthogonal to the speed vector depending on alpha by projecting the x and z surface.
        S_x = cos(alpha)*S_front + sin(alpha) * S_wings
        """
        alpha = abs(alpha)
        return np.round(
            cos(alpha) * self.S_front + sin(alpha) * self.S_wings, PRECISION
        )

    def S_z(self, alpha):
        """
        update the value of the surface colinear to the speed vector depending on alpha by projecting the x and z surface.
        S_x = sin(alpha)*S_front + cos(alpha) * S_wings
        !IMPORTANT!
        The min allows the function to be stable, I don't understand why yet.
        """
        alpha = abs(alpha)
        return np.round(
            (sin(alpha) * self.S_front) + (cos(alpha) * self.S_wings), PRECISION
        )

    def check_colisions(self):
        """
        Check if the plane is touching the ground
        """
        return self.Pos[1] <= 0

    def compute_acceleration(self, thrust):
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
        V = norm(self.V)
        # Compute gamma based on speed
        gamma = gamma(V)
        # Compute alpha based on gamma and theta
        alpha = self.alpha(gamma)

        # Compute P
        P = self.m * self.g

        # Compute Drag magnitude
        S_x = self.S_x(alpha)
        S_z = self.S_z(alpha)
        C_x = self.C_x(alpha)
        C_z = self.C_z(alpha)

        if self.Pos[1] > 122:
            self.flaps_factor = 1
        else:
            self.flaps_factor = 1.7

        drag = self.drag(S_x, V, C_x) * self.flaps_factor

        # Compute lift magnitude
        lift = self.drag(S_z, V, C_z) * self.flaps_factor

        # Newton's second law
        # Z-Axis
        # Project onto Z-axis
        lift_z = cos(self.theta) * lift
        drag_z = -sin(gamma) * drag
        thrust_z = sin(self.theta) * thrust
        # Compute the sum
        F_z = lift_z + drag_z + thrust_z - P

        # X-Axis
        # Project on X-axis
        lift_x = -sin(self.theta) * lift
        drag_x = -abs(cos(gamma) * drag)
        thrust_x = cos(self.theta) * thrust
        # Compute the sum
        F_x = lift_x + drag_x + thrust_x
        # Check if we are on the ground, if so prevent from going underground by setting  vertical position and vertical speed to 0.
        if self.task == "take-off":
            if self.check_colisions() and F_z <= 0:
                F_z = 0
                energy = 0.5 * self.m * self.V[1] ** 2
                if energy > self.critical_energy:
                    self.crashed = True
                self.V[1] = 0
                self.Pos[1] = 0

        # Compute Acceleration using a = F/m
        A = [F_x / self.m, F_z / self.m]

        # Append all the interesting values to their respective lists for monitoring
        self.lift_vec[0].append(lift_x)
        self.lift_vec[1].append(lift_z)
        self.P_vec.append(P)
        self.T_vec[1].append(thrust_z)
        self.T_vec[0].append(thrust_x)
        self.drag_vec[0].append(drag_x)
        self.drag_vec[1].append(drag_z)
        self.alpha_vec.append(np.degrees(alpha))
        self.gamma_vec.append(np.degrees(gamma))
        self.theta_vec.append(np.degrees(self.theta))
        self.S_vec[0].append(S_x)
        self.S_vec[1].append(S_z)
        self.C_vec[0].append(C_x)
        self.C_vec[1].append(C_z)

        return A

    def compute_dyna(self, thrust):
        """
        Compute the dynamcis : Acceleration, Speed and Position
        Speed(t+1) = Speed(t) + Acceleration(t) * Delta_t
        Position(t+1) = Position(t) + Speed(t) * Delta_t
        """
        # Update acceleration, speed and position
        self.A = self.compute_acceleration(thrust)
        self.V = [self.V[i] + self.A[i] * self.DELTA_T for i in range(2)]
        self.Mach = self.V[0] / 343
        # self.V[0] = 240
        self.Pos = [self.Pos[i] + self.V[i] * self.DELTA_T for i in range(2)]
        # if self.V[0] > self.MAX_SPEED:
        #     self.V[0] = self.MAX_SPEED

        # Update plot lists
        self.A_vec[0].append(self.A[0])
        self.A_vec[1].append(self.A[1])
        self.V_vec[0].append(self.V[0])
        self.V_vec[1].append(self.V[1])
        self.Pos_vec[0].append(self.Pos[0])
        self.Pos_vec[1].append(self.Pos[1])

    def altitude_factor(self):
        """
        Compute the reducting in reactor's power with rising altitude.
        """
        alt = self.Pos[1]
        a = 1 / (math.exp(alt / 7500))

        return max(0, min(1, a ** (0.7)))

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

        self.theta = np.radians(
            action["theta"] * 90
        )  # convert the pitch angle to radians
        thrust_modified = int(
            (action["thrust"][0]) * self.altitude_factor() * self.THRUST_MAX
        )  # Apply the atitude factor to the thrust
        # Compute the dynamics for the episode

        self.compute_dyna(thrust_modified)
        self.fuel_consumption()
        self.get_obs()
        self.timestep += 1

        return self.obs
