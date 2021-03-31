import os
import numpy as np
from configparser import ConfigParser
from functools import lru_cache
from numpy import arcsin, cos, sin
from numpy.linalg import norm
from utils import timing

parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)

MACH_CRITIC = eval(parser.get("flight_model", "Mach_critic"))
PRECISION = eval(parser.get("flight_model", "Precision"))
CACHE_SIZE = int(1e10)
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


def compute_alpha(theta, gamma):
    """
    Compute alpha (the angle between the plane's axis and the speed vector).
    alpha = theta - gamma
    """
    # print("theta", np.degrees(theta), "gamma", np.degrees(gamma))
    return theta - gamma


def compute_gamma(V_z, norm_V):
    """
    Compute gamma (the angle between ground and the speed vector) using trigonometry.
    sin(gamma) = V_z / V -> gamma = arcsin(V_z/V)
    """

    if norm_V > 0:
        return arcsin(V_z / norm_V)
    else:
        return 0


def compute_Cx(alpha, Mach):
    """
    Compute the drag coefficient at M = 0 depending on alpha (the higher alpha the higher the drag)
    """
    alpha += np.radians(0)
    C_x = np.power(np.degrees(alpha) * 0.02, 2) + C_X_MIN
    # print("Alpha", alpha, "Mac", Mach, "Cx", C_x)
    return Mach_Cx(C_x, Mach)


def compute_Cz(alpha, Mach):
    """
    Compute the lift coefficient at M=0 depending on alpha (the higher the alpha, the higher the lift until stall)
    """
    alpha = alpha + np.radians(5)
    # print("alpha", alpha)
    alpha_degrees = np.degrees(alpha)
    sign = np.sign(alpha_degrees)
    # print('sign', sign)

    if abs(alpha_degrees) < 15:
        # Quadratic evolution  from C_z = 0 for 0 degrees and reaching a max value of C_z = 1.5 for 15 degrees
        C_z = sign * abs((alpha_degrees / 15) * C_Z_MAX)
    elif abs(alpha_degrees) < 20:
        # Quadratic evolution  from C_z = 1.5 for 15 degrees to C_2 ~ 1.2 for 20 degrees.
        C_z = sign * abs((1 - ((abs(alpha_degrees) - 15) / 15)) * C_Z_MAX)
    else:
        ##if alpha > 20 degrees : Stall => C_z = 0
        C_z = 0
    C_z = Mach_Cz(C_z, Mach)
    # print("Alpha", alpha, "Mac", Mach, "Cz", C_z)
    return C_z


def Mach_Cx(Cx, Mach):
    """
    Compute the drag coefficient based on Mach Number and drag coefficient at M =0
    """
    if Mach < MACH_CRITIC:
        return Cx / np.sqrt(1 - (Mach ** 2))
    else:
        return Cx * 15 * (Mach - MACH_CRITIC) + Cx / np.sqrt(1 - (MACH_CRITIC ** 2))


def Mach_Cz(Cz, Mach):
    """
    Compute the lift coefficient based on Mach Number and lift coefficient at M =0
    """
    M_d = MACH_CRITIC + (1 - MACH_CRITIC) / 4
    if Mach <= MACH_CRITIC:
        return Cz
    elif Mach <= M_d:
        return Cz + 0.1 * (Mach - MACH_CRITIC)
    else:
        maximal = Cz + 0.1 * (M_d - MACH_CRITIC)
        return maximal - 0.8 * (Mach - M_d)


def compute_fuel_variation(thrust):
    """
    Compute the fuel mass variation at each timestep based on the thrust.
    Update the remaining fuell mass and plane mass accordingly
    """
    return SFC * DELTA_T * thrust / 1000


def compute_drag(S, V, C, altitude_factor):
    """
    Compute the drag using:
    S : Surface peripendicular to the drag direction
    V : Speed colinear with the drag direction
    C : Coefficient of drag/lift regarding the drag direction
    RHO : air density
    F = 1/2 * S * C * V^2
    """
    # print(
    #     "COMPUTE DRAG", "S", S, "V", V, "C", C, "altfact", altitude_factor, "rho", RHO
    # )
    # print("RHO", RHO, "alt fact", altitude_factor, "S", S, "C", C, "V", V)
    return 0.5 * RHO * altitude_factor * S * C * np.power(V, 2)


def compute_altitude_factor(altitude):
    """
    Compute the reducting in reactor's power with rising altitude.
    """

    a = 1 / (np.exp(altitude / 7500))

    return max(0, min(1, a ** (0.7)))


def compute_Sx(alpha):
    """
    update the value of the surface orthogonal to the speed vector depending on alpha by projecting the x and z surface.
    S_x = cos(alpha)*S_front + sin(alpha) * S_wings
    """
    alpha = abs(alpha)
    return cos(alpha) * S_FRONT + sin(alpha) * S_WINGS


def compute_Sz(alpha):
    """
    update the value of the surface colinear to the speed vector depending on alpha by projecting the x and z surface.
    S_x = sin(alpha)*S_front + cos(alpha) * S_wings
    !IMPORTANT!
    The min allows the function to be stable, I don't understand why yet.
    """
    alpha = abs(alpha)
    return (sin(alpha) * S_FRONT) + (cos(alpha) * S_WINGS)


def compute_next_position(position, altitude, V_x, V_z):
    position += V_x * DELTA_T
    altitude += V_z * DELTA_T
    return [position, altitude]
