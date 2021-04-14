import os
import logging
from configparser import ConfigParser


import numpy as np
from numpy import arcsin, cos, sin
from utils import setup_logger


# create and configure parser
parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)

MACH_CRITIC = float(parser.get("flight_model", "Mach_critic"))
PRECISION = int(parser.get("flight_model", "Precision"))
CACHE_SIZE = int(1e10)
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
DEBUG = bool(parser.get("debug", "debug"))

# create and configure logger
os.makedirs("logs", exist_ok=True)
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOG_FORMAT = "%(levelno)s %(asctime)s %(funcName)s - %(message)s"
logger = setup_logger(
    "aerodynamics_logger",
    "logs/aerodynamics.log",
    level=level,
    format=LOG_FORMAT,
)


def compute_alpha(theta, gamma):
    """
    Compute alpha (the angle between the plane's axis and the speed vector).
    alpha = theta - gamma
    """
    logger.debug(f"theta {theta},gamma {gamma}")
    # print("theta", np.degrees(theta), "gamma", np.degrees(gamma))
    return theta - gamma


def compute_gamma(V_z, norm_V):
    """
    Compute gamma (the angle between ground and the speed vector) using trigonometry.
    sin(gamma) = V_z / V -> gamma = arcsin(V_z/V)
    """
    logger.debug(f"V_z : {V_z}, norm_V : {norm_V}")
    if (norm_V >= 343) or (V_z >= 343):
        err = ValueError(f"Supersonic speed {norm_V/343}")
        logger.error(err)
        raise err
    if V_z > norm_V:
        err = ValueError("V_z higher than V")
        logger.error(err)
        raise err
    if norm_V > 0:
        return arcsin(V_z / norm_V)
    if norm_V == 0:
        return 0
    else:
        err = ValueError("Negative norm")
        logger.error(err)
        raise err


def compute_Cx(alpha, Mach):
    """
    Compute the drag coefficient at M = 0 depending on alpha
    (the higher alpha the higher the drag)
    """
    logger.debug(f"Compute Cx  in : alpha {alpha}, Mach {Mach}")
    alpha += np.radians(5)
    C_x = np.power(np.degrees(alpha) * 0.02, 2) + C_X_MIN
    # print("Alpha", alpha, "Mac", Mach, "Cx", C_x)
    C_x = Mach_Cx(C_x, Mach)
    logger.debug(f"Compute Cx out : {C_x}")

    return C_x


def compute_Cz(alpha, Mach):
    """
    Compute the lift coefficient at M=0 depending on alpha
    (the higher the alpha, the higher the lift until stall)
    """
    logger.debug("alpha %s, Mach %s", np.degrees(alpha), Mach)
    alpha = alpha + np.radians(5)
    # print("alpha", alpha)
    alpha_degrees = np.degrees(alpha)
    sign = np.sign(alpha_degrees)
    alpha_degrees = abs(alpha_degrees)
    # print('sign', sign)

    if alpha_degrees < 15:

        # Quadratic evolution  from C_z = 0 for 0 degrees and reaching a max value of C_z = 1.5 for 15 degrees
        C_z = (alpha_degrees / 15) * C_Z_MAX
        logger.debug(f"Compute Cz <15 {C_z}")
    elif alpha_degrees < 20:

        # Quadratic evolution  from C_z = 1.5 for 15 degrees to C_2 ~ 1.2 for 20 degrees.
        C_z = (1 - ((alpha_degrees - 15) / 15)) * C_Z_MAX
        logger.debug(f"Compute Cz <20 {C_z}")
    else:
        ##if alpha > 20 degrees : Stall => C_z = 0

        C_z = 0
        logger.debug(f"Compute Cz >=20 {C_z}")

    C_z = sign * Mach_Cz(C_z, Mach)
    # print("Alpha", alpha, "Mac", Mach, "Cz", C_z)
    logger.debug(f"Compute Cz out : {C_z}")
    return C_z


def Mach_Cx(Cx, Mach):
    """
    Compute the drag coefficient based on Mach Number and drag coefficient at M =0
    """
    logger.debug(f"Cx {Cx}, Mach {Mach}")
    if Mach < MACH_CRITIC:
        return Cx / np.sqrt(1 - (Mach ** 2))
    if Mach < 1:
        return Cx * 15 * (Mach - MACH_CRITIC) + Cx / np.sqrt(1 - (MACH_CRITIC ** 2))
    else:
        err = ValueError(f"Supersonic speed {Mach}")
        logger.error(err)
        raise err


def Mach_Cz(Cz, Mach):
    """
    Compute the lift coefficient based on Mach Number and lift coefficient at M =0
    """
    Cz_min = Cz / 2
    M_d = MACH_CRITIC + (1 - MACH_CRITIC) / 4
    logger.debug(f"Cz {Cz}, Mach {Mach}, Mach critic {MACH_CRITIC}, M_d {M_d}")
    if 0 <= Mach <= MACH_CRITIC:
        return Cz
    if MACH_CRITIC < Mach <= M_d:
        logger.debug(f"Cz {Mach}, Mach {Cz + 0.1 * (Mach - MACH_CRITIC)}")
        return Cz + (0.2 / 0.18) * (Mach - MACH_CRITIC)
    if Mach < 1:
        maximal = Cz + (0.2 / 0.18) * (M_d - MACH_CRITIC)
        logger.debug(f"Cz {Mach}, Mach {maximal - 0.8 * (Mach - M_d)}")
        return max(maximal - 0.8 * (Mach - M_d), Cz_min)
    else:
        err = ValueError(f"Supersonic speed {Mach}")
        logger.error(err)
        raise err


def compute_fuel_variation(thrust):
    """
    Compute the fuel mass variation at each timestep based on the thrust.
    Update the remaining fuell mass and plane mass accordingly
    """
    logger.debug(f"thrust {thrust}")
    if thrust < 0:
        err = ValueError("Negative thrust")
        logger.error(err)
        raise err

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
    logger.debug(f"S:{S},V:{V},C:{C},altitude_factor:{altitude_factor}")
    if V >= 343:
        err = ValueError(f"Supersonic speed {V/343}")
        logger.error(err)
        raise err
    if S <= 0:
        err = ValueError(f"Negative or zero surface :{S}")
        logger.error(err)
        raise err
    drag = 0.5 * RHO * altitude_factor * S * C * np.power(V, 2)
    if (C > 0) & (drag < 0):
        err = ValueError(f"Negative drag for posiive C = {C},{drag}")
        logger.error(err)
        raise err
    return drag


def compute_altitude_factor(altitude):
    """
    Compute the reducting in reactor's power with rising altitude.
    """
    logger.debug(f"altitude:{altitude}")
    if altitude < 0:
        err = ValueError(f"Negative altitude :{altitude}")
        logger.error(err)
        raise err
    a = 1 / (np.exp(altitude / 7500))
    return max(0, min(1, a ** (0.7)))


def compute_Sx(alpha):
    """
    update the value of the surface orthogonal to the speed vector
    depending on alpha by projecting the x and z surface.
    S_x = cos(alpha)*S_front + sin(alpha) * S_wings
    """
    logger.debug(f"alpha:{alpha}")
    alpha = abs(alpha)
    return cos(alpha) * S_FRONT + sin(alpha) * S_WINGS


def compute_Sz(alpha):
    """
    update the value of the surface colinear to the speed vector depending on alpha by projecting the x and z surface.
    S_x = sin(alpha)*S_front + cos(alpha) * S_wings
    !IMPORTANT!
    The min allows the function to be stable, I don't understand why yet.
    """
    logger.debug(f"alpha:{alpha}")
    alpha = abs(alpha)
    return (sin(alpha) * S_FRONT) + (cos(alpha) * S_WINGS)


def compute_next_position(position, altitude, V_x, V_z):
    logger.debug(f"position:{position},altitude:{altitude},V_x:{V_x},V_z:{V_z}")
    if altitude < 0:
        err = ValueError(f"Negative altitude :{altitude}")
        logger.error(err)
        raise err

    position += V_x * DELTA_T
    altitude += V_z * DELTA_T
    return [position, altitude]


def compute_next_speed(V_x, V_z, A_x, A_z):
    logger.debug(f"V_x:{V_x},V_z:{V_z},A_x:{A_x},A_z:{A_z}")

    V_x += A_x * DELTA_T
    V_z += A_z * DELTA_T
    return [V_x, V_z]


if __name__ == "__main__":
    print("aerodynamics")
