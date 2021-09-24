import os
import time
import logging
from configparser import ConfigParser
import numpy as np
from math import asin, cos, sin, sqrt
from utils import setup_logger, timing
from numba import njit, jit, prange

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
g = float(parser.get("flight_model", "g"))
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


@njit(nogil=True)
def compute_gamma(vz, norm_V):
    """
    Compute gamma (the angle between ground and the speed vector) using trigonometry.
    sin(gamma) = V_z / V -> gamma = arcsin(V_z/V)
    """
    # logger.debug(f"V_z : {V_z}, norm_V : {norm_V}")
    if norm_V > 0:
        return asin(vz / norm_V)
    if norm_V == 0:
        return 0


@njit(nogil=True)
def compute_cx(alpha, mach):
    """
    Compute the drag coefficient at M = 0 depending on alpha
    (the higher alpha the higher the drag)
    """
    # logger.debug(f"Compute Cx  in : alpha {alpha}, Mach {Mach}")
    cx = np.power((np.degrees(alpha) + 5) * 0.03, 2) + C_X_MIN
    # print("Alpha", alpha, "Mac", Mach, "Cx", C_x)
    if mach < MACH_CRITIC:
        return cx / np.sqrt(1 - (mach ** 2))
    if mach < 1:
        return cx * 15 * (mach - MACH_CRITIC) + (cx / np.sqrt(1 - (mach ** 2)))


@njit(nogil=True)
def compute_cz(alpha, mach):
    """
    Compute the lift coefficient at M=0 depending on alpha
    (the higher the alpha, the higher the lift until stall)
    """
    # logger.debug("alpha %s, Mach %s", np.degrees(alpha), Mach)
    # print("alpha", alpha)
    alpha = np.degrees(alpha) + 5
    sign = np.sign(alpha)
    alpha = sign * alpha
    # print('sign', sign)

    if alpha < 15:

        # Quadratic evolution  from C_z = 0 for 0 degrees and reaching a max value of C_z = 1.5 for 15 degrees
        cz = (alpha / 15) * C_Z_MAX
        # logger.debug(f"Compute Cz <15 {C_z}")
    elif alpha < 20:

        # Quadratic evolution  from C_z = 1.5 for 15 degrees to C_2 ~ 1.2 for 20 degrees.
        cz = (1 - ((alpha - 15) / 15)) * C_Z_MAX
        # logger.debug(f"Compute Cz <20 {C_z}")
    else:
        ##if alpha > 20 degrees : Stall => C_z = 0

        cz = 0
        # logger.debug(f"Compute Cz >=20 {C_z}")
    cz_min = cz / 2
    md = MACH_CRITIC + (1 - MACH_CRITIC) / 4
    if 0 <= mach <= MACH_CRITIC:
        return sign * cz
    # print("Alpha", alpha, "Mac", Mach, "Cz", C_z)
    if MACH_CRITIC < mach <= md:
        # logger.debug(f"cz {Mach}, Mach {cz + 0.1 * (mach - MACH_CRITIC)}")
        return sign * (cz + (0.2 / 0.18) * (mach - MACH_CRITIC))
    if mach < 1:
        maximal = cz + (0.2 / 0.18) * (md - MACH_CRITIC)
        # logger.debug(f"Cz {Mach}, Mach {maximal - 0.8 * (Mach - M_d)}")
        return sign * max(maximal - 0.8 * (mach - md), cz_min)


@njit(nogil=True)
def compute_fuel_variation(thrust):
    """
    Compute the fuel mass variation at each timestep based on the thrust.
    Update the remaining fuell mass and plane mass accordingly
    """
    return SFC * DELTA_T * thrust / 1000


@njit(nogil=True)
def compute_drag(S, V, C, altitude_factor):
    """
    Compute the drag using:
    S : Surface peripendicular to the drag direction
    V : Speed colinear with the drag direction
    C : Coefficient of drag/lift regarding the drag direction
    RHO : air density
    F = 1/2 * S * C * V^2
    """
    return 0.5 * RHO * altitude_factor * S * C * np.power(V, 2)


@njit(nogil=True)
def compute_altitude_factor(altitude):
    """
    Compute the reducting in reactor's power with rising altitude.
    """
    a = 1 / (np.exp(altitude / 7500))
    return max(0, min(1, a ** (0.7)))


@njit(nogil=True, cache=True)
def compute_sx(alpha):
    """
    update the value of the surface orthogonal to the speed vector
    depending on alpha by projecting the x and z surface.
    S_x = cos(alpha)*S_front + sin(alpha) * S_wings
    """
    # logger.debug(f"alpha:{alpha}")
    alpha = abs(alpha)
    return cos(alpha) * S_FRONT + sin(alpha) * S_WINGS


@njit(nogil=True)
def compute_sz(alpha):
    """
    update the value of the surface colinear to the speed vector depending on alpha by projecting the x and z surface.
    S_x = sin(alpha)*S_front + cos(alpha) * S_wings
    !IMPORTANT!
    The min allows the function to be stable, I don't understand why yet.
    """
    # logger.debug(f"alpha:{alpha}")
    alpha = abs(alpha)
    return (sin(alpha) * S_FRONT) + (cos(alpha) * S_WINGS)


@njit(nogil=True)
def next_speed_and_pos(A, V, Pos):
    V = V + A * DELTA_T
    Pos = Pos + V * DELTA_T
    return Pos, V


@njit(nogil=True, fastmath=True)
def norm(l):
    return np.linalg.norm(l)
    # s = 0.0
    # for i in range(l.shape[0]):
    #     s += l[i] ** 2
    # return np.sqrt(s)


@njit(nogil=True)
def compute_mach(V):
    norm_v = norm(np.array([V[0], V[1]], dtype=np.float64))
    return norm_v / 343, norm_v


def compute_dyna(thrust, theta, A, V, Pos, m, altitude_factor):
    """
    Compute the dynamics : Acceleration, Speed and Position
    Speed(t+1) = Speed(t) + Acceleration(t) * Delta_t
    Position(t+1) = Position(t) + Speed(t) * Delta_t
    """
    times = []
    start_time = time.process_time()
    # Compute the magnitude of the speed vector
    start_mach = time.process_time()
    mach, norm_v = compute_mach(V)
    mach_time = time.process_time() - start_mach

    start_gamma = time.process_time()
    # Compute gamma based on speed
    gamma = compute_gamma(V[1], norm_v)
    alpha = theta - gamma
    gamma_time = time.process_time() - start_gamma

    start_P = time.process_time()
    # Compute P
    P = m * g
    P_time = time.process_time() - start_P

    # cmpute surfaces and coefficients for drag
    start_sx = time.process_time()
    sx = compute_sx(alpha)
    sx_time = time.process_time() - start_sx

    start_sz = time.process_time()
    sz = compute_sz(alpha)
    sz_time = time.process_time() - start_sz

    start_cx = time.process_time()
    cx = compute_cx(alpha, mach)
    cx_time = time.process_time() - start_cx

    start_cz = time.process_time()
    cz = compute_cz(alpha, mach)
    cz_time = time.process_time() - start_cz

    # simulate flaps
    start_flaps = time.process_time()
    if Pos[1] < 122:
        cz *= 1.5
        cx *= 1.5
    flaps_time = time.process_time() - start_flaps

    # Compute Drag and lift magnitude
    start_drag = time.process_time()
    drag = compute_drag(sx, norm_v, cx, altitude_factor)
    lift = compute_drag(sz, norm_v, cz, altitude_factor)
    drag_time = time.process_time() - start_drag

    # Newton's second law
    start_newton = time.process_time()
    A = newton(theta, gamma, thrust, lift, drag, P, m)
    newton_time = time.process_time() - start_newton

    # compute the new speed and pos
    start_new_pos_V = time.process_time()
    Pos, V = next_speed_and_pos(A, V, Pos)
    new_pos_V_time = time.process_time() - start_new_pos_V

    start_crashed = time.process_time()
    Pos, V, A, crashed = crash(Pos, m, V, A)
    crashed_time = time.process_time() - start_crashed

    total_time = time.process_time() - start_time
    times = [
        total_time,
        mach_time,
        gamma_time,
        P_time,
        sx_time,
        sz_time,
        cx_time,
        cz_time,
        flaps_time,
        drag_time,
        newton_time,
        new_pos_V_time,
        crashed_time,
    ]
    return A, V, Pos, lift, crashed, times


@njit(nogil=True)
def crash(Pos, m, V, A):
    crashed = False
    if Pos[1] < 0:
        energy = 0.5 * m * V[1] ** 2
        crashed = energy > CRITICAL_ENERGY
        Pos[1] = 0
        V[1] = 0
        # make the plane bounce when touching the ground
        A[1] = abs(A[1] * 0.8)
    return Pos, V, A, crashed


@njit(nogil=True)
def newton(theta, gamma, thrust, lift, drag, P, m):
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    lift_drag_thrust = np.array([lift, drag, thrust])
    # Z-axis
    F_z = np.sum(lift_drag_thrust * np.array([cos_theta, -sin(gamma), sin_theta])) - P
    # X-axis
    F_x = np.sum(lift_drag_thrust * np.array([-sin_theta, -abs(cos(gamma)), cos_theta]))
    # Compute Acceleration using a = F/m
    return np.array([F_x / m, F_z / m])


if __name__ == "__main__":
    n_loops = int(1e1)

    # # gamma
    # start = time.process_time()
    # print("compute_gamma")
    # for i in range(n_loops):
    #     compute_gamma(1, 2)
    # print(time.process_time() - start)

    # n_f = njit(nogil=True)(compute_gamma)
    # start = time.process_time()
    # print("compute_gamma numba")
    # for i in range(n_loops):
    #     n_f(1, 2)
    # print(time.process_time() - start)

    # start = time.process_time()
    # print("compute_gamma C++")
    # for i in range(n_loops):
    #     c_compute_gamma(1, 2)
    # print(time.process_time() - start)

    # # cx
    # start = time.process_time()
    # print("compute_cx")
    # for i in range(n_loops):
    #     compute_Cx(1, 0.7)
    # print(time.process_time() - start)

    # n_f = njit(nogil=True)(compute_Cx)
    # start = time.process_time()
    # print("compute_Cx numba")
    # for i in range(n_loops):
    #     n_f(1, 2)
    # print(time.process_time() - start)

    # start = time.process_time()
    # print("compute_cx C++")
    # for i in range(n_loops):
    #     c_compute_cx(1, 0.7)
    # print(time.process_time() - start)

    # # cz
    # start = time.process_time()
    # print("compute_cz")
    # for i in range(n_loops):
    #     compute_cz(1, 0.7)
    # print(time.process_time() - start)

    # n_f = njit(nogil=True)(compute_cz)
    # start = time.process_time()
    # print("compute_cz numba")
    # for i in range(n_loops):
    #     n_f(1, 2)
    # print(time.process_time() - start)

    # start = time.process_time()
    # print("compute_cz C++")
    # for i in range(n_loops):
    #     c_compute_cz(1, 0.7)
    # print(time.process_time() - start)

    # # fuel consumption
    # start = time.process_time()
    # print("compute_fuel_variation")
    # for i in range(n_loops):
    #     compute_fuel_variation(1000)
    # print(time.process_time() - start)

    # n_f = njit(nogil=True)(compute_fuel_variation)
    # start = time.process_time()
    # print("compute_fuel_variation numba")
    # for i in range(n_loops):
    #     n_f(1000)
    # print(time.process_time() - start)

    # start = time.process_time()
    # print("compute_fuel_variation C++")
    # for i in range(n_loops):
    #     c_compute_fuel_variation(1000)
    # print(time.process_time() - start)

    # # compute_drag
    # start = time.process_time()
    # print("compute_drag")
    # for i in range(n_loops):
    #     compute_drag(100, 250, 0.5, 0.9)
    # print(time.process_time() - start)

    # n_f = njit(nogil=True)(compute_drag)
    # start = time.process_time()
    # print("compute_drag numba")
    # for i in range(n_loops):
    #     n_f(100, 250, 0.5, 0.9)
    # print(time.process_time() - start)

    # start = time.process_time()
    # print("compute_drag C++")
    # for i in range(n_loops):
    #     c_compute_drag(100, 250, 0.5, 0.9)
    # print(time.process_time() - start)

    # # compute_altitude_factor
    # start = time.process_time()
    # print("compute_altitude_factor")
    # for i in range(n_loops):
    #     compute_altitude_factor(5000)
    # print(time.process_time() - start)

    # n_f = njit(nogil=True)(compute_altitude_factor)
    # start = time.process_time()
    # print("compute_altitude_factor numba")
    # for i in range(n_loops):
    #     n_f(5000)
    # print(time.process_time() - start)

    # start = time.process_time()
    # print("compute_altitude_factor C++")
    # for i in range(n_loops):
    #     c_compute_altitude_factor(5000)
    # print(time.process_time() - start)

    # # compute_sx
    # start = time.process_time()
    # print("compute_sx")
    # for i in range(n_loops):
    #     compute_sx(0.3)
    # print(time.process_time() - start)

    # n_f = njit(nogil=True)(compute_sx)
    # start = time.process_time()
    # print("compute_sx numba")
    # for i in range(n_loops):
    #     n_f(0.3)
    # print(time.process_time() - start)

    # start = time.process_time()
    # print("compute_sx C++")
    # for i in range(n_loops):
    #     c_compute_sx(0.3)
    # print(time.process_time() - start)

    # # compute_sz
    # start = time.process_time()
    # print("compute_sz")
    # for i in range(n_loops):
    #     compute_sx(0.3)
    # print(time.process_time() - start)

    # n_f = njit(nogil=True)(compute_sz)
    # start = time.process_time()
    # print("compute_sz numba")
    # for i in range(n_loops):
    #     n_f(0.3)
    # print(time.process_time() - start)

    # start = time.process_time()
    # print("compute_sz C++")
    # for i in range(n_loops):
    #     c_compute_sz(0.3)
    # print(time.process_time() - start)

    # # compute_next_position
    # start = time.process_time()
    # print("compute_next_position")
    # for i in range(n_loops):
    #     compute_next_position(0, 0, 10, 15)
    # print(time.process_time() - start)

    # n_f = njit(nogil=True)(compute_next_position)
    # start = time.process_time()
    # print("compute_next_position numba")
    # for i in range(n_loops):
    #     n_f(0, 0, 10, 15)
    # print(time.process_time() - start)

    # start = time.process_time()
    # print("compute_next_position C++")
    # for i in range(n_loops):
    #     compute_next_position(0, 0, 10, 15)
    # print(time.process_time() - start)

    # compute_acceleration
    start = time.process_time()
    print("compute_next_position")
    for i in range(n_loops):
        compute_acceleration(150000.0, 10.0, 0.0, 0.2, 65000.0, 3000.0, 0.9)
    print(time.process_time() - start)

    n_f = njit(nogil=True)(compute_acceleration)
    start = time.process_time()
    print("compute_next_position numba")
    for i in range(n_loops):
        n_f(150000.0, 10.0, 0.0, 0.2, 65000.0, 3000.0, 0.9)
    print(time.process_time() - start)

    # compute_dyna
    # start = time.process_time()
    # print("compute_dyna")
    # for i in range(n_loops):
    #     compute_dyna(12000, 0.2, [1, 1], [12, 45], [120, 2], 55000, 0.9)
    # print(time.process_time() - start)

    # n_f = njit(nogil=True)(compute_dyna)
    # start = time.process_time()
    # print("compute_dyna numba")
    # for i in range(n_loops):
    #     n_f(
    #         12000.0,
    #         0.2,
    #         np.array([1.0, 1.0]),
    #         np.array([12.0, 45.0]),
    #         np.array([120.0, 2.0]),
    #         55000.0,
    #         0.9,
    #     )
    # print(time.process_time() - start)
