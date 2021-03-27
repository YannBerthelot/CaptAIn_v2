import numpy as np
from configparser import ConfigParser
from functools import cache

parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)

MACH_CRITIC = parser.get("flight_model", "Mach_critic")
PRECISION = parser.get("flight_model", "Precision")


@cache
def gamma(V):
    """
    Compute gamma (the angle between ground and the speed vector) using trigonometry.
    sin(gamma) = V_z / V -> gamma = arcsin(V_z/V)
    """
    norm_V = norm(V)
    if norm_V > 0:
        return np.round(arcsin(V[1] / norm_V), PRECISION)
    else:
        return 0


@cache
def Mach_Cx(Cx, Mach):
    """
    Compute the drag coefficient based on Mach Number and drag coefficient at M =0
    """
    if Mach < MACH_CRITIC:
        return np.round(Cx / math.sqrt(1 - (Mach ** 2)), PRECISION)
    else:
        return np.round(
            Cx * 15 * (Mach - MACH_CRITIC) + Cx / math.sqrt(1 - (MACH_CRITIC ** 2)),
            PRECISION,
        )


@cache
def Mach_Cz(Cz, Mach):
    """
    Compute the lift coefficient based on Mach Number and lift coefficient at M =0
    """
    M_d = MACH_CRITIC + (1 - MACH_CRITIC) / 4
    if Mach <= MACH_CRITIC:
        return np.round(Cz, PRECISION)
    elif Mach <= M_d:
        return np.round(Cz + 0.1 * (Mach - MACH_CRITIC), PRECISION)
    else:
        maximal = Cz + 0.1 * (M_d - self.MACH_CRITIC)
        return maximal - 0.8 * (Mach - M_d)
