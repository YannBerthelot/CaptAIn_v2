import os
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, sqrt
from aerodynamics import (
    compute_gamma,
    compute_Cz,
    compute_Cx,
    Mach_Cx,
    Mach_Cz,
    compute_fuel_variation,
    compute_drag,
    compute_altitude_factor,
    compute_Sx,
    compute_Sz,
    compute_next_position,
    compute_next_speed,
)
from configparser import ConfigParser

# create and configure parser
parser = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(thisfolder, "config.ini")
parser.read(config_path)
MACH_CRITIC = eval(parser.get("flight_model", "Mach_critic"))
C_X_MIN = eval(parser.get("flight_model", "C_x_min"))


class TestAerodynamic(unittest.TestCase):
    def test_compute_gamma(self):
        # Try different gammas
        self.assertAlmostEqual(compute_gamma(0, 0), 0)
        self.assertAlmostEqual(compute_gamma(0, 1), 0)
        self.assertAlmostEqual(compute_gamma(1, 1), pi / 2)
        self.assertAlmostEqual(compute_gamma(-1, 1), -pi / 2)
        self.assertAlmostEqual(compute_gamma(1, sqrt(2)), pi / 4)
        self.assertAlmostEqual(compute_gamma(-1, sqrt(2)), -pi / 4)
        # Check for error for supersonic speed
        self.assertRaises(ValueError, compute_gamma, 343, 343)
        # Check for error for Vz>V
        self.assertRaises(ValueError, compute_gamma, 1, 0)

    def test_compute_cz(self):
        # Test if we get the right Cz when stalling
        self.assertAlmostEqual(compute_Cz(np.radians(20), 0), 0)
        self.assertRaises(ValueError, compute_Cz, 0, 1)
        # Check that Cz is positive for theta >=-5 degree
        self.assertGreaterEqual(compute_Cz(np.radians(-5), 0), 0)
        self.assertGreaterEqual(compute_Cz(np.radians(0), 0), 0)
        self.assertGreaterEqual(compute_Cz(np.radians(5), 0), 0)
        # Check that Cz is negative for theta <=-5 degree
        self.assertLessEqual(compute_Cz(np.radians(-5), 0), 0)
        self.assertLessEqual(compute_Cz(np.radians(-10), 0), 0)

    def test_compute_cx(self):
        # Check supersonic speed
        self.assertRaises(ValueError, compute_Cx, 1, 2)

        # Check that Cx is minimal at -5 degrees
        self.assertAlmostEqual(compute_Cx(np.radians(-5), 0), C_X_MIN)
        # Check that Cx is never negative
        self.assertGreaterEqual(compute_Cx(np.radians(-5), 0), 0)
        self.assertGreaterEqual(compute_Cx(np.radians(-10), 0), 0)
        self.assertGreaterEqual(compute_Cx(np.radians(10), 0), 0)

    def test_mach_cx(self):
        # Check supersonic speed
        self.assertRaises(ValueError, Mach_Cx, 1, 2)

    def test_mach_cz(self):
        # Check error for supersonic speed
        self.assertRaises(ValueError, Mach_Cz, 1, 2)
        # Check that MAch C_z is never negative for positive input
        M_d = MACH_CRITIC + (1 - MACH_CRITIC) / 4
        self.assertGreaterEqual(Mach_Cz(0, 0), 0)
        self.assertGreaterEqual(Mach_Cz(0, 0.5), 0)
        self.assertGreaterEqual(Mach_Cz(0, MACH_CRITIC), 0)
        self.assertGreaterEqual(Mach_Cz(0, MACH_CRITIC + 0.01), 0)
        self.assertGreaterEqual(Mach_Cz(0, M_d), 0)
        self.assertGreaterEqual(Mach_Cz(0, M_d + 0.1), 0)

    def test_compute_fuel_variation(self):
        # Check supersonic speed
        self.assertRaises(ValueError, compute_fuel_variation, -1)
        self.assertGreater(compute_fuel_variation(1), 0)

    def test_compute_drag(self):
        # Check error for null or negative surface
        self.assertRaises(ValueError, compute_drag, 0, 1, 1, 1)
        self.assertRaises(ValueError, compute_drag, -1, 1, 1, 1)
        # Check error supersonic speed
        self.assertRaises(ValueError, compute_drag, 1, 343, 1, 1)
        # Check drag for nul speed is null
        self.assertAlmostEqual(compute_drag(1, 0, 1, 1), 0)
        # Check drag for positive speed and C is greater than 0
        self.assertGreater(compute_drag(1, 1, 1, 1), 0)
        # Check drag for positive speed and negative C is lesser than 0
        self.assertLess(compute_drag(1, 1, -1, 1), 0)

    def test_compute_altitude_factor(self):
        # Check error for negative altitude
        self.assertRaises(ValueError, compute_altitude_factor, -1)

    def test_compute_Sx(self):
        # Check that the surface is always greater than 0
        for val in [-1, 0, 1]:
            self.assertGreater(compute_Sx(val), 0)

    def test_compute_Sz(self):
        # Check that the surface is always greater than 0
        for val in [-1, 0, 1]:
            self.assertGreater(compute_Sz(val), 0)

    def test_compute_next_position(self):
        # Check error for negative altitude
        self.assertRaises(ValueError, compute_next_position, 0, -1, 0, 0)
        # Check no change when speed is null
        self.assertAlmostEqual(compute_next_position(0, 0, 0, 0), [0, 0])
        # Check for change when speed is not null
        self.assertNotEqual(compute_next_position(0, 0, 1, 0), [0, 0])
        self.assertNotEqual(compute_next_position(0, 0, 0, 1), [0, 0])

    def test_compute_next_speed(self):
        # Check no change when A is null
        self.assertAlmostEqual(compute_next_speed(0, 0, 0, 0), [0, 0])
        # Check for change when A is not null
        self.assertNotEqual(compute_next_speed(0, 0, 1, 0), [0, 0])
        self.assertNotEqual(compute_next_speed(0, 0, 0, 1), [0, 0])
