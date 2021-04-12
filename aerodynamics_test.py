import unittest
import numpy as np
from aerodynamics import (
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
)


class TestAerodynamic(unittest.TestCase):
    def test_cz(self):
        # Test if we get the right Cz when stalling
        self.assertAlmostEqual(compute_Cz(np.radians(20), 0), 0)
        self.assertRaises(ValueError, compute_Cz, 0, 1)

    def test_mach_cx(self):
        # Check supersonic speed
        self.assertRaises(ValueError, Mach_Cx, 1, 2)

    def test_mach_cz(self):
        # Check supersonic speed
        self.assertRaises(ValueError, Mach_Cz, 1, 2)

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
        # Check drag for positive speed is greater than 0
        self.assertGreater(compute_drag(1, 1, 1, 1), 0)

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
