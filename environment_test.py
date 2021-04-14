import unittest
import numpy as np
from environment import FlightModel


class TestEnvironment(unittest.TestCase):
    def test_compute_dyna(self):
        # Test if we get an error with a negative thrust
        self.assertRaises(
            ValueError,
            FlightModel().compute_dyna,
            -1,
            1,
            [0, 0],
            [0, 0],
            [0, 0],
            1000,
            1,
        )
        # Test if we get an error with a negative weight
        self.assertRaises(
            ValueError, FlightModel().compute_dyna, 0, 1, [0, 0], [0, 0], [0, 0], -1, 1
        )
        # Test if we get an error with a negative vertical speed
        self.assertRaises(
            ValueError,
            FlightModel().compute_dyna,
            0,
            1,
            [0, 0],
            [0, -1],
            [0, 0],
            1000,
            1,
        )
        # Test if we get an error with a negative altitude
        self.assertRaises(
            ValueError,
            FlightModel().compute_dyna,
            0,
            1,
            [0, 0],
            [0, 0],
            [0, -1],
            1000,
            1,
        )
        # Test if we get an error with a negative altitude factor
        self.assertRaises(
            ValueError,
            FlightModel().compute_dyna,
            0,
            1,
            [0, 0],
            [0, 0],
            [0, 0],
            1000,
            -1,
        )

    def test_compute_acceleration(self):
        # Test if we get an error for supersonic speed
        self.assertRaises(
            ValueError, FlightModel().compute_acceleration, 1, 343, 343, 1, 1, 1, 1
        )
        # Test if we get an error for negative or null mass
        self.assertRaises(
            ValueError, FlightModel().compute_acceleration, 1, 343, 343, 1, 0, 1, 1
        )
        self.assertRaises(
            ValueError, FlightModel().compute_acceleration, 1, 343, 343, 1, -1, 1, 1
        )
        # Test if we get an error for negative altitude
        self.assertRaises(
            ValueError, FlightModel().compute_acceleration, 1, 343, 343, 1, 1, -1, 1
        )
