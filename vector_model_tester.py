#!/usr/bin/python

import numpy as np
from harmonic_tester import HarmonicTester, Point


class VectorModelTester(HarmonicTester):
    """
    Abstract child class specialized for models which take a vector input and return scalar output
    """
    def __init__(self, model, radius):
        super().__init__(model, radius)

    def average_model_value(self, points:list[Point]):
        """
        Computes a suitable average model value over the given points
        """
        return np.mean(np.array([self.model(p) for p in points]))

        
    def ball_center_compare(self, central_value, ball_avg_value):
        """
        Computes the disparity between a central value and the average value on a surrounding ball as absolute difference
        """
        return abs(central_value - ball_avg_value)

    
    
