#!/usr/bin/python
import numpy as np

class Point():
    """
    Abstract class representing input to a model, e.g., vector of features for a decision tree, or string for a LLM
    """
    
    def __init__(self, raw_input):
        self.value = raw_input
    

class HarmonicTester():
    """
    Abstract class for computing anharmoniticity for a given model (function)
    """

    def __init__(self, model, radius):
        """
        Args:
        model: function representing the model to test, e.g. decision tree, NN, or LLM; takes Point input
        radius: distance from central point to compute ball
        """
        
        self.model = model
        self.radius = radius

    def ball(self, point:Point, radius) -> list[Point]:
        """
        Computes points on the ball of given radius surrounding the given point
        """
        pass

    def ball_isotropy(self, central_point:Point, ball_points:list[Point]):
        """
        Diagnostic function.
        Compute the extent to which the ball points are isotropically distributed about the central point.
        """
        pass
        
    def average_model_value(self, points:list[Point]):
        """
        Computes a suitable average model value over the given points
        """
        pass
    
    def average_ball_value(self, point:Point):
        """
        Computes the average value of the model on a ball surrounding the given point
        """
        ball_points = self.ball(point, self.radius)
        return self.average_model_value(ball_points)

    def ball_center_compare(self, central_value, ball_avg_value):
        """
        Computes the disparity between a central value and the average value on a surrounding ball
        """
        pass
        
    
    def anharmoniticity(self, point: Point) -> float:
        """
        Computes anharmoniticity at the input point by evaluating model on a ball around this point and taking average 
        """
        central_value = self.model(point)
        ball_avg_value = self.average_ball_value(point)
        return self.ball_center_compare(central_value, ball_avg_value)
    

    def follow_anharmonic_gradient(self, central_point: Point, upgrad=True) -> (Point, float):
        """
        Returns the Point on the ball surrounding input point with the greatest (lowest if upgrad is False) anharmoniticity.
        Called repeatedly, this can be used to converge on high(adversarial) or low(stable) points

        Args:
        central_point: compute ball around this point
        upgrad: if True(False), will look for point on ball with greatest(smallest) anharmoniticity

        Returns:
        tuple of (extremal point on ball, anharmoniticity)
        """

        ball_points = self.ball(central_point, self.radius)
        anharms = [self.anharmoniticity(p) for p in ball_points]
        if upgrad == True:
            return (ball_points[np.argmax(anharms)], max(anharms))
        else:
            return (ball_points[np.argmin(anharms)], min(anharms))
            
