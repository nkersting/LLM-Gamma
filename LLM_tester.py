#!/usr/bin/python

import numpy as np
from utils import sin_theta, cos_theta, normalized
from harmonic_tester import HarmonicTester, Point

class LLMTester(HarmonicTester):
    """                                                                                                                          
    Abstract child class specialized for models which take a string input and return string output,
    with an available embedding space to allow string -> vector mapping
    """
    def __init__(self, model, radius, embedding):
        """
        Args:
        embedding: function mapping strings to vectors
        """
        super().__init__(model, radius)
        self.embedding = embedding

    def average_model_value(self, points:list[Point]):
        """
        Computes the average embedding of the model outputs for each of the Points (strings)
        """
        model_vals = [self.model(x) for x in points]
        print(model_vals)
        vecs = [self.embedding(x) for x in model_vals]
        return np.mean(vecs, axis=0)
    
    def ball_center_compare(self, central_value, ball_avg_value):
        """
        Difference between central value and ball_avg taken as sine of angle
        Args:
        central_value: a natural string output
        ball_avg_value: average embedding of the ball outputs
        """
        print("Central Answer:", central_value)
        central_vector = self.embedding(central_value)
        return sin_theta(central_vector, ball_avg_value)

    def ball_isotropy(self, central_point:Point, ball_points:list[Point]):
        """
        For string points, we compute isotropy by measuring the cosine of the angle between the central_point's embedding and the
        average of the ball_points' embeddings.
        """
        vecs = [self.embedding(x) for x in ball_points]
        average_vec = np.mean(vecs, axis=0)
        average_vec_plus = average_vec + np.std(vecs, axis=0)
        #average_normed_vec = np.mean(normalized(vecs), axis=0)
        central_vec = self.embedding(central_point)
        cos_angles = [cos_theta(central_vec, v) for v in vecs]
        cos_avg = cos_theta(central_vec, average_vec)
        return np.mean(cos_angles), np.std(cos_angles)/np.sqrt(len(cos_angles)), cos_avg, abs(cos_avg - cos_theta(central_vec, average_vec_plus))
