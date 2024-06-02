#!/usr/bin/python

from utils import simplex_plus_antisimplex, simplex_highd
import torch
import random
from vector_model_tester import VectorModelTester
from harmonic_tester import Point

class HiDimTester(VectorModelTester):
    """
    Concrete class for models with scalar outputs and vector inputs in high dimensions (dim >~ 500)
    Here the dimension is sufficiently high that simplex points converge to 1-hot vectors.
    """
    def __init__(self, model, radius, n_dim, sampling_fraction=1):
        super().__init__(model, radius)
        self.deltas = simplex_highd(n_dim, radius)
        self.sample_number = int(sampling_fraction * n_dim)
        
    def ball(self, point:Point, radius) -> list[Point]:
        total_points = [torch.add(torch.tensor(d, dtype=torch.float),
                                  torch.tensor(point, dtype=torch.float)) for d in self.deltas]
        return random.choices(total_points, k=self.sample_number)
