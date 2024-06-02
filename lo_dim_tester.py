#!/usr/bin/python

from utils import simplex_plus_antisimplex
import torch
from vector_model_tester import VectorModelTester
from harmonic_tester import Point

class LoDimTester(VectorModelTester):
    """
    Concrete class for models with scalar outputs and vector inputs in low dimensions (dim <~ 500).
    Here we need to compute the actual simplex coordinates as they are non-trivial.
    """
    def __init__(self, model, radius, n_dim):
        super().__init__(model, radius)
        self.deltas = simplex_plus_antisimplex(n_dim, radius)

    def ball(self, point:Point, radius) -> list[Point]:
        return [torch.add(torch.tensor(d, dtype=torch.float),
                          torch.tensor(point, dtype=torch.float)) for d in self.deltas]


    
    
