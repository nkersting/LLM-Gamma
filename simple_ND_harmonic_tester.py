#!/usr/bin/python

import numpy as np
from utils import random_vecs
from lo_dim_tester import LoDimTester

class SimpleNDHarmonicTester(LoDimTester):
    """
    Concrete PoC implementation of e.g., f(x1,x2,x3,x4,...) = x1^2 - x2^2 + x3^2 - x4^2 + ... tester
    Generally speaking should keep dimension of input point less than 500 to qualify as low-dimensional.
    """
    
    def __init__(self, model, radius, n_dim):
        super().__init__(model, radius, n_dim)

def simpleHarmonicFn(point):
    """
    f(x1,x2,x3,x4,...) = x1^2 - x2^2 + x3^2 - x4^2 + ... 
    """
    n = len(point)
    if n <= 1:
        print("not enough dimensions, casting to dim=0")
        n = 0
    elif n % 2 != 0:
        print("odd number dimensions, dropping last")
        n -= 1
    return np.sum([((-1)**i)*point[i]**2 for i in range(n)]) 

def simpleAnharmonicFn(point):
    """
    f(x1,x2,x3,x4,...) = x1^2 + x2^2 + x3^2 + x4^2 + ... 
    """    
    return np.sum([point[i]**2 for i in range(len(point))])

def main():
    
    # testing zero anharmoniticity
    print("Simple Harmonic 2-d Function:")
    currtester = SimpleNDHarmonicTester(simpleHarmonicFn, 1, 2)
    avg_anharm = 0
    numpoints = 100
    points = random_vecs(numpoints, 2)
    for point in points:
        avg_anharm += currtester.anharmoniticity(point)
    avg_anharm /= numpoints
    print("AVG_ANHARM=", avg_anharm)
    assert avg_anharm <= 1e-6

    
    print("\nSimple Harmonic 12-d Function:")
    currtester = SimpleNDHarmonicTester(simpleHarmonicFn, 1, 12)
    avg_anharm = 0
    numpoints = 100
    points = random_vecs(numpoints, 12)
    for point in points:
        avg_anharm += currtester.anharmoniticity(point)
    avg_anharm /= numpoints
    print("AVG_ANHARM=", avg_anharm)
    assert avg_anharm <= 1e-6

    
    print("\nSimple Anharmonic 2-d Function:")
    currtester = SimpleNDHarmonicTester(simpleAnharmonicFn, 1, 2)
    avg_anharm = 0
    numpoints = 100
    points = random_vecs(numpoints, 2)
    for point in points:
        avg_anharm += currtester.anharmoniticity(point)
    avg_anharm /= numpoints
    print("AVG_ANHARM=", avg_anharm)
    assert avg_anharm >= 0.5

    print("\nSimple Anharmonic 12-d Function:")
    currtester = SimpleNDHarmonicTester(simpleAnharmonicFn, 1, 12)
    avg_anharm = 0
    numpoints = 100
    points = random_vecs(numpoints, 12)
    for point in points:
        avg_anharm += currtester.anharmoniticity(point)
    avg_anharm /= numpoints
    print("AVG_ANHARM=", avg_anharm)
    assert avg_anharm >= 0.5

    
if __name__ == "__main__":
    main()

    
