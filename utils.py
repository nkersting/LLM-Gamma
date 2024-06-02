#!/usr/bin/python


import numpy as np
import random

def magnitude(vector): 
    return np.sqrt(sum(pow(element, 2) for element in vector))

def sin_theta(v1, v2):
    return np.sqrt(1 - cos_theta(v1,v2))

def cos_theta(v1,v2):
    return np.dot(v1,v2)/magnitude(v1)/magnitude(v2)

def random_vecs(num_points, dim):
    """
    Generates random vectors of given dimension.
    Each component is between -1 and 1.

    Args:
    num_points: number of vectors to generate
    dim: dimension

    Returns:
    list of random vectors
    """
    r_vecs = []
    for _ in range(num_points):
        rawvec = normalize(np.random.rand(dim))
        for i in range(dim):
            if random.random() >= 0.5:
                rawvec[i] = -rawvec[i]
        r_vecs.append(rawvec)
    return r_vecs

def find_orig_coords(n):
    """
    Finds all the 1-hot vectors in n-dimensions, e.g., for n=2, gives [1,0] and [0,1]

    Args:
    n: desired dimension

    Returns:
    All 1-hot vectors with dim=n
    """
    coords = []
    for i in range(n):
        curr_point = [0]*n
        curr_point[i] = 1
        coords.append(np.array(curr_point))
    return coords

def center_origin(coords):
    """
    Shifts given 1-hot vectors so that they are centered on the origin

    Args:
    coords: collection of 1-hot vectors spanning input space

    Returns:
    original vectors shifted by same amount so they average to the origin
    """
    n = len(coords)
    for i in range(n):
        coords[i] = [coords[i][j] - [x/n for x in [1]*n][j] for j in range(n)]
    return coords


def rotate_away_last_dim(orig):
    """
    Performs a rotation on given vectors into the plane perpendicular to the last dimension, i.e. so as to transform the last dimension of all vectors to zero. We make use of a known formula which computes the normal to the plane formed by the vectors, rotating by the angle formed by this normal and the unit vector in the last dimension (https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm).

    Args:
    orig: collection of vectors

    Returns:
    original collection rotated so last dimension is 0
    """
    
    rotated = []
    n = len(orig)

    cos_theta = 1/np.sqrt(n)

    m1 = np.zeros((n,n))
    for i in range(n):
        m1[i][n-1] = 1
        m1[n-1][i] = -1
    m1[n-1][n-1] = 0

    m2 = np.ones((n,n))
    for i in range(n):
        m2[i][n-1] = 0
        m2[n-1][i] = 0
    m2[n-1][n-1] = n-1
    
    rotation = np.identity(n) - 1/np.sqrt(n) * m1 + (cos_theta - 1)/(n-1) * m2

    rotated = []
    for v in orig:
        rotated.append(np.matmul(rotation,v))

    return rotated


def normalized(vecs, norm=1):
    """
    Computes normalization of collection of vectors to have given norm

    Args:
    vecs: list of vectors
    norm: desired norm

    Returns:
    list of vectors normalized to given norm
    """
    return [normalize(v, norm) for v in vecs]

def normalize(v, mag=1):
    """
    Normalizes a vector to have given magnitude

    Args:
    v: vector to normalize
    mag: desired final magnitude

    Returns:
    vector normalized to given magnitude
    """
    norm = np.linalg.norm(v) / mag
    if norm == 0: 
       return v
    return v / norm
        

def simplex_points(n_dim, mag=1):
    """
    Finds the simplex points centered on the origin with given magnitude 

    Args:
    n_dim: dimension of origin point
    mag: distance from central point to each simplex vertex

    Returns:
    List of vertices representing simplex points
    """
    
    super_vecs = normalized(rotate_away_last_dim(center_origin(find_orig_coords(n_dim+1))), mag)
    simplex_vecs = [s[:-1] for s in super_vecs] # drop the auxiliary last coordinate
    return simplex_vecs


def simplex_plus_antisimplex(n_dim, mag=1):
    """
    Returns simplex and anti-simplex points around n-dim origin with magnitude
    """
    simplex_vecs = simplex_points(n_dim, mag)
    total_vecs = simplex_vecs + [-x for x in simplex_vecs]  # add the anti-simplex
    return total_vecs

def simplex_highd(n_dim, mag=1):
    """
    For very high dim, the simplex points are approx. the 1-hot and -1-hot vectors

    Args:
    n_dim: dimension of origin
    mag: desired magnitude of 1-hot vectors

    Returns:
    list of 1-hot and -1-hot vectors of given magnitude about the origin
    """
    vecs = []
    for i in range(n_dim):
        curr_vec = np.array([0]*n_dim)
        curr_vec[i] = mag
        vecs.append(curr_vec)

    for i in range(n_dim):
        curr_vec = np.array([0]*n_dim)
        curr_vec[i] = -mag
        vecs.append(curr_vec)

    return vecs


def main():
    # a few unit tests here

    # testing 1-hot vecs
    hot2 = find_orig_coords(2)
    assert len(hot2) == 2
    assert np.mean(hot2) == 0.5

    # testing normalization
    v = np.array([1,2,2])
    assert (normalize(v,1)[0] - 0.333333 <= 1e-5)

    # testing simplex vecs centered on origin and same length
    vecs = simplex_points(2)
    assert np.sum(vecs) < 1e-10
    norms = [np.linalg.norm(v) for v in vecs]
    assert np.sum([x-norms[0] for x in  norms]) < 1e-10
    vecs = simplex_points(5)
    assert np.sum(vecs) < 1e-10
    norms = [np.linalg.norm(v) for v in vecs]
    assert np.sum([x-norms[0] for x in  norms]) < 1e-10

    # testing simplex + antisimplex are balanced
    total = simplex_plus_antisimplex(6)
    assert np.sum(total) < 1e-10

    # testing high-d simplex
    vecs = simplex_highd(5,2)
    for i in range(5):
        for j in range(i+1,5):
            dotprod = np.dot(vecs[i],vecs[j])
            assert (dotprod == 0 or dotprod == -4)

if __name__ == "__main__":
    main()
