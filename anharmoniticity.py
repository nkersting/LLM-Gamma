#!/usr/bin/python

import torch
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image, ImageOps
import torchvision.transforms as transforms

import numpy as np
import itertools
import random
from functools import partial


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def random_vecs(num_points, mag, dim):
    """
    Generates random floating point vectors of given magnitude and dimension

    Args:
      num_points: number of vectors to generate
      mag: magnitude
      dim: dimension

    Returns:
      list of randomly generated vectors (lists)
    """
    r_vecs = []
    for _ in range(num_points):
        rawvec = normalize(np.random.rand(dim))
        for i in range(dim):
            rawvec[i] = mag * rawvec[i]
            if random.random() >= 0.5:
                rawvec[i] = -rawvec[i]
        r_vecs.append(rawvec)
    return r_vecs


def simplex_anharmoniticity(fn, point, simplex_vecs, sampling_fraction=1):
    """
    Computes anharmoniticity of given function at the point, by sampling function at surrounding simplex points

    Args:
      fn: function to test; must take point as argument and return scalar
      point: multidimensional point to test
      simplex_vecs: list of simplex points to translate from point (of same dimension)
      sampling_fraction: fraction of simplex vecs to test

    Returns:
      value of anharmoniticity
    """
    anharmoniticity = 0
    num_sampled = 0
    for vec in simplex_vecs:
        if random.random() < sampling_fraction:
            anharmoniticity += fn(
                torch.add(torch.tensor(vec, dtype=torch.float), point)
            )
            num_sampled += 1
    anharmoniticity /= num_sampled
    anharmoniticity -= fn(point)
    return anharmoniticity


def average_anharmoniticity(
    fn, central_point, num_trials, mag, n, simplex_vecs, sampling_fraction=1
):
    """
    Computes the average anharmoniticity about an input point by averaging anharmoniticities about that point

    Args:
       fn: function to test; must take point as argument and return scalar
       central_point: multidimensional point to test
       num_trials: number of points to randomly sample about central point
       mag: magnitude to wander away from central point in generating sampling points
       n: dimension of central point
       simplex_vecs: list of simplex points to translate from point (of same dimension)
       sampling_fraction: fraction of simplex vecs to test
    Returns:
       tuple of (average anharmoniticity, point with highest anharmoniticity near this point)
    """
    avg_anharmoniticity = 0
    max_ahar = 0
    for _ in tqdm(range(num_trials)):
        random_point = torch.add(
            central_point, torch.tensor(random_vecs(1, mag, n)[0], dtype=torch.float)
        )
        anharmoniticity = simplex_anharmoniticity(
            fn, random_point, simplex_vecs, sampling_fraction
        )
        avg_anharmoniticity += abs(anharmoniticity)
        if abs(anharmoniticity) > max_ahar:
            max_ahar = abs(anharmoniticity)
            maxpoint = random_point
    avg_anharmoniticity /= num_trials
    return avg_anharmoniticity, maxpoint


def load_simplex_highd(n_dim, mag):
    """
    For very high dim, the simplex points are approximately just 1-hot and -1-hot vectors

    Args:
       n_dim: dimension of simplex to generate
       mag: magnitude to scale each simplex point

    Returns:
       list of approximate simplex vectors
    """
    vecs = []
    for i in range(n_dim):
        curr_vec = np.array([0] * n_dim)
        curr_vec[i] = mag
        vecs.append(curr_vec)

    for i in range(n_dim):
        curr_vec = np.array([0] * n_dim)
        curr_vec[i] = -mag
        vecs.append(curr_vec)

    return vecs


def ResNet_predict(processor, model, image):
    """
    Applies ResNet model to image, predicting label and class

    Args:
      processor: ''
      model: ''
      image: image object
    Returns:
      tuple of (predicted label, label number)
    """
    inputs = processor(image, return_tensors="pt")
    inputs.to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label], predicted_label


def ResNet_predict_avg_logits(processor, model, image):
    """
    Applies ResNet model to image, predicting average of all logits

    Args:
      processor: ''
      model: ''
      image: image object
    Returns:
      average of logits over all classes
    """
    inputs = processor(image, return_tensors="pt")
    inputs.to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    return torch.mean(logits[0]).item()


def ResNet_predict_logit(processor, model, image, idx):
    """
    Applies ResNet model to image, giving value of logit at given index

    Args:
      processor: ''
      model: ''
      image: image object
      idx: position of index at which to return logit value
    Returns:
      value of desired logit
    """
    inputs = processor(image, return_tensors="pt")
    inputs.to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    return logits[0][idx]


def resnet_idx_logit(to_img_transform, processor, model, idx, dim, point):
    return ResNet_predict_logit(
        processor, model, vector_to_gray_image(to_img_transform, point, dim, dim), idx
    )


def vector_to_gray_image(to_img_transform, vector, xdim, ydim, scale_factor=1 / 255):
    """
    Converts 1d tensor of floats to image
    """
    x = vector.reshape(1, xdim, ydim) * scale_factor
    y = x.expand(3, xdim, ydim)
    return to_img_transform(y)


def gray_image_to_vector(transform, image):
    """
    Converts gray image to 1d tensor of floats (dimension = Length*Width)
    """
    tensor = transform(image).float()
    xdim = tensor.shape[1]
    ydim = tensor.shape[2]
    return tensor[0][:][:].reshape(xdim * ydim)


def image_to_gray_image(to_img_transform, transform, image):
    """
    Converts RGB image to grayscale image
    """
    gray_image = ImageOps.grayscale(image)
    gray_img_tensor = transform(gray_image).float() / 255
    xdim = gray_img_tensor.shape[1]
    ydim = gray_img_tensor.shape[2]
    new_gray_img_tensor = gray_img_tensor.expand(3, xdim, ydim)
    return to_img_transform(new_gray_img_tensor)
