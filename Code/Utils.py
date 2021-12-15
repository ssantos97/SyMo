#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 00:54:02 2020

@author: saul
"""

import torch
import numpy as np
from numpy.linalg import inv
import json
import pickle
from torch import nn

def non_linearity(activation):
    activations = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'softplus': nn.Softplus,
        'sigmoid': nn.Sigmoid,
        'leaky_relu': nn.LeakyReLU,
        'selu': nn.SELU,
        'identity': nn.Identity
    }
    return activations[activation]

# from Neural_Networks import Jacobian
def get_n_params(model):
    """
        Returns the number of learning parameters of the model
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def wrap_to_pi(state, mask):
    '''
    wrap generalized coordinates to [-pi, pi]
    --> mask indicates which indexes must be wrapped: list 
    '''
    state[:,mask] = (state[:, mask] + pi()) % (2 * pi()) - pi()
    
    return state

def pi(n=None):
    #Returns pi - Avoid numpy
    return torch.acos(torch.zeros(1)).item() * 2

class Params():
    """
    Copied from https://github.com/cs230-stanford/cs230-code-examples
    
    
    Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```

    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

