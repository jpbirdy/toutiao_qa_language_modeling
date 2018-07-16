import numpy as np


def MaxMinNormalization(x, Max=None, Min=None):
    if Min == None:
        Min = np.min(x)
    if Max == None:
        Max = np.max(x)
    x = (x - Min) / (Max - Min);
    return x, Max, Min


def Z_ScoreNormalization(x, mu, sigma):
    if mu is None:
        mu = np.average(x)
    if sigma is None:
        sigma = np.std(x)
    x = (x - mu) / sigma
    return x, mu, sigma


def sigmoid(X):
    return 1.0 / (1 + np.exp(-float(X)))
