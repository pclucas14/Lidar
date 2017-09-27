import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision
import numpy as np
from numpy import unravel_index
import matplotlib.pyplot as plt
import torch.optim as optim
import pdb
import hickle as hkl
import torch.autograd as autograd
from PIL import Image
import ot

#######################################################
#              Embed to Control Helpers               #
#######################################################

'''
adapted from https://github.com/ericjang/e2c/blob/master/e2c_plane.py
'''
class Gaussian(object):
    def __init__(self, mu, sigma, logsigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma
        self.logsigma = logsigma
        self.v = v
        self.r = r


def KL_gaussians(Q, N):
    sum = lambda x : x.sum(dim=1)
    eps = 1e-9
    k = Q.mu.size(1) # dimension of distribution
    s02, s12 = torch.square(Q.sigma), torch.square(N.sigma) + eps
    a = sum(s02*(1.+2.*Q.v*Q.r)/s12) + sum(torch.square(Q.v)/s12) * sum(torch.square(Q.r)*s02)
    b = sum(torch.square(N.mu - Q.mu) / s12)
    c = 2. * (sum(N.logsigma - Q.logsigma) - torch.log(1 + sum(Q.v*Q.r)))
    return .5 * (a + b - k + c)








#######################################################
#                 Regular utils                       #
#######################################################

def oned_to_threed(velo):
    heights = np.arange(-.02, .02, .04 / velo.shape[0])
    # velo has shape 60, 512, 1
    out = np.zeros((velo.shape[0], velo.shape[1], 3))
    angles = np.arange(0, 2* np.pi, 2 * np.pi / velo.shape[1])
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            x = velo[i][j] * np.cos(angles[j])
            y = velo[i][j] * np.sin(angles[j])
            out[i][j] = np.array([x, y, heights[i]])

    return out


def load_kitti_lidar(bbs=100, batch_size=32, train='train'):
    import hickle as hkl
    # X = hkl.load('../prednet/kitti_data/X_'+train+'_lidar_raw.hkl')
    X = hkl.load('data.hkl')
    X = X * 2 # [-.5, .5] --> [-1, 1]
    datap, n, c = X.shape
    # scale each 3 dimension so that they E (0, 1)
    while True :
        batches = np.zeros((bbs * batch_size, n, c))
        indices = np.random.choice(X.shape[0], bbs * batch_size)
        batches = X[indices]
        # batches[:, :, 0] /= 80.
        # batches[:, :, 1] /= 80.
        # batches[:, :, 2] /= 30.
        yield batches.astype('float32')


def iter_minibatches(inputs, batch_size, extra=None, forever=False):
    while True : 
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            excerpt = slice(start_idx, start_idx + batch_size)
            if extra is not None : 
                yield (inputs[excerpt], extra[excerpt]) # [x[excerpt] for x in extra])
            else : 
                yield inputs[excerpt]
        if not forever : 
            break


