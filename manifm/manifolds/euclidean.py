import math
import numpy as np
import torch
from torch.func import jacrev, vmap
from geoopt.manifolds import Euclidean as geoopt_Euclidean


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.0)
    log_std = log_std + torch.tensor(0.0)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


class Euclidean(geoopt_Euclidean):
    def __init__(self, scale_std=1.0, ndim=0):
        super().__init__(ndim)
        self.scale_std = scale_std

    def random_base(self, *size, dtype=None, device=None):
        return self.random_normal(*size, mean=0.0, std=self.scale_std, dtype=dtype, device=device)

    def base_logprob(self, x):
        return normal_logprob(x, 0.0, np.log(self.scale_std)).sum(-1)
