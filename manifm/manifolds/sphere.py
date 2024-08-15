"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
import numpy as np
import torch
from torch.func import jacrev, vmap
from geoopt.manifolds import Sphere as geoopt_Sphere
from manifm.manifolds.euclidean import normal_logprob


class Sphere(geoopt_Sphere):
    # scale_std 0.5
    def __init__(self, scale_std=0.5, intersection: torch.Tensor = None, complement: torch.Tensor = None):
        super().__init__(intersection, complement)
        self.scale_std = scale_std

    def transp(self, x, y, v):
        denom = 1 + self.inner(x, x, y, keepdim=True)
        res = v - self.inner(x, y, v, keepdim=True) / denom * (x + y)
        cond = denom.gt(1e-3)
        return torch.where(cond, res, -v)

    def uniform_logprob(self, x):
        dim = x.shape[-1]
        return torch.full_like(
            x[..., 0],
            math.lgamma(dim / 2) - (math.log(2) + (dim / 2) * math.log(math.pi)),
        )

    def random_wrapped(self, *size, dtype=None, device=None):
        bsz = int(np.prod(size[:-1]))
        d = size[-1]

        # Wrap a Gaussian centered at (0, 0, ..., 0, 1)
        c = torch.zeros(size, dtype=dtype, device=device)
        c[..., -1] = 1.0

        # Construct tangent vectors where elements are iid Normal.
        u_tgt = self.scale_std * torch.randn(bsz, d-1).to(dtype=dtype, device=device)
        u_zero = torch.zeros(bsz, 1).to(dtype=dtype, device=device)
        u = torch.cat((u_tgt, u_zero), -1)

        # Exponential map to the manifold.
        x = self.expmap(c, u)

        return x

    def random_base(self, *size, dtype=None, device=None):
        # return self.random_uniform(*args, **kwargs)
        return self.random_wrapped(*size, dtype=dtype, device=device)

    def base_logprob(self, x):
        # return self.uniform_logprob(*args, **kwargs)
        return self.wrapped_logprob(x)

    def wrapped_logprob(self, x):
        size = x.shape
        d = x.shape[-1]
        x = x.reshape(-1, d)

        # Wrap a Gaussian centered at (0, 0, ..., 0, 1)
        c = torch.zeros(size, dtype=x.dtype, device=x.device)
        c[..., -1] = 1.0

        u = self.logmap(c, x)
        logpu = normal_logprob(u, 0.0, np.log(self.scale_std)).sum(-1)

        # Warning: For some reason, functorch doesn't play well with the sqrtmh implementation.
        with torch.inference_mode(mode=False):

            def logdetjac(f):
                def _logdetjac(*args):
                    jac = jacrev(f, chunk_size=256)(*args)
                    return torch.linalg.slogdet(jac)[1]

                return _logdetjac

            # Change of variables in Euclidean space
            ldjs = vmap(logdetjac(self.expmap))(c, u)
            logpu = logpu - ldjs

        ldgs = torch.slogdet(torch.eye(d, dtype=x.dtype, device=x.device))[1]

        logpx = logpu - 0.5 * ldgs
        return logpx.reshape(*size[:-1])





