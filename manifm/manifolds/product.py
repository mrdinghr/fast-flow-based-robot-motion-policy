import math
import numpy as np
import torch
import geoopt
from geoopt.manifolds import ProductManifold as geoopt_ProductManifold


class ProductManifold(geoopt_ProductManifold):
    def random_base(self, *size, dtype=None, device=None) -> "geoopt.ManifoldTensor":
        shape = geoopt.utils.size2shape(*size)
        self._assert_check_shape(shape, "x")
        batch_shape = shape[:-1]
        points = []
        for manifold, shape in zip(self.manifolds, self.shapes):
            points.append(
                manifold.random_base(*(batch_shape + shape), dtype=dtype, device=device)
            )
        tensor = self.pack_point(*points)
        return geoopt.ManifoldTensor(tensor, manifold=self)

    def base_logprob(self, x):
        prob = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            prob.append(manifold.base_logprob(point))

        prod_prob = prob[0]
        for i in range(1, len(self.manifolds)):
            prod_prob *= prob[i]

        return prod_prob


class ProductManifoldTrajectories(ProductManifold):
    # !! Assumes that all manifolds in self.manifolds are the same !!
    def random_base(self, *size, dtype=None, device=None, different_sample=False) -> "geoopt.ManifoldTensor":
        shape = geoopt.utils.size2shape(*size)
        self._assert_check_shape(shape, "x")
        batch_shape = shape[:-1]
        # !! Assumes that all manifolds in self.manifolds are the same !!
        random_point = self.manifolds[0].random_base(*(batch_shape + self.shapes[0]), dtype=dtype, device=device)
        points = []
        for manifold, shape in zip(self.manifolds, self.shapes):
            if different_sample:
                random_point = self.manifolds[0].random_base(*(batch_shape + self.shapes[0]), dtype=dtype, device=device)
            points.append(random_point)
        tensor = self.pack_point(*points)
        return geoopt.ManifoldTensor(tensor, manifold=self)


# this one is to deal with robot end effector dataset, dim = 7, first 3 in euclidean, last 4 in sphere
class ProductManifoldRobotTrajectories(ProductManifold):
    def random_base(self, *size, dtype=None, device=None, different_sample=True) -> "geoopt.ManifoldTensor":
        shape = geoopt.utils.size2shape(*size)
        self._assert_check_shape(shape, "x")
        batch_shape = shape[:-1]
        # !! Assumes that all manifolds in self.manifolds are the same !!
        if not different_sample:
            random_point = self.manifolds[0].random_base(*(batch_shape + self.shapes[0]), dtype=dtype, device=device)
        points = []
        for manifold, shape in zip(self.manifolds, self.shapes):
            if different_sample:
                random_point = manifold.random_base(*(batch_shape + self.shapes[0]), dtype=dtype,
                                                             device=device)
            points.append(random_point)
        tensor = self.pack_point(*points)
        return geoopt.ManifoldTensor(tensor, manifold=self)
