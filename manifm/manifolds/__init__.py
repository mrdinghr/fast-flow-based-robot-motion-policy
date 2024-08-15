"""Copyright (c) Meta Platforms, Inc. and affiliates."""

# wrap around the manifolds from geoopt
from .euclidean import Euclidean
from .torus import FlatTorus
from .spd import SPD
from .sphere import Sphere
from .hyperbolic import PoincareBall
from .product import ProductManifold, ProductManifoldTrajectories
from .mesh import Mesh
from .utils import geodesic
