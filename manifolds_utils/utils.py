import torch


def compute_christoffel_symbols(metric, metric_derivative, metric_inverse=None):
    """
    Computes the Christoffel symbols of the metric G at x

    Parameters
    ----------
    :param metric: Riemannian metric G(x)
    :param metric_derivative: derivative of the Riemannian metric dG(x)/dx

    Optional parameters
    -------------------
    :param metric_inverse: inverse of the metric inv(G(x))

    :return: Christoffels symbols at x
    """
    if not metric_inverse:
        metric_inverse = torch.linalg.inv(metric)

    return 0.5 * (torch.einsum('im,mkl->ikl', metric_inverse, metric_derivative)
                  + torch.einsum('im,mlk->ikl', metric_inverse, metric_derivative)
                  - torch.einsum('im,klm->ikl', metric_inverse, metric_derivative))


def duplication_matrix(n=2) -> torch.Tensor:
    """
    Returns the duplication matrix d such that vec(A) = d @ v(A) where v(A) is the
    reduced representation of A and vec(A) is the vectorized representation of A.

    Let A.j be the j-th column of a n x n matrix A, then vec(A) = [A.1, A.2, ..., A.n]^T is
    the n^2-dim vector representation of A.

    Symmetric matrices contain redundant information. For example for n=3 vec(A)
    has the form vec(A) = [a11, a21, a31, a12, a22, a32, a13, a23, a33]^T.
    The reduced form v(A) has then the form v(A) = [a11, a22, a33, a21, a32, a31]^T.

    Parameters
    ----------
    :param n: number of rows and columns of the matrix A

    :return: duplication matrix d
    """
    # TODO generalize duplication matrices
    # recover vectorized representation vec(A) = d @ v(A)
    return torch.Tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0]
    ])


def inv_transpose_duplication_matrix(n=2) -> torch.Tensor:
    """
    Returns the inverse transpose duplication matrix d_inv_t such that v(A) = d_inv_t^T @ vec(A) where v(A) is the
    reduced representation of A and vec(A) is the vectorized representation of A.

    Let A.j be the j-th column of a n x n matrix A, then vec(A) = [A.1, A.2, ..., A.n]^T is
    the n^2-dim vector representation of A.

    Symmetric matrices contain redundant information. For example for n=3 vec(A)
    has the form vec(A) = [a11, a21, a31, a12, a22, a32, a13, a23, a33]^T.
    The reduced form v(A) has then the form v(A) = [a11, a22, a33, a21, a32, a31]^T.

    Parameters
    ----------
    :param n: number of rows and columns of the matrix A

    :return: inverse transpose duplication matrix d_inv_t
    """
    # TODO generalize duplication matrices
    # reduced representation v(A) = d_inv_t^T @ vec(A)
    return torch.Tensor([
        [1, 0, 0],
        [0, 0, .5],
        [0, 0, .5],
        [0, 1, 0]
    ])


def inv_duplication_matrix(n=2) -> torch.Tensor:
    """
    Returns the inverse transpose duplication matrix d_inv_t such that v(A) = d_inv_t^T @ vec(A) where v(A) is the
    reduced representation of A and vec(A) is the vectorized representation of A.

    Let A.j be the j-th column of a n x n matrix A, then vec(A) = [A.1, A.2, ..., A.n]^T is
    the n^2-dim vector representation of A.

    Symmetric matrices contain redundant information. For example for n=3 vec(A)
    has the form vec(A) = [a11, a21, a31, a12, a22, a32, a13, a23, a33]^T.
    The reduced form v(A) has then the form v(A) = [a11, a22, a33, a21, a32, a31]^T.

    Parameters
    ----------
    :param n: number of rows and columns of the matrix A

    :return: inverse transpose duplication matrix d_inv_t
    """
    # TODO generalize duplication matrices
    # reduced representation v(A) = d_inv @ vec(A)
    return inv_transpose_duplication_matrix(n).transpose(0, 1)


def vector_unstack(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Unstacks a vector x into two vectors x1 and x2 such that x = [x1, x2]^T.

    Parameters
    ----------
    :param x: vector to unstack

    :return: two vectors x1 and x2
    """
    assert len(x.shape) == 1
    nb_dofs = x.shape[0] // 2
    x1 = x[:nb_dofs]
    x2 = x[nb_dofs:]

    return x1, x2


def matrix_unstack(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Unstacks a vector x into two matrices x1 and x2 such that x = [x1, x2]^T.

    Parameters
    ----------
    :param x: vector to unstack

    :return: two matrices x1 and x2
    """
    assert len(x.shape) == 2
    nb_dofs = x.shape[-1] // 2
    x1 = x[:, :nb_dofs]
    x2 = x[:, nb_dofs:]
    return x1, x2
