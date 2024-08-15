import numpy as np
import matplotlib.pyplot as plt


def plot_sphere(ax, base=None, color=None, alpha=0.8, r=0.99, linewidth=0, lim=1.1, n_elems=100, **kwargs):
    """
    Plots a sphere
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param ax: figure axes

    Optional parameters
    -------------------
    :param color: color of the surface
    :param alpha: transparency index
    :param r: radius
    :param linewidth: linewidth of sphere lines
    :param lim: axes limits
    :param n_elems: number of points in the surface
    :param kwargs:

    Returns
    -------
    :return: -
    """
    if base is None:
        base = [0, 0, 1]
    else:
        if len(base) != 3:
            base = [0, 0, 1]
            print('Base was set to its default value as a wrong argument was given!')

    if color is None:
        color = [0.8, 0.8, 0.8]
    else:
        if len(color) != 3:
            color = [0.8, 0.8, 0.8]
            print('Sphere color was set to its default value as a wrong color argument was given!')

    u = np.linspace(0, 2 * np.pi, n_elems)
    v = np.linspace(0, np.pi, n_elems)

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, linewidth=linewidth, alpha=alpha, **kwargs)
    # ax.plot(xs=[base[0]], ys=[base[1]], zs=[base[2]], marker='*', color=color)

    ax.set_xlim3d([-lim, lim])
    ax.set_ylim3d([-lim, lim])
    ax.set_zlim3d([-0.75*lim, 0.75*lim])