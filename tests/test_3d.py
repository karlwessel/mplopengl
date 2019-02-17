"""
@author: Karl Royen
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

pytestmark = pytest.mark.backend('module://mplopengl.backend_qtgl')


@image_comparison(baseline_images=['threeD_example'],
                  extensions=['png'])
def test_3d_example():
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    def randrange(n, vmin, vmax):
        '''
        Helper function to make an array of random numbers having shape (n, )
        with each number distributed Uniform(vmin, vmax).
        '''
        return (vmax - vmin) * np.random.rand(n) + vmin

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zlow, zhigh)
        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


@image_comparison(baseline_images=['surface_example'],
                  extensions=['png'])
def test_surface_example():
    def fun(x, y):
        return x ** 2 + y

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-3.0, 3.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


@image_comparison(baseline_images=['quadmesh_example'],
                  extensions=['png'])
def test_quadmesh_example():
    import copy
    from matplotlib import cm

    n = 12
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n * 2)
    X, Y = np.meshgrid(x, y)
    Qx = np.cos(Y) - np.cos(X)
    Qz = np.sin(Y) + np.sin(X)
    Z = np.sqrt(X ** 2 + Y ** 2) / 5
    Z = (Z - Z.min()) / (Z.max() - Z.min())

    # The color array can include masked values.
    Zm = np.ma.masked_where(np.abs(Qz) < 0.5 * np.max(Qz), Z)

    fig, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].pcolormesh(Qx, Qz, Z, shading='gouraud')
    axs[0].set_title('Without masked values')

    # You can control the color of the masked region. We copy the default colormap
    # before modifying it.
    cmap = copy.copy(cm.get_cmap(plt.rcParams['image.cmap']))
    cmap.set_bad('y', 1.0)
    axs[1].pcolormesh(Qx, Qz, Zm, shading='gouraud', cmap=cmap)
    axs[1].set_title('With masked values')

    # Or use the default, which is transparent.
    axs[2].pcolormesh(Qx, Qz, Zm, shading='gouraud')
    axs[2].set_title('With masked values')

    fig.tight_layout()
