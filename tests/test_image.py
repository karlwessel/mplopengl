import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import pytest
import matplotlib.cm as cm

pytestmark = pytest.mark.backend('module://mplopengl.backend_qtgl')

@image_comparison(baseline_images=['image_example'],
                  extensions=['png'])
def test_image_example():
    delta = 0.025
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2) * 2

    fig, ax = plt.subplots()
    im = ax.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
                   origin='lower', extent=[-3, 3, -3, 3],
                   vmax=abs(Z).max(), vmin=-abs(Z).max())

@image_comparison(baseline_images=['image_aniso'],
                  extensions=['png'])
def test_image_aniso():
    delta = 0.025
    x = np.arange(-2.0, 2.0, delta)
    y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2) * 2

    fig, ax = plt.subplots()
    im = ax.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
                   origin='lower', extent=[-2, 2, -3, 3],
                   vmax=abs(Z).max(), vmin=-abs(Z).max())

@image_comparison(baseline_images=['image_no_interp'],
                  extensions=['png'])
def test_image_no_interp():
    delta = 0.025
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2) * 2

    fig, ax = plt.subplots()
    im = ax.imshow(Z, interpolation='None', cmap=cm.RdYlGn,
                   origin='lower', extent=[-3, 3, -3, 3],
                   vmax=abs(Z).max(), vmin=-abs(Z).max())