"""
@author: Karl Royen
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

pytestmark = pytest.mark.backend('module://mplopengl.backend_qtgl')

@image_comparison(baseline_images=['gouradtri_example'],
                  extensions=['png'])
def test_gouradtri_example():
    figure = plt.figure()
    ax = figure.add_subplot(111)

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

    ax.pcolormesh(Qx, Qz, Z, shading='gouraud')
    ax.set_title('Without masked values')
