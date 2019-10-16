"""
@author: Karl Royen
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

pytestmark = pytest.mark.backend('module://mplopengl.backend_qtgl')


@image_comparison(baseline_images=['basics_example'],
                  extensions=['png'])
def test_basics_example():
    # SF bug 2852168
    fig = plt.figure()
    x = np.linspace(0, 2 * np.pi, 100)
    y = 2 * np.sin(x)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('centered spines')
    ax.plot(x, y)
    ax.spines['right'].set_position(('axes', 0.1))
    ax.yaxis.set_ticks_position('right')
    ax.spines['top'].set_position(('axes', 0.25))
    ax.xaxis.set_ticks_position('top')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
