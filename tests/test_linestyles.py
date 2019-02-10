"""
@author: Karl Royen
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

pytestmark = pytest.mark.backend('module://mplopengl.backend_qtgl')


@image_comparison(baseline_images=['linestyle_example'],
                  extensions=['png'])
def test_linestyle_example():
    color = 'cornflowerblue'
    points = np.ones(5)  # Draw 5 points for each line
    text_style = dict(horizontalalignment='right', verticalalignment='center',
                      fontsize=12, fontdict={'family': 'monospace'})

    def format_axes(ax):
        ax.margins(0.2)
        ax.set_axis_off()

    def nice_repr(text):
        return repr(text).lstrip('u')

    # Plot all line styles.
    fig, ax = plt.subplots()

    linestyles = ['-', '--', '-.', ':']
    for y, linestyle in enumerate(linestyles):
        ax.text(-0.1, y, nice_repr(linestyle), **text_style)
        ax.plot(y * points, linestyle=linestyle, color=color, linewidth=3)
        format_axes(ax)
        ax.set_title('line styles')
