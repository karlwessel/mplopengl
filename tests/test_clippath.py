"""
@author: Karl Royen
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cbook as cbook
import pytest
import numpy as np
from matplotlib.testing.decorators import image_comparison

pytestmark = pytest.mark.backend('module://mplopengl.backend_qtgl')

@pytest.mark.xfail
@image_comparison(baseline_images=['clippath_example'],
                  extensions=['png'])
def test_clippath_example():
    with cbook.get_sample_data('grace_hopper.png') as image_file:
        image = plt.imread(image_file)

    fig, ax = plt.subplots()
    im = ax.imshow(image)
    patch = patches.Circle((260, 200), radius=200, transform=ax.transData)
    im.set_clip_path(patch)

    ax.axis('off')
