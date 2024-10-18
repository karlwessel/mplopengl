# Not actively maintainend
While it still works (state of 2024), this library is not actively maintained. 

Mainly because I don't use python for plotting anymore. 

Secondly because I it didn't really fix the performance issues I had when plotting. In most cases the actual rendering is not the bottleneck but the matplotlib pipeline coming before it (state of 2019, love to stand corrected if that changed). This library can only improve the speed of the rendering.

# mplopengl
OpenGL based backend for matplotlib

## Requirements
- matplotlib >= 3.1
- pyopengl >= 3.1
- pyqt5
- numpy
- at least OpenGL 2.1 capable graphics device (most machines have that)

Latest versions tested with:
- matplotlib 3.9.2
- pyqt 5.15.11
- numpy 2.1.2
- pyopengl 3.1.7
- python 3.12.7

Oldest working versions tested with:
- matplotlib 3.1
- pyopengl 3.1
- pyqt5 5.7.1
- numpy 1.14.5

Tested with:
- Linux (archlinux)

Tested on:
- Mesa DRI Intel Ironlake Mobile (Thinkpad 410s integrated graphics), OpenGL v2.1
- Mesa Intel(R) HD Graphics 520 (SKL GT2), OpenGL v4.6

## Installation
You can install directly from the github repository using pip with:
```bash
pip install git+https://github.com/karlwessel/mplopengl.git
```
and use the OpenGL based backend with
```python
import matplotlib
matplotlib.use('module://mplopengl.backend_qtgl')
import matplotlib.pyplot as plt
plt.plot([1,2,1])
plt.show()
```
