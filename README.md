# mplopengl
OpenGL based backend for matplotlib

## Requirements
- matplotlib >= 3.1
- pyopengl >= 3.1
- pyqt5
- numpy
- at least OpenGL 2.1 capable graphics device (most machines have that)

Latest versions tested with:
- matplotlib 3.1.1
- pyqt 5.13.1
- numpy 1.17.2 (1.12 install fails) 
- pyopengl 3.1

Oldest working versions tested with:
- matplotlib 3.1
- pyopengl 3.1
- pyqt5 5.7.1
- numpy 1.14.5

Tested with:
- Linux (archlinux)

Tested on:
- Mesa DRI Intel(R) Ironlake Mobile (Thinkpad 410s integrated graphics), OpenGL v2.1

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
