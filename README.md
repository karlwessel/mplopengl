# mplopengl
OpenGL based backend for matplotlib

## Requirements
- matplotlib
- PyOpenGL
- PyQt6
- numpy
- at least OpenGL 2.1 capable graphics device (most machines have that)

Latest versions tested with:
- matplotlib 3.8.0
- PyOpenGL 3.1.7
- PyQt6 6.4.2
- numpy 1.26.1

Oldest working versions tested with:
- matplotlib 3.8.0
- PyOpenGL 3.1.7
- PyQt6 6.4.2
- numpy 1.26.1

Tested with:
- Linux (Debian)

Tested on:
- NVIDIA Quadro P620/PCIe, Debian 11 & KDE Plasma, OpenGL ES 3.2

## Installation
You can install directly from the github repository using pip with:
```bash
pip install git+https://github.com/buokae/mplopengl.git
```
and use the OpenGL based backend with
```python
import matplotlib
matplotlib.use('module://mplopengl.backend_qtgl')
import matplotlib.pyplot as plt
plt.plot([1,2,1])
plt.show()
```
