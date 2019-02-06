# mplopengl
OpenGL based backend for matplotlib

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
