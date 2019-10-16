from setuptools import setup

setup(
    name='mplopengl',
    version='0.1',
    packages=['mplopengl'],
    url='https://github.com/karlwessel/mplopengl',
    license='MIT',
    author='Karl Royen',
    author_email='kwessel@astro.physik.uni-goettingen.de',
    description='OpenGL based backend for matplotlib',
    install_requires=['matplotlib>=3.1', 'numpy', 'pyqt5', 'pyopengl>=3.1']
)
