"""
Created on Feb 4, 2019

@author: Karl Royen
"""

from OpenGL.GL import *
from PyQt5.QtCore import pyqtSignal
from matplotlib.backend_bases import GraphicsContextBase
from matplotlib.backends.backend_qt5 import FigureCanvasQT
from matplotlib.backends.backend_qt5 import FigureManagerQT
from matplotlib.backends.qt_compat import QtWidgets

from mplopengl.renderergl import RendererGL


class GraphicsContextQtGL(GraphicsContextBase):
    """
    The graphics context provides the color, line styles, etc...  See the gtk
    and postscript backends for examples of mapping the graphics context
    attributes (cap styles, join styles, line widths, colors) to a particular
    backend.  In GTK this is done by wrapping a gtk.gdk.GC object and
    forwarding the appropriate calls to it using a dictionary mapping styles
    to gdk constants.  In Postscript, all the work is done by the renderer,
    mapping line styles to postscript calls.

    If it's more appropriate to do the mapping at the renderer level (as in
    the postscript backend), you don't need to override any of the GC
    methods.
    If it's more appropriate to wrap an instance (as in the GTK backend) and
    do the mapping here, you'll need to override several of the setter
    methods.

    The base GraphicsContext stores colors as a RGB tuple on the unit
    interval, e.g., (0.5, 0.0, 1.0). You may need to map this to colors
    appropriate for your backend.
    """
    pass


class FigureCanvasQtGL(FigureCanvasQT):
    # noinspection PyPep8Naming
    class GLCanvas(QtWidgets.QOpenGLWidget):
        available = pyqtSignal()

        def __init__(self, figure):
            super().__init__()
            self.figure = figure
            self.renderer = None

            w, h = self.width(), self.height()
            self.resize(w, h)

        def paintGL(self):
            glClear(GL_COLOR_BUFFER_BIT)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            self.figure.draw(self.renderer)

        def resizeGL(self, w, h):
            if w > 0 and h > 0:
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                glOrtho(0, w, 0.0, h, 0, 1.0)
                glViewport(0, 0, w, h)
                self.renderer = RendererGL(w, h, self.figure.dpi)
                self.available.emit()

        def initializeGL(self):
            glEnable(GL_LINE_STIPPLE)
            # enable antialiasing
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glEnable(GL_COLOR_MATERIAL)

            glClearColor(1.0, 1.0, 1.0, 1.0)

    def __init__(self, figure):
        super().__init__(figure=figure)

        # necessary for base class to update plot grid (the value doesn't
        # really matter but shouldn't be None
        self._dpi_ratio_prev = 1

        self.canvas = self.GLCanvas(figure)
        self.canvas.available.connect(self.on_renderer)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.canvas)

        self.save_queue = []

    def get_renderer(self, clear=False):
        if self.canvas.renderer is None:
            return RendererGL(self, self.width(), self.height(), self.figure.dpi)
        return self.canvas.renderer

    renderer = property(get_renderer, None)

    def print_png(self, file, *args, **kwargs):
        self.window().show()
        # self.activateWindow()
        # self.raise_()
        if self.canvas.renderer is None:
            self.save_queue.append(file)
        else:
            self.canvas.grabFramebuffer().save(file)

    def on_renderer(self):
        for file in self.save_queue:
            self.canvas.grabFramebuffer().save(file)

    def draw(self):
        """
        Draw the figure using the renderer
        """

        if self.canvas is None:
            return

        self.canvas.update()


class FigureManagerQtGL(FigureManagerQT):
    """
    Wrap everything up into a window for the pylab interface

    For non interactive backends, the base class does all the work
    """
    pass


########################################################################
#
# Now just provide the standard names that backend.__init__ is expecting
#
########################################################################

FigureCanvas = FigureCanvasQtGL
FigureManager = FigureManagerQtGL
