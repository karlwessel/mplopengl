"""
This is a copy of matplotlibs "Embedding in Qt" example modified to use the
OpenGL based qt widget instead of the Agg based widget.
"""

import sys
import time

import numpy as np

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5

# the only thing changed from the example is that we need qt5...
assert is_pyqt5()
# ... and don't use qt5agg's FigureCanvas...
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# ... but qtgl's FigureCanvas
from mplopengl.backend_qtgl import FigureCanvas
#from matplotlib.backends.backend_qt5agg import FigureCanvas

from matplotlib.figure import Figure
import matplotlib.cm as cm


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(static_canvas)
        self.addToolBar(NavigationToolbar(static_canvas, self))

        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(dynamic_canvas)
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                        NavigationToolbar(dynamic_canvas, self))

        self._static_ax = static_canvas.figure.subplots()
        delta = 0.025
        x = y = np.arange(-3.0, 3.0, delta)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X ** 2 - Y ** 2)
        Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
        Z = (Z1 - Z2) * 2
        print(Z.shape)
        im = self._static_ax.imshow(Z, interpolation="None", cmap=cm.RdYlGn,
                                    origin='lower', extent=[0, 10, -100, 100],
                                    vmax=abs(Z).max(), vmin=-abs(Z).max(),
                                    aspect='auto')
        t = np.linspace(0, 10, 501)

        self._static_ax.plot(t, np.tan(t), ".")

        self._dynamic_ax = dynamic_canvas.figure.subplots()
        self._timer = dynamic_canvas.new_timer(
            100, [(self._update_canvas, (), {})])
        self._timer.start()
        self.data = None

    def _update_canvas(self):
        #self._dynamic_ax.clear()
        t = np.linspace(0, 10, 101)
        # Shift the sinusoid as a function of time.
        if self.data is None:
            self.data = self._dynamic_ax.plot(t, np.sin(t + time.time()), marker="o")[0]
        else:
            self.data.set_data(t, np.sin(t + time.time()))
        self._dynamic_ax.figure.canvas.draw()


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()