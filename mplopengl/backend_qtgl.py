"""
Created on Feb 4, 2019

@author: Karl Royen
"""
from logging import warning
from math import sin, cos, radians

import matplotlib
import numpy
from OpenGL.GL import *
from PyQt5.QtCore import pyqtSignal
from matplotlib import rcParams
from matplotlib.backend_bases import RendererBase, GraphicsContextBase
from matplotlib.backends import backend_agg
from matplotlib.backends.backend_qt5 import FigureCanvasQT
from matplotlib.backends.backend_qt5 import FigureManagerQT
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.font_manager import findfont, get_font
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path, get_paths_extents
from matplotlib.transforms import Affine2D

MIN_LINE_WIDTH = 1.5


class RendererQtGL(RendererBase):

    def __init__(self, widget, width, height, dpi):
        super().__init__()
        self.width = width
        self.height = height
        self.widget = widget
        self.dpi = dpi
        self.mathtext_parser = MathTextParser('Agg')

    def draw_gouraud_triangle(self, gc, points, colors, transform):
        """
        Draw a Goraud-shaded triangle.

        Parameters
        ----------
        gc : GraphicsContextBase
            the :class:`GraphicsContextBase` instance
        points : array_like, shape=(3, 2)
            Array of (x, y) points for the triangle.

        colors : array_like, shape=(3, 4)
            RGBA colors for each point of the triangle.

        transform : `matplotlib.transforms.Transform`
            An affine transform to apply to the points.

        """

        # NOT TESTED!!!
        glPushAttrib(GL_SCISSOR_BIT)
        self.set_clipping(gc)
        transform.transform(points)
        glBegin(GL_TRIANGLES)
        for i in range(3):
            glColor4fv(colors[i])
            glVertex2fv(points[i])
        glEnd()

        glPopAttrib()

    # noinspection PyPep8Naming
    def draw_markers(self, gc, marker_path, marker_trans, path,
                     trans, rgbFace=None):
        """
        Draws a marker at each of the vertices in path.  This includes
        all vertices, including control points on curves.  To avoid
        that behavior, those vertices should be removed before calling
        this function.

        *gc*
            the :class:`GraphicsContextBase` instance

        *marker_trans*
            is an affine transform applied to the marker.

        *trans*
             is an affine transform applied to the path.

        This provides a fallback implementation of draw_markers that
        makes multiple calls to :meth:`draw_path`.  Some backends may
        want to override this method in order to draw the marker only
        once and reuse it multiple times.
        """
        disp_list = glGenLists(1)
        glNewList(disp_list, GL_COMPILE)
        self.draw_path(gc, marker_path, marker_trans, rgbFace)
        glEndList()

        for vertices, _ in path.iter_segments(trans, simplify=False):
            if len(vertices):
                x, y = vertices[-2:]
                glPushMatrix()
                glTranslatef(x, y, 0)
                glCallList(disp_list)
                glPopMatrix()

        glDeleteLists(disp_list, 1)

    @staticmethod
    def path_to_poly(path):
        try:
            polygons = path.to_polygons(closed_only=False)
        except TypeError:
            print(2)
            polygons = path.to_polygons()

        if not polygons:
            print(3)
            polygons = []
            i = 0
            while i < len(path.vertices):
                assert path.codes[i] == 1
                start = i
                i += 1
                while (i < len(path.vertices)
                       and path.codes[i] == Path.LINETO):
                    i += 1

                polygon = numpy.array(path.vertices[start:i])
                if (i < len(path.vertices)
                        and path.codes[i] == Path.CLOSEPOLY):
                    polygon = numpy.append(polygon, path.vertices[start])

                polygons.append(polygon)

        return polygons

    def set_clipping(self, gc):
        bbox = gc.get_clip_rectangle()
        if bbox:
            glEnable(GL_SCISSOR_TEST)
            glScissor(int(bbox.x0), int(bbox.y0), int(bbox.width),
                      int(bbox.height))

        clippath, clippath_trans = gc.get_clip_path()
        if clippath is not None:
            glEnable(GL_SCISSOR_TEST)
            bbox = get_paths_extents([clippath_trans.transform_path(clippath)],
                                     [Affine2D()])
            glScissor(int(bbox.x0), int(bbox.y0), int(bbox.width),
                      int(bbox.height))

    def draw_filled(self, gc, polygons, rgb_face):
        if gc.get_forced_alpha():
            fillopacity = gc.get_alpha()
        else:
            fillopacity = rgb_face[3] if len(rgb_face) > 3 else 1.0

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(*rgb_face[:3], fillopacity)

        for polygon in polygons:
            if len(polygon) > 2:
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(2, GL_DOUBLE, 0, polygon)
                glDrawArrays(GL_POLYGON, 0, len(polygon))
                glDisableClientState(GL_VERTEX_ARRAY)

    def draw_stroked(self, gc, polygons):

        if gc.get_forced_alpha():
            strokeopacity = gc.get_alpha()
        else:
            strokeopacity = gc.get_rgb()[3]

        col = gc.get_rgb()[:3]
        if strokeopacity < 1:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(*col, strokeopacity)
        else:
            glColor3fv(col)

        glPushAttrib(GL_LINE_BIT)
        self.set_line_style(gc, col, strokeopacity)

        for polygon in polygons:
            num_vertices = len(polygon)
            if num_vertices > 1:
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(2, GL_DOUBLE, 0, polygon)
                glDrawArrays(GL_LINE_STRIP, 0, num_vertices)
                glDisableClientState(GL_VERTEX_ARRAY)
            else:
                glBegin(GL_LINE_STRIP)
                for i in range(num_vertices):
                    glVertex2f(polygon[i][0], polygon[i][1])
                glEnd()

        glPopAttrib()  # GL_LINE_BIT

    # noinspection PyPep8Naming
    def draw_path(self, gc, path, transform, rgbFace=None):
        path = transform.transform_path(path)
        polygons = self.path_to_poly(path)

        glPushAttrib(GL_SCISSOR_BIT)
        self.set_clipping(gc)

        if rgbFace is not None:
            self.draw_filled(gc, polygons, rgbFace)

        if gc.get_linewidth() > 0:
            self.draw_stroked(gc, polygons)

        glPopAttrib()  # GL_SCISSOR_BIT
        # glPopMatrix()

    def set_line_style(self, gc, col, alpha):
        width = self.points_to_pixels(gc.get_linewidth())
        # minimum line width is 1.5 for compatibilty with old and/or intel hardware
        # thinner lines get alpha faded

        if width < MIN_LINE_WIDTH:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(*col, alpha * width / MIN_LINE_WIDTH)
            glLineWidth(MIN_LINE_WIDTH)
        else:
            glLineWidth(width)

        style = gc.get_dashes()[1]
        if style is None:
            glLineStipple(1, int('1111111111111111', 2))
        elif len(style) == 2:
            glLineStipple(1, int('0000000011111111', 2))
        elif len(style) == 4:
            glLineStipple(1, int('0000110000111111', 2))
        elif style == "dotted":
            glLineStipple(1, int('0001100011000011', 2))
        else:
            warning("Unknown line style: {}".format(style))
            glLineStipple(1, int('1111111111111111', 2))

    # draw_path_collection is optional, and we get more correct
    # relative timings by leaving it out. backend implementers concerned with
    # performance will probably want to implement it
    #     def draw_path_collection(self, gc, master_transform, paths,
    #                              all_transforms, offsets, offsetTrans,
    #                              facecolors,
    #                              edgecolors, linewidths, linestyles,
    #                              antialiaseds):
    #         pass

    # draw_quad_mesh is optional, and we get more correct
    # relative timings by leaving it out.  backend implementers concerned
    # with performance will probably want to implement it
    #     def draw_quad_mesh(self, gc, master_transform, meshWidth,
    #                        meshHeight,
    #                        coordinates, offsets, offsetTrans, facecolors,
    #                        antialiased, edgecolors):
    #         pass

    def option_scale_image(self):
        return True

    def _get_agg_font(self, prop):
        """
        Get the font for text instance t, caching for efficiency
        """
        fname = findfont(prop)
        font = get_font(fname)

        font.clear()
        size = prop.get_size_in_points()
        font.set_size(size, self.dpi)

        return font

    def draw_text_f2font(self, gc, x, y, s, prop, angle):

        flags = backend_agg.get_hinting_flag()
        font = self._get_agg_font(prop)

        if font is None:
            return None
        if len(s) == 1 and ord(s) > 127:
            font.load_char(ord(s), flags=flags)
        else:
            # We pass '0' for angle here, since it will be rotated (in raster
            # space) in the following call to draw_text_image).
            font.set_text(s, 0, flags=flags)
        font.draw_glyphs_to_bitmap(antialiased=rcParams['text.antialiased'])
        d = font.get_descent() / 64.0
        # The descent needs to be adjusted for the angle.
        xo, yo = font.get_bitmap_offset()
        xo /= 64.0
        yo /= 64.0
        xd = -d * sin(radians(angle))
        yd = d * cos(radians(angle))
        self.draw_text_image(
            font.get_image(), numpy.round(x - xd + xo), numpy.round(y + yd + yo) + 1, angle, gc)

    def bind_image(self, im):
        texture_id = glGenTextures(1)
        # glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, im.shape[1], im.shape[0],
                     0, GL_RGBA, GL_UNSIGNED_BYTE, im)
        glEnable(GL_TEXTURE_2D)
        return texture_id

    def unbind_image(self, texture_id):
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([texture_id])

    def draw_text_image(self, im_raw, x, y, angle, gc):
        if isinstance(im_raw, matplotlib.ft2font.FT2Image):
            im_raw = im_raw.as_array()
        im = numpy.ones((*im_raw.shape, 4), dtype=numpy.uint8) * 255
        im[:, :, 3] = im_raw
        texture_id = self.bind_image(im)
        x0 = [0, 0]
        x1 = im.shape[1], im.shape[0]
        col = gc.get_rgb()[:3]
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_COLOR_MATERIAL)

        y = self.height - y
        glTranslatef(x, y, 0)
        glRotatef(angle, 0, 0, 1)
        glColor3fv(col)
        glBegin(GL_QUADS)
        glVertex2f(x0[0], x0[1])
        glTexCoord2f(1, 1)
        glVertex2f(x1[0], x0[1])
        glTexCoord2f(1, 0)
        glVertex2f(x1[0], x1[1])
        glTexCoord2f(0, 0)
        glVertex2f(x0[0], x1[1])
        glTexCoord2f(0, 1)
        glEnd()

        self.unbind_image(texture_id)
        glRotatef(-angle, 0, 0, 1)
        glTranslatef(-x, -y, 0)

    def draw_image(self, gc, x, y, im, transform=None):
        texture_id = self.bind_image(im)

        if transform is None:
            x0 = [0, 0]
            x1 = [1, 1]
            print(x, y)
            print(im)
        else:
            x0 = transform.transform([0, 0])
            x1 = transform.transform([1, 1])
        glTranslatef(x, y, 0)
        glColor3f(1, 1, 1)
        glBegin(GL_QUADS)
        glVertex2f(x0[0], x0[1])
        glTexCoord2f(1, 0)
        glVertex2f(x1[0], x0[1])
        glTexCoord2f(1, 1)
        glVertex2f(x1[0], x1[1])
        glTexCoord2f(0, 1)
        glVertex2f(x0[0], x1[1])
        glTexCoord2f(0, 0)
        glEnd()

        self.unbind_image(texture_id)
        glTranslatef(-x, -y, 0)

    def draw_mathtext_f2font(self, gc, x, y, s, prop, angle):
        """
        Draw the math text using matplotlib.mathtext
        """
        ox, oy, width, height, descent, font_image, used_characters = \
            self.mathtext_parser.parse(s, self.dpi, prop)

        xd = descent * sin(radians(angle))
        yd = descent * cos(radians(angle))
        x = numpy.round(x + ox + xd)
        y = numpy.round(y - oy + yd)
        self.draw_text_image(font_image, x, y + 1, angle, gc)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if ismath:
            return self.draw_mathtext_f2font(gc, x, y, s, prop, angle)

        return self.draw_text_f2font(gc, x, y, s, prop, angle)

    def flipy(self):
        """
        Return true if y-direction goes top-down for renderer.
        Is used for drawing text (:mod:`matplotlib.text`) and images
        (:mod:`matplotlib.image`) only.
        """
        return True

    def get_canvas_width_height(self):
        # necessary for matplotlib to calculate data to canvas transform
        # for example to position tick markers
        return self.width, self.height

    def points_to_pixels(self, points):
        # if backend doesn't have dpi, e.g., postscript or svg
        return points * self.dpi / 72


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
                self.renderer = RendererQtGL(self, w, h, self.figure.dpi)
                self.available.emit()

        def initializeGL(self):
            glEnable(GL_LINE_STIPPLE)
            # enable antialiasing

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

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
            return RendererQtGL(self, self.width(), self.height(), self.figure.dpi)
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
