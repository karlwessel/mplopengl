from logging import warning
from math import sin, radians, cos

import matplotlib
import numpy
from OpenGL.GL import glBegin, glColor4fv, glVertex2fv, glEnd, glVertexPointer, glColor3fv, glGenTextures, \
    glTexImage2D, \
    glDeleteTextures, GL_TEXTURE_2D, GL_TRIANGLES, GL_QUADS, GL_SCISSOR_TEST, GL_VERTEX_ARRAY, glPushAttrib, \
    glPopAttrib, glPushMatrix, \
    glTranslatef, glPopMatrix, glEnable, glScissor, glBlendFunc, glColor4f, glVertex2f, \
    glLineWidth, glLineStipple, glTexParameteri, glDisable, glRotatef, glTexCoord2f, glColor3f, GL_SCISSOR_BIT, \
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, \
    glEnableClientState, glDrawArrays, GL_POLYGON, glDisableClientState, GL_LINE_BIT, GL_LINE_STRIP, glBindTexture, \
    GL_RGBA, GL_COLOR_MATERIAL, GL_TEXTURE_BASE_LEVEL, GL_TEXTURE_MAX_LEVEL, GL_DOUBLE, GL_UNSIGNED_BYTE, \
    glBindBuffer, GL_ARRAY_BUFFER, ArrayDatatype, glBufferData, GL_STATIC_DRAW, glGenBuffers
from matplotlib import rcParams
from matplotlib.backend_bases import RendererBase
from matplotlib.backends import backend_agg
from matplotlib.font_manager import findfont, get_font
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path, get_paths_extents
from matplotlib.transforms import Affine2D

MIN_LINE_WIDTH = 1.5


class Context:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        self.set_context(*self.args, **self.kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        self.clean_context()

    def clean_context(self):
        pass


class FilledContext(Context):
    def set_context(self, gc, rgb_face):
        if gc.get_forced_alpha():
            fillopacity = gc.get_alpha()
        else:
            fillopacity = rgb_face[3] if len(rgb_face) > 3 else 1.0

        glColor4f(*rgb_face[:3], fillopacity)


class StrokeColorContext(Context):
    def set_context(self, gc, renderer):
        width = renderer.points_to_pixels(gc.get_linewidth())

        if gc.get_forced_alpha():
            strokeopacity = gc.get_alpha()
        else:
            strokeopacity = gc.get_rgb()[3]

        col = gc.get_rgb()[:3]
        # minimum line width is 1.5 for compatibilty with old and/or intel hardware
        # thinner lines get alpha faded

        if width < MIN_LINE_WIDTH:
            glColor4f(*col, strokeopacity * width / MIN_LINE_WIDTH)
        else:
            if strokeopacity < 1:
                glColor4f(*col, strokeopacity)
            else:
                glColor3fv(col)


class StrokedContext(Context):
    def set_context(self, gc, renderer):
        glPushAttrib(GL_LINE_BIT)
        width = renderer.points_to_pixels(gc.get_linewidth())

        if width < MIN_LINE_WIDTH:
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

    def clean_context(self):
        glPopAttrib()

class ClippingContext(Context):
    def set_context(self, gc):
        glPushAttrib(GL_SCISSOR_BIT)
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

    def clean_context(self):
        glPopAttrib()


class RendererGL(RendererBase):

    def __init__(self, width, height, dpi):
        super().__init__()
        self.width = width
        self.height = height
        self.dpi = dpi
        self.mathtext_parser = MathTextParser('Agg')
        self._vbos = {}

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
        with ClippingContext(gc):
            transform.transform(points)
            glBegin(GL_TRIANGLES)
            for i in range(3):
                glColor4fv(colors[i])
                glVertex2fv(points[i])
            glEnd()


    def buffer(self, arr_data):
        key = hash(arr_data)
        if key in self._vbos:
            return self._vbos[key]

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        # arr_data = numpy.array(polygons).flatten()
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(arr_data),
                     arr_data, GL_STATIC_DRAW)
        self._vbos[key] = vbo
        return vbo

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
        # disp_list = self.marker_list(gc, marker_path, marker_trans, rgbFace)

        marker_path = marker_trans.transform_path(marker_path)
        polygons = self.path_to_poly(marker_path)

        vbo = self.buffer(numpy.array(polygons).tobytes())
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_DOUBLE, 0, None)

        positions = [vertices[-2:] for vertices, _ in path.iter_segments(trans, simplify=False)]

        with ClippingContext(gc):
            if rgbFace is not None:
                with FilledContext(gc, rgbFace):
                    self.repeat_primitive(GL_POLYGON, polygons, positions)

            if gc.get_linewidth() > 0:
                with StrokeColorContext(gc, self), StrokedContext(gc, self):
                    self.repeat_primitive(GL_LINE_STRIP, polygons, positions)

        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def repeat_primitive(self, primitive, polygons, positions):
        for x, y in positions:
            glPushMatrix()
            glTranslatef(x, y, 0)
            offset = 0
            for polygon in polygons:
                num_vertices = len(polygon)
                glDrawArrays(primitive, offset, num_vertices)
                offset += num_vertices
            glPopMatrix()

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

    def draw_polys_filled(self, polygons):
        for polygon in polygons:
            if len(polygon) > 2:
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(2, GL_DOUBLE, 0, polygon)
                glDrawArrays(GL_POLYGON, 0, len(polygon))
                glDisableClientState(GL_VERTEX_ARRAY)

    def draw_polys_stroked(self, polygons):
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

    # noinspection PyPep8Naming
    def draw_path(self, gc, path, transform, rgbFace=None):
        path = transform.transform_path(path)
        polygons = self.path_to_poly(path)

        with ClippingContext(gc):
            if rgbFace is not None:
                with FilledContext(gc, rgbFace):
                    self.draw_polys_filled(polygons)

            if gc.get_linewidth() > 0:
                with StrokedContext(gc, self), StrokeColorContext(gc, self):
                    self.draw_polys_stroked(polygons)


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
