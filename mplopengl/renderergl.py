"""
@author: Karl Royen
"""
from builtins import NotImplementedError
from functools import lru_cache
from math import sin, radians, cos, log2
from random import randrange

import matplotlib
import numpy
from OpenGL.GL import *
from OpenGL.arrays import vbo
from matplotlib import rcParams
from matplotlib.backend_bases import RendererBase
from matplotlib.backends import backend_agg
from matplotlib.font_manager import findfont, get_font
from matplotlib.mathtext import MathTextParser
from matplotlib.path import get_paths_extents
from matplotlib.transforms import Affine2D

MIN_LINE_WIDTH = 1.5


class Context:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self.set_context(*self.args, **self.kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        self.clean_context()

    def set_context(self, *args, **kwargs):
        raise NotImplementedError()

    def clean_context(self):
        pass


def get_fill_color(gc, rgb_face):
    if gc.get_forced_alpha():
        fillopacity = gc.get_alpha()
    else:
        fillopacity = rgb_face[3] if len(rgb_face) > 3 else 1.0

    return [*rgb_face[:3], fillopacity]


class FilledContext(Context):
    def set_context(self, gc, rgb_face):
        glColor4fv(get_fill_color(gc, rgb_face))


def get_stroke_color(gc, renderer):
    width = renderer.points_to_pixels(gc.get_linewidth())

    if gc.get_forced_alpha():
        strokeopacity = gc.get_alpha()
    else:
        strokeopacity = gc.get_rgb()[3]

    col = gc.get_rgb()[:3]
    # minimum line width is 1.5 for compatibilty with old and/or intel hardware
    # thinner lines get alpha faded

    if width < MIN_LINE_WIDTH:
        strokeopacity *= width / MIN_LINE_WIDTH

    return [*col, strokeopacity]


class StrokeColorContext(Context):
    def set_context(self, gc, renderer):
        glColor4fv(get_stroke_color(gc, renderer))


def linestipple_from_style(offset, dashes):
    if dashes is None:
        return 1, int('1111111111111111', 2)

    return _cached_linestipple_from_style(offset, tuple(dashes))


@lru_cache(10)
def _cached_linestipple_from_style(offset, dashes):
    """
    Create a line stipple pattern from matplotlibs offset dash list.

    This probably could be done much better/simple, but for now it gets
    the job done.
    """
    # print("Input: ", offset, dashes)
    length = numpy.sum(dashes)
    # rescale to length of multiple of 2 but maximum of length 16
    order = int(numpy.ceil(log2(length)))
    if order > 4:
        fac = 2 ** (order - 4)
        order = 4
    else:
        fac = 1

    length2 = 2 ** order
    style = numpy.multiply(dashes, length2 / length)
    offset *= length2 / length

    # transform to pixel units
    pix_style = numpy.zeros(len(style), dtype=numpy.int)
    for i in range(len(style)):
        pix_sum = sum(pix_style[:i])
        style_sum = sum(style[:i + 1])
        pix_style[i] = numpy.round(style_sum) - pix_sum
    pix_offset = int(round(offset))
    # print(style, pix_style)

    # create pattern
    pattern = []
    val = 1
    while len(pattern) < 16:
        for l in pix_style:
            pattern.extend([str(val % 2)] * l)
            val += 1

    # apply offset to pattern
    numpy.roll(pattern, -pix_offset)
    pattern = ''.join(pattern)
    # print(pattern)
    return fac, int(pattern, 2)


class StrokedContext(Context):
    def set_context(self, gc, renderer):
        glPushAttrib(GL_LINE_BIT)
        width = renderer.points_to_pixels(gc.get_linewidth())

        if width < MIN_LINE_WIDTH:
            glLineWidth(MIN_LINE_WIDTH)
        else:
            glLineWidth(width)

        offset, style = gc.get_dashes()
        fac, pattern = linestipple_from_style(offset, style)
        glLineStipple(fac, pattern)

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


class VBO:
    def __init__(self, arr_data):
        if not isinstance(arr_data, bytes):
            arr_data = numpy.array(arr_data).astype(numpy.float32).tobytes()

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # arr_data = numpy.array(polygons).flatten()
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(arr_data),
                     arr_data, GL_STATIC_DRAW)

    def bind_to(self, location=None):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        if location is None:
            glVertexPointer(2, GL_FLOAT, 0, None)
        else:
            glVertexAttribPointer(location, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)


class PolygonVBO(VBO):
    def __init__(self, polygons, arr_data=None):
        if arr_data is None:
            super().__init__(polygons)
        else:
            super().__init__(arr_data)

        self._poly_lens = [len(polygon) for polygon in polygons]

    def poly_sizes(self):
        return self._poly_lens


class PolygonVBOContext(Context):
    def set_context(self, vbo):
        glEnableClientState(GL_VERTEX_ARRAY)
        vbo.bind_to()

        return vbo.poly_sizes()

    def clean_context(self):
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)


class Texture:
    def __init__(self, im, cached=False, grey_level=False):
        texture_id = glGenTextures(1)
        # this is necessary for images with resolution not a multiple of two
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)

        if grey_level:
            swizzle_mask = [GL_ONE, GL_ONE, GL_ONE, GL_RED]
            glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle_mask)
            input_type = GL_RED
        else:
            input_type = GL_RGBA

        # if no interpolation is chosen we shouldn't "invent" datapoints by interpolation
        # when magnifying
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, im.shape[1], im.shape[0],
                     0, input_type, GL_UNSIGNED_BYTE, im)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.tex_id = texture_id
        self.size = numpy.array([im.shape[1], im.shape[0]])
        self._cached = cached

    def draw_at(self, x, y, angle=0, transform=None, col=None):
        if col is None:
            col = [1, 1, 1]

        if transform is None:
            x0 = [0, 0]
            x1 = self.size
        else:
            x0 = transform.transform([0, 0])
            x1 = transform.transform([1, 1])

        glPushMatrix()
        glTranslatef(x, y, 0)
        glRotatef(angle, 0, 0, 1)

        glColor3fv(col)
        glBegin(GL_QUADS)
        glTexCoord2f(1, 0)
        glVertex2f(x1[0], x0[1])
        glTexCoord2f(1, 1)
        glVertex2f(x1[0], x1[1])
        glTexCoord2f(0, 1)
        glVertex2f(x0[0], x1[1])
        glTexCoord2f(0, 0)
        glVertex2f(x0[0], x0[1])

        glEnd()

        glPopMatrix()


class TextureContext(Context):
    def set_context(self, texture, keep=False):
        self.tex_id = texture.tex_id
        self._keep = keep
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glEnable(GL_TEXTURE_2D)

    def clean_context(self):
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        if not self._keep:
            glDeleteTextures([self.tex_id])


class TextTexture(Texture):
    mathtext_parser = MathTextParser('Agg')

    @staticmethod
    def _text_image(text, prop, dpi):
        font = TextTexture._get_agg_font(prop, dpi)

        flags = backend_agg.get_hinting_flag()
        if len(text) == 1 and ord(text) > 127:
            font.load_char(ord(text), flags=flags)
        else:
            # We pass '0' for angle here, since it will be rotated (in raster
            # space) in the following call to draw_text_image).
            font.set_text(text, 0, flags=flags)
        font.draw_glyphs_to_bitmap(antialiased=rcParams['text.antialiased'])
        d = font.get_descent() / 64.0
        # The descent needs to be adjusted for the angle.
        xo, yo = font.get_bitmap_offset()
        xo /= 64.0
        yo /= 64.0

        return font.get_image(), xo, yo + 1, d

    @staticmethod
    def _math_tex_image(text, prop, dpi):
        ox, oy, width, height, descent, font_image, used_characters = \
            TextTexture.mathtext_parser.parse(text, dpi, prop)
        return font_image.as_array(), ox, -oy + 1, descent

    @staticmethod
    def _get_agg_font(prop, dpi):
        """
        Get the font for text instance t, caching for efficiency
        """
        fname = findfont(prop)
        font = get_font(fname)

        font.clear()
        size = prop.get_size_in_points()
        font.set_size(size, dpi)

        return font

    def __init__(self, text, prop, ismath, dpi):
        if ismath:
            image, ox, oy, descent = self._math_tex_image(text, prop, dpi)
        else:
            image, ox, oy, descent = self._text_image(text, prop, dpi)

        self.offset = numpy.array([ox, -oy])
        self.descent = descent

        super().__init__(image[::-1, :], grey_level=True)

    def draw_at(self, x, y, angle, **kwargs):
        angle_off = self.descent * numpy.array([sin(radians(angle)), -cos(radians(angle))])
        pos = numpy.round(numpy.array([x, y]) + self.offset + angle_off)
        super().draw_at(*pos, angle=angle, **kwargs)


class TextTextureContext(Context):

    def set_context(self, image):
        glBindTexture(GL_TEXTURE_2D, image.tex_id)
        glEnable(GL_TEXTURE_2D)

    def clean_context(self):
        glDisable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)


class GPUObjectCache:
    def __init__(self, max_objects=None):
        self._cache = {}
        self._limit = max_objects
        if self._limit is not None:
            self._obj_stack = []

    def fetch(self, context, hash_value, factory, *args, **kwargs):
        if context not in self._cache:
            self._cache[context] = {}
        cache = self._cache[context]

        # check if object is already in cache and return it from there
        if hash_value in cache:
            if self._limit is not None:
                # push object to back of list
                # (front of list is destroyed first if cache is full)
                i = self._obj_stack.index((hash_value, context))
                self._obj_stack.append(self._obj_stack.pop(i))
            return cache[hash_value]

        # create the object and add it to cache
        obj = factory(*args, **kwargs)
        if self._limit is not None:
            # if necessary make room for the new object by
            # destroying old cached objects
            while len(self._obj_stack) >= self._limit:
                o, context = self._obj_stack.pop(0)
                tmp_cache = self._cache[context]
                del tmp_cache[o]
            self._obj_stack.append((hash_value, context))
        cache[hash_value] = obj
        return obj

    def __call__(self, context, hash_value, factory, *args, **kwargs):
        return self.fetch(context, hash_value, factory, *args, **kwargs)


vertexShaderSource = """#version 120
attribute vec2 pos;
attribute vec2 shift;
uniform mat3 trans;
void main()
{
    gl_Position = gl_ModelViewProjectionMatrix * vec4((trans*vec3(shift,1)).xy+pos, 0.0, 1.0);
}"""

fragmentShaderSource = """#version 120
uniform vec4 color;
void main()
{
    gl_FragColor = color;
}"""


def create_shader(shader_type, source):
    """compile a shader."""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader


class Shader:
    def __init__(self, vertex_source, fragment_source, attrs, uniforms):
        vert_shader = create_shader(GL_VERTEX_SHADER, vertex_source)
        frag_shader = create_shader(GL_FRAGMENT_SHADER, fragment_source)

        program = glCreateProgram()
        glAttachShader(program, vert_shader)
        glAttachShader(program, frag_shader)

        for attr, loc in attrs.items():
            glBindAttribLocation(program, loc, attr)

        glLinkProgram(program)
        glValidateProgram(program)
        if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
            log = glGetProgramInfoLog(program)
            if log:
                raise RuntimeError(log)

        self.uniforms = {}
        for uniform in uniforms:
            self.uniforms[uniform] = glGetUniformLocation(program, uniform)

        self.program = program
        self.attrs = attrs

    def bind(self):
        glUseProgram(self.program)
        for attr in self.attrs.values():
            glEnableVertexAttribArray(attr)

    def unbind(self):
        for attr in self.attrs.values():
            glDisableVertexAttribArray(attr)
        glUseProgram(0)

    def set_uniform4f(self, uniform, *args):
        loc = self.uniforms[uniform]
        glUniform4f(loc, *args)

    def set_uniform3m(self, uniform, value, transpose=False):
        loc = self.uniforms[uniform]
        glUniformMatrix3fv(loc, 1, transpose, value.astype(numpy.float32))

    def set_attr_divisor(self, attr, div):
        glVertexAttribDivisor(self.attrs[attr], div)

    def bind_attr_vbo(self, attr, vbo):
        loc = self.attrs[attr]
        vbo.bind_to(loc)


class ObjectContext(Context):
    def set_context(self, obj, *args, **kwargs):
        self.obj = obj
        obj.bind(*args, **kwargs)
        return obj

    def clean_context(self):
        self.obj.unbind()


class RendererGL(RendererBase):

    def __init__(self, width, height, dpi):
        super().__init__()
        self.width = width
        self.height = height
        self.dpi = dpi
        self.context = randrange(9999999999)
        self._gpu_cache = GPUObjectCache(200)
        self._image_cache = GPUObjectCache(10)

        self.particle_shader = Shader(vertexShaderSource, fragmentShaderSource,
                                      {"pos": 0, "shift": 4},
                                      ["color", "trans"])

    def option_image_nocomposite(self):
        return False

    def option_scale_image(self):
        return True

    def draw_gouraud_triangles(self, gc, triangles_array, colors_array,
                               transform):
        """
        Draws a series of Gouraud triangles.

        Parameters
        ----------
        triangles_array : array_like, shape=(N, 3, 2)
            Array of *N* (x, y) points for the triangles.

        colors_array : array_like, shape=(N, 3, 4)
            Array of *N* RGBA colors for each point of the triangles.

        transform : `matplotlib.transforms.Transform`
            An affine transform to apply to the points.
        """
        vertex_vbo = vbo.VBO(numpy.array(triangles_array, dtype=numpy.float32), usage=GL_STATIC_DRAW)
        vertex_vbo.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, vertex_vbo)

        color_vbo = vbo.VBO(numpy.array(colors_array, dtype=numpy.float32), usage=GL_STATIC_DRAW)
        color_vbo.bind()
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4, GL_FLOAT, 0, color_vbo)

        A = transform.get_matrix()
        B = numpy.identity(4)
        B[:2, :2] = A[:2, :2]
        B[:2, 3] = A[:2, 2]
        glPushMatrix()
        glMultMatrixf(B.astype(numpy.float32).transpose())
        with ClippingContext(gc):
            glDrawArrays(GL_TRIANGLES, 0, len(triangles_array) * 3)
        glPopMatrix()
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

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
        # apply transform
        A = transform.get_matrix()
        B = numpy.identity(4)
        B[:2, :2] = A[:2, :2]
        B[:2, 3] = A[:2, 2]
        glPushMatrix()
        glMultMatrixf(B.astype(numpy.float32).transpose())

        with ClippingContext(gc):
            glBegin(GL_TRIANGLES)
            for i in range(3):
                glColor4fv(colors[i, :])
                glVertex2fv(points[i, :])
            glEnd()
        glPopMatrix()

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
        positions = path.vertices
        if len(positions) == 1:  # single markers don't need a particle shader to draw
            positions = trans.transform(positions)
            translation = Affine2D().translate(*positions[0])
            return self.draw_path(gc, marker_path, marker_trans + translation, rgbFace)

        marker_path = marker_trans.transform_path(marker_path)
        polygons = self.path_to_poly(marker_path, rgbFace is not None)

        arr_data = numpy.array(positions).astype(numpy.float32).tobytes()
        pos_vbo = self._gpu_cache(self.context, hash(arr_data), VBO, arr_data)

        with ObjectContext(self.particle_shader) as program, ClippingContext(gc):
            program.bind_attr_vbo("shift", pos_vbo)
            program.set_uniform3m("trans", trans.get_matrix(), transpose=True)
            program.set_attr_divisor("pos", 0)
            program.set_attr_divisor("shift", 1)

            for polygon in polygons:
                arr_data = numpy.array(polygon).astype(numpy.float32).tobytes()
                poly_vbo = self._gpu_cache(self.context, hash(arr_data), VBO, arr_data)
                program.bind_attr_vbo("pos", poly_vbo)

                if rgbFace is not None and len(polygon) >= 3:
                    col = get_fill_color(gc, rgbFace)
                    program.set_uniform4f("color", *col)
                    glDrawArraysInstanced(GL_POLYGON, 0, len(polygon) - 1, len(positions))

                if gc.get_linewidth() > 0:
                    with StrokedContext(gc, self):
                        col = get_stroke_color(gc, self)
                        program.set_uniform4f("color", *col)
                        glDrawArraysInstanced(GL_LINE_STRIP, 0, len(polygon), len(positions))

    @staticmethod
    def path_to_poly(path, closed=False):
        polys = path.to_polygons(closed_only=False)

        # matplotlibs to_polygons does not always handle closing of polygons correctly
        # concrete it seems to have problems with closed triangles
        if closed:
            poly2 = []
            for poly in polys:
                if len(poly) > 2 and numpy.any(poly[0] != poly[-1]):
                    poly2.append(numpy.append(poly, [poly[0]], axis=0))
                else:
                    poly2.append(poly)
            return poly2
        return polys

    def draw_dash(self, gc, polygon):
        with ClippingContext(gc), StrokedContext(gc, self):
            col = get_stroke_color(gc, self)
            glColor4fv(col)
            glBegin(GL_LINES)
            glVertex2fv(polygon[0])
            glVertex2fv(polygon[1])
            glEnd()

    # noinspection PyPep8Naming
    def draw_path(self, gc, path, transform, rgbFace=None):
        path = transform.transform_path(path)
        polygons = self.path_to_poly(path, rgbFace is not None)

        # keep it simple (no VBOs) for a single dash (e.g. tick marks...)
        if len(polygons) == 1 and len(polygons[0]) == 2:
            return self.draw_dash(gc, polygons[0])

        arr_data = numpy.array(polygons).astype(numpy.float32).tobytes()
        poly_vbo = self._gpu_cache(self.context, hash(arr_data), PolygonVBO, polygons, arr_data)
        with PolygonVBOContext(poly_vbo) as polygons, ClippingContext(gc):
            if rgbFace is not None:
                with FilledContext(gc, rgbFace):
                    offset = 0
                    for num_vertices in polygons:
                        # the returned polygons are closed (first and last vertex are the same)
                        # so we only need N-1 points for the polygon
                        glDrawArrays(GL_POLYGON, offset, num_vertices - 1)
                        offset += num_vertices

            if gc.get_linewidth() > 0:
                with StrokedContext(gc, self), StrokeColorContext(gc, self):
                    offset = 0
                    for num_vertices in polygons:
                        glDrawArrays(GL_LINE_STRIP, offset, num_vertices)
                        offset += num_vertices

    # draw_path_collection is optional, and we get more correct
    # relative timings by leaving it out. backend implementers concerned with
    # performance will probably want to implement it
    # def draw_path_collection(self, gc, master_transform, paths,
    #                          all_transforms, offsets, offsetTrans,
    #                          facecolors,
    #                          edgecolors, linewidths, linestyles,
    #                          antialiaseds):
    #     print("collection")
    #     super().draw_path_collection(gc, master_transform, paths,
    #                                  all_transforms, offsets, offsetTrans,
    #                                  facecolors,
    #                                  edgecolors, linewidths, linestyles,
    #                                  antialiaseds)

    # draw_quad_mesh is optional, and we get more correct
    # relative timings by leaving it out.  backend implementers concerned
    # with performance will probably want to implement it
    def draw_quad_mesh(self, gc, master_transform, meshWidth,
                       meshHeight,
                       coordinates, offsets, offsetTrans, facecolors,
                       antialiased, edgecolors):

        # apply transform
        A = master_transform.get_matrix()
        B = numpy.identity(4)
        B[:2, :2] = A[:2, :2]
        B[:2, 3] = A[:2, 2]
        glPushMatrix()
        glMultMatrixf(B.astype(numpy.float32).transpose())

        # render quads
        with ClippingContext(gc):
            glBegin(GL_QUADS)
            for row in range(meshHeight):
                for quad in range(meshWidth):
                    glColor4fv(facecolors[row * meshWidth + quad])
                    glVertex2fv(coordinates[row, quad])
                    glVertex2fv(coordinates[row + 1, quad])
                    glVertex2fv(coordinates[row + 1, quad + 1])
                    glVertex2fv(coordinates[row, quad + 1])
            glEnd()
        glPopMatrix()

    def bind_image(self, im):
        # TODO: seems this method is never used, remove it?

        texture_id = glGenTextures(1)
        # gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, im.shape[1], im.shape[0],
                     0, GL_RGBA, GL_UNSIGNED_BYTE, im.tobytes())
        glEnable(GL_TEXTURE_2D)
        return texture_id

    def unbind_image(self, texture_id):
        # TODO: seems this method is never used, remove it?

        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([texture_id])

    def draw_image(self, gc, x, y, im, transform=None):
        tex = self._image_cache.fetch(self.context, hash(im.tobytes()),
                                      Texture, im)
        with TextureContext(tex, keep=True), ClippingContext(gc):
            tex.draw_at(x, y, transform=transform)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        text_image = TextTexture(s, prop, ismath, self.dpi)
        # text_image = self._gpu_cache(self.context, hash((s, prop, ismath, self.dpi, TextTexture)),
        #                              TextTexture, s, prop, ismath, self.dpi)
        with TextTextureContext(text_image):
            text_image.draw_at(x, self.height - y, angle=angle, col=gc.get_rgb()[:3])

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
