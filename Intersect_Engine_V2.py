try:
    import threading
    import json
    from typing import Any
    import numpy as np
    from pathlib import Path
    import moderngl_window as mglw
    from pyglet import image
    from pyglet.window import key, mouse
    from PIL import Image
    from random import random, randint
    import math
    import os
except ImportError as e:
    print("Intersect Engine requires the following modules which arent provided by Python: numpy, moderngl_window, pyglet, and PIL")
    raise

# Globals
com = 8765
registry = {}
gs = [800, 600] #global scale since to prevent vertex stretch

class IntersectEngine(mglw.WindowConfig):
    window_size = (gs[0], gs[1])
    title = "Intersect Engine"
    resource_dir = '.'
    aspect_ratio = None
    resizable = True
    keys = True

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key == "window_size":
                    gs[0] = value[0]
                    gs[1] = value[1]
                setattr(self, key, value)
        super().__init__(**kwargs)
        self._setup_icon()
        self._setup_shader()
        self.meshes = {}  # key = object_id, value = (vao, count, mode, texture, color)
        self.dynamic_proj = None
        self.static_proj = None
        self.command_queue = []
        self.lock = threading.Lock()
        self._update_projection()
        self._texture_cache = {}
        self.current_meshes = []

        self.default_homography = np.identity(3, dtype='f4')

        self._key_state = set()
        self._key_callbacks = {}

        try:
            self.wnd._window.push_handlers(
                on_key_press=self._on_key_press,
                on_key_release=self._on_key_release,
                on_mouse_motion=self._on_mouse_motion,
                on_mouse_press=self._on_mouse_press,
                on_mouse_release=self._on_mouse_release
            )
        except Exception as e:
            print("⚠️ Input Hook Error:", e)

    def register_key_action(self, symbol, press=None, release=None, mode="single"):
        self._key_callbacks[symbol] = {"press": press, "release": release, "mode": mode}

    def unregister_key_action(self, symbol):
        self._key_callbacks.pop(symbol, None)

    def _on_key_press(self, symbol, modifiers):
        if symbol in self._key_callbacks:
            cb = self._key_callbacks[symbol]
            if cb["mode"] == "single":
                if cb["press"]:
                    cb["press"]()
            elif cb["mode"] == "hold":
                if symbol not in self._key_state:
                    if cb["press"]:
                        cb["press"]()
                self._key_state.add(symbol)

    def _on_mouse_press(self, x, y, symbol, modifiers):
        if symbol in self._key_callbacks:
            cb = self._key_callbacks[symbol]
            if cb["mode"] == "single":
                if cb["press"]:
                    cb["press"]()
            elif cb["mode"] == "hold":
                if symbol not in self._key_state:
                    if cb["press"]:
                        cb["press"]()
                self._key_state.add(symbol)

    def _on_key_release(self, symbol, modifiers):
        if symbol in self._key_state:
            self._key_state.remove(symbol)
        cb = self._key_callbacks.get(symbol)
        if cb and cb["release"]:
            cb["release"]()

    def _on_mouse_release(self, x, y, symbol, modifiers):
        if symbol in self._key_state:
            self._key_state.remove(symbol)
        cb = self._key_callbacks.get(symbol)
        if cb and cb["release"]:
            cb["release"]()

    def _on_mouse_motion(self, x, y, dx, dy):
        pass

    def _setup_icon(self):
        try:
            icon_path = getattr(self, "icon_path", "C:/Users/bendy/Desktop/INTERSECT Files/ICOs/Intersect Logo.ico")
            icon_path = Path(icon_path)
            if icon_path.exists():
                icon = image.load(str(icon_path))
                self.wnd._window.set_icon(icon)
        except Exception as e:
            print("⚠️ Icon failed:", e)

    def _setup_shader(self):
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec2 in_uv;
                uniform mat4 modelViewProj;
                uniform mat3 homography;
                out vec2 v_uv;

                void main() {
                    gl_Position = modelViewProj * vec4(in_pos, 1.0);
                    vec3 uv = homography * vec3(in_uv, 1.0);
                    v_uv = uv.xy / uv.z;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D tex;
                uniform vec4 use_color;
                in vec2 v_uv;
                out vec4 fragColor;

                void main() {
                    vec4 tex_color = texture(tex, v_uv);
                    fragColor = mix(tex_color, use_color, use_color.a > 0.0 ? 1.0 : 0.0);
                }
            """
        )

    def resize(self, width: int, height: int):
        self._update_projection(width, height)

    def on_resize(self, width: int, height: int):
        self._update_projection(width, height)


    def _update_projection(self, width=None, height=None):
        if width is None:
            width = self.window_size[0]
        if height is None:
            height = self.window_size[1]

        self.height = self.wnd.height
        self.width = self.wnd.width

        self.dynamic_proj = np.array([
            [2 / width, 0, 0, -1],
            [0, 2 / height, 0, -1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype='f4')

        self.static_proj = np.identity(4, dtype='f4')

    def _generate_uvs(self, vertices):
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)

        width = max_x - min_x
        height = max_y - min_y
        width = width if width != 0 else 1
        height = height if height != 0 else 1

        return [[(v[0] - min_x) / width, (v[1] - min_y) / height] for v in vertices]

    def _create_test_texture(self, name="autogen_texture.png"):
        path = Path(name)
        if not path.exists():
            color = tuple(randint(50, 200) for _ in range(3)) + (255,)
            img = Image.new("RGBA", (64, 64), color)
            img.save(path)
        return name

    def set_mesh(self, object_id, faces, uvs=None, texture=None, color=None):
        """Create VAOs for each face and store them using a unique id."""
        for face_id, verts in faces.items():
            mesh_id = f"{object_id}:{face_id}"
            vertices = np.asarray(verts, dtype=np.float32)

            face_uvs = None
            if isinstance(uvs, dict):
                face_uvs = uvs.get(face_id)
            elif uvs is not None:
                face_uvs = uvs

            face_color = None
            if isinstance(color, dict):
                face_color = color.get(face_id)
            else:
                face_color = color

            if texture is None and face_color is None:
                face_color = (random(), random(), random(), 1.0)

            vertices = self.coord_scaling(vertices, self.wnd.width, self.wnd.height)

            if face_uvs is None:
                face_uvs = self._generate_uvs(vertices.tolist())

            vbo_data = vertices.astype('f4').tobytes()

            if isinstance(face_uvs, (list, np.ndarray)):
                try:
                    face_uvs = np.array(face_uvs, dtype='f4')
                    uvbo_data = face_uvs.tobytes()
                    vbo = self.ctx.buffer(vbo_data)
                    uvbo = self.ctx.buffer(uvbo_data)
                    vao = self.ctx.vertex_array(self.prog, [
                        (vbo, '3f', 'in_pos'),
                        (uvbo, '2f', 'in_uv'),
                    ])
                except Exception as e:
                    print(f"⚠️ Invalid UVs for {mesh_id}: {face_uvs} →", e)
                    vbo = self.ctx.buffer(vbo_data)
                    vao = self.ctx.vertex_array(self.prog, [
                        (vbo, '3f', 'in_pos'),
                    ])
            else:
                vbo = self.ctx.buffer(vbo_data)
                vao = self.ctx.vertex_array(self.prog, [
                    (vbo, '3f', 'in_pos'),
                ])

            tex_obj = None
            if texture and face_color is None:
                try:
                    if texture not in self._texture_cache:
                        self._texture_cache[texture] = self.load_texture_2d(texture)
                    tex_obj = self._texture_cache[texture]
                except Exception as e:
                    print(f"⚠️ Texture Load Failed ({texture}):", e)

            self.meshes[mesh_id] = (vao, len(vertices), self.ctx.TRIANGLE_FAN, tex_obj, face_color)

    def remove_mesh(self, object_id):
        if object_id in self.meshes:
            del self.meshes[object_id]

    def on_render(self, time, frame_time):
        self.ctx.clear(0.05, 0.05, 0.05)

        with self.lock:
            ids = list(registry.keys())
            for i in ids:
                self.set_mesh(i, registry[i].faces, registry[i].uvs, registry[i].texture, registry[i].colour)
            for mesh_id in list(self.meshes.keys()):
                element_id = mesh_id.split(":", 1)[0]
                try:
                    element_id = int(element_id)
                except ValueError:
                    pass
                if element_id not in registry.keys():
                    self.remove_mesh(mesh_id)

        for mesh_id, (vao, count, mode, texture, color) in self.meshes.items():
            oid = mesh_id.split(":", 1)[0]
            try:
                oid_int = int(oid)
            except ValueError:
                oid_int = oid
            self.prog["modelViewProj"].write(self.static_proj.tobytes())
            self.prog["tex"].value = 0
            try:
                self.prog["use_color"].value = tuple(color)
            except Exception as e:
                print(f"⚠️ Color assignment failed for {mesh_id}: {color} →", e)
                self.prog["use_color"].value = (random(), random(), random(), 1.0)

            element = self.get_element(oid_int)
            if element:
                H = element.get_homography_matrix()
                self.prog["homography"].write(H.tobytes())

            if texture:
                texture.use(location=0)
            vao.render(mode=mode, vertices=count)

    def get_all_elements(self):
        return list(registry.keys())

    def get_element(self, id):
        return registry.get(id)

    def set_element_homography(self, element_id, matrix3x3):
        for mesh_id in self.meshes:
            if str(mesh_id).startswith(f"{element_id}:"):
                self.default_homography = matrix3x3.astype('f4')
                break

    def coord_scaling(self, vertices, width, height):
        for i in range(len(vertices)):
            vertices[i][0] = (0 - (width / 2) + vertices[i][0]) / width * 2
            vertices[i][1] = (((height - vertices[i][1]) - (height / 2)) / height) * 2
        return vertices

def start(app_class):
    mglw.run_window_config(app_class)


class Element:
    STRUCTURE = "01"
    METADATA = "02"
    TRANSFORM = "03"
    RENDER = "04"

    def __init__(self, id=None, iee_filepath=None):
        self.id = id
        self.faces = {}
        self.collisions = []
        self.pos = [0, 0, 0]
        self.rot = [0, 0, 0]
        self.uvs = {}
        self.colour = {}  # now a dict per-face
        self.texture = None
        if id is None:
            id = 0
            while id in registry:
                id += 1
            self.id = id
        registry[self.id] = self
        if iee_filepath:
            self.parse_iee(iee_filepath)

    def safe_floats(self, parts):
        try:
            return [float(p) for p in parts]
        except ValueError:
            return []

    def parse_iee(self, path):
        if not path.lower().endswith((".iee", ".ie_e")):
            print(f"Unsupported file: {path}")
            return

        with open(path) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue

                args = line.split()
                if len(args) < 2:
                    continue

                prefix = args[0]

                if prefix == self.STRUCTURE:
                    if args[1] == "f":
                        face_id = args[-1]
                        self.faces[face_id] = []
                        for y in args[2:-1]:
                            yt = y.split("/")
                            vx = float(yt[0])
                            vy = float(yt[1]) if len(yt) > 1 and yt[1] else -1
                            vz = float(yt[2]) if len(yt) > 2 and yt[2] else -1
                            self.faces[face_id].append([vx, vy, vz])
                    elif args[1] == "c":
                        for y in args[2:]:
                            yt = y.split("/")
                            point = self.safe_floats(yt)
                            if point:
                                self.collisions.append(point)

                elif prefix == self.METADATA:
                    setattr(self, args[1], args[2])

                elif prefix == self.TRANSFORM:
                    if args[1] == "pos":
                        self.pos = self.safe_floats(args[2].split("/"))
                    elif args[1] == "rot":
                        self.rot = self.safe_floats(args[2].split("/"))

                elif prefix == self.RENDER:
                    if args[1] == "uv":
                        face_id = args[-1]
                        self.uvs[face_id] = []
                        for y in args[2:-1]:
                            yt = self.safe_floats(y.split("/"))
                            if yt:
                                self.uvs[face_id].append(yt)
                    elif args[1] in ["colour", "color"]:
                        face_id = args[-1]
                        method = args[2]
                        values = args[3].split("/")
                        self.colour[face_id] = []
                        for i, v in enumerate(values):
                            if v == "random":
                                value = random()
                            else:
                                try:
                                    value = float(v)
                                except ValueError:
                                    value = 1.0 if i == 3 else 0.5  # reasonable defaults
                            self.colour[face_id].append(value)

                        # Ensure RGBA has exactly 4 values
                        while len(self.colour[face_id]) < 4:
                            self.colour[face_id].append(1.0 if len(self.colour[face_id]) == 3 else 0.5)


    @property
    def vertices(self):
        verts = []
        for face in self.faces.values():
            verts.extend(face)
        return np.array(verts, dtype='f4')

    def get_homography_matrix(self):
        r = 0
        sx, sy = 1, 1
        ox, oy = 0.5, 0.5
        tx, ty = 0.0, 0.0

        cos = np.cos(r)
        sin = np.sin(r)

        to_origin = np.array([
            [1, 0, -ox],
            [0, 1, -oy],
            [0, 0, 1]
        ])

        transform = np.array([
            [cos * sx, -sin * sy, 0],
            [sin * sx,  cos * sy, 0],
            [0,         0,        1]
        ])

        perspective = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [tx, ty, 1]
        ])

        from_origin = np.array([
            [1, 0, ox],
            [0, 1, oy],
            [0, 0, 1]
        ])

        H = from_origin @ perspective @ transform @ to_origin
        return H.astype("f4")

def coord_scaling(vertices, width, height):
    for i in range(len(vertices)):
        vertices[i][0] = (0 - (width / 2) + vertices[i][0]) / width * 2
        vertices[i][1] = (((height - vertices[i][1]) - (height / 2)) / height) * 2
    return vertices
