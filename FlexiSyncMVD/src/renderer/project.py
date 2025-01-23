import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Dict, BinaryIO, cast
from pytorch3d.common.datatypes import Device
from pytorch3d.io import load_obj, IO
from pytorch3d.io.obj_io import _write_normals, _write_faces
from pytorch3d.io.experimental_gltf_io import (
    _GLTFLoader,
    _PrimitiveMode,
    _ComponentType,
    _TargetType,
    _ITEM_TYPES,
    _DTYPE_BYTES,
    OurEncoder,
    _GLTF_MAGIC,
    _JSON_CHUNK_TYPE,
    _BINARY_CHUNK_TYPE,
)
from io import BytesIO

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    AmbientLights,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    TexturesAtlas,
    TexturesVertex,
    TexturesUV
)
from pytorch3d.renderer.blending import BlendParams

from PIL import Image
from collections import OrderedDict, defaultdict
import struct

import os
import json
from pathlib import Path
from iopath.common.file_io import PathManager
import traceback

from .geometry import HardGeometryShader
from .shader import HardNChannelFlatShader
from .voronoi import voronoi_solve

class SubMeshData:
    def __init__(self, name: str):
        self.name = name
        self.v_i_start2end = (None, None)
        self.vt_i_start2end = (None, None)
        self.vn_i_start2end = (None, None)
        self.f_i_start2end = (None, None)
        self.material = None

    def set_start_index(self, v_i, vt_i, vn_i, f_i):
        self.v_i_start2end = (v_i, self.v_i_start2end[1])
        self.vt_i_start2end = (vt_i, self.vt_i_start2end[1])
        self.vn_i_start2end = (vn_i, self.vn_i_start2end[1])
        self.f_i_start2end = (f_i, self.f_i_start2end[1])

    def set_end_index(self, v_i, vt_i, vn_i, f_i):
        self.v_i_start2end = (self.v_i_start2end[0], v_i)
        self.vt_i_start2end = (self.vt_i_start2end[0], vt_i)
        self.vn_i_start2end = (self.vn_i_start2end[0], vn_i)
        self.f_i_start2end = (self.f_i_start2end[0], f_i)

class GLBWriter:
    def __init__(self, buffer_stream: BinaryIO) -> None:
        self._json_data = defaultdict(list)
        self.buffer_stream = buffer_stream

    def _write_accessor_json(self, mesh: Meshes, key: str, buffer_view) -> Tuple[int, np.ndarray]:
        if key == "positions":
            data = mesh.verts_packed().cpu().numpy()
            component_type = _ComponentType.FLOAT
            element_type = "VEC3"
            buffer_view_offset = 0
            element_min = list(map(float, np.min(data, axis=0)))
            element_max = list(map(float, np.max(data, axis=0)))
            byte_per_element = 3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
        elif key == "normals":
            data = mesh.verts_normals_packed().cpu().numpy()
            component_type = _ComponentType.FLOAT
            element_type = "VEC3"
            buffer_view_offset = 3
            element_min = list(map(float, np.min(data, axis=0)))
            element_max = list(map(float, np.max(data, axis=0)))
            byte_per_element = 3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
        elif key == "texcoords":
            component_type = _ComponentType.FLOAT
            data = mesh.textures.verts_uvs_list()[0].cpu().numpy()
            data[:, 1] = 1 - data[:, -1]  # flip y tex-coordinate
            element_type = "VEC2"
            buffer_view_offset = 2
            element_min = list(map(float, np.min(data, axis=0)))
            element_max = list(map(float, np.max(data, axis=0)))
            byte_per_element = 2 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
        elif key == "texvertices":
            component_type = _ComponentType.FLOAT
            data = mesh.textures.verts_features_list()[0].cpu().numpy()
            element_type = "VEC3"
            byte_per_element = 3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
        elif key == "indices":
            component_type = _ComponentType.UNSIGNED_INT
            data = (
                mesh.faces_packed()
                .cpu()
                .numpy()
                .astype(_ITEM_TYPES[component_type])
            )
            element_type = "SCALAR"
            buffer_view_offset = 1
            element_min = list(map(int, np.min(data, keepdims=True)))
            element_max = list(map(int, np.max(data, keepdims=True)))
            byte_per_element = (
                3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.UNSIGNED_INT]]
            )
        else:
            raise NotImplementedError(
                "invalid key accessor, should be one of positions, indices or texcoords"
            )

        buffer_view += buffer_view_offset
        name = f"Node-Mesh_{key}_{buffer_view}"
        count = int(data.shape[0])
        byte_length = count * byte_per_element
        accessor_json = {
            "name": name,
            "componentType": component_type,
            "type": element_type,
            "bufferView": buffer_view,
            "byteOffset": 0,
            "min": element_min,
            "max": element_max,
            "count": count * 3 if key == "indices" else count,
        }
        self._json_data["accessors"].append(accessor_json)
        return (byte_length, data)

    def _write_bufferview(self, key: str, buffer_view: int, **kwargs):
        if key not in ["positions", "indices", "texcoords", "texvertices", "normals"]:
            raise ValueError(
                "key must be one of positions, indices, texcoords, texvertices, or normals"
            )

        bufferview = {
            "name": f"bufferView_{key}_{buffer_view}",
            "buffer": 0,
        }
        target = _TargetType.ARRAY_BUFFER
        if key == "positions" or key == "normals":
            byte_per_element = 3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
            bufferview["byteStride"] = int(byte_per_element)
        elif key == "texcoords":
            byte_per_element = 2 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
            target = _TargetType.ARRAY_BUFFER
            bufferview["byteStride"] = int(byte_per_element)
        elif key == "texvertices":
            byte_per_element = 3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.FLOAT]]
            target = _TargetType.ELEMENT_ARRAY_BUFFER
            bufferview["byteStride"] = int(byte_per_element)
        elif key == "indices":
            byte_per_element = (
                3 * _DTYPE_BYTES[_ITEM_TYPES[_ComponentType.UNSIGNED_INT]]
            )
            target = _TargetType.ELEMENT_ARRAY_BUFFER

        bufferview["target"] = target
        bufferview["byteOffset"] = kwargs.get("offset")
        bufferview["byteLength"] = kwargs.get("byte_length")
        self._json_data["bufferViews"].append(bufferview)

    def _write_image_buffer(self, image, texture_name, buffer_view, **kwargs) -> Tuple[int, bytes]:
        image_np = image.cpu().numpy()
        image_array = (image_np * 255.0).astype(np.uint8)
        im = Image.fromarray(image_array)
        with BytesIO() as f:
            im.save(f, format="PNG")
            image_data = f.getvalue()

        image_data_byte_length = len(image_data)
        bufferview_image = {
            "name": f"bufferView_image_{buffer_view}",
            "buffer": 0,
        }
        bufferview_image["byteOffset"] = kwargs.get("offset")
        bufferview_image["byteLength"] = image_data_byte_length
        self._json_data["bufferViews"].append(bufferview_image)

        # add image json
        if texture_name.endswith(".png"):
            imageType = "image/png"
        elif texture_name.endswith(".jpg"):
            imageType = "image/jpeg"
        else:
            raise ValueError(f"Texture image format not supported for {texture_name}, use .png or .jpg")
        
        image_name = os.path.splitext(texture_name)[0]
        image = {"name": image_name, "mimeType": imageType, "bufferView": buffer_view}
        self._json_data["images"].append(image)
        return (image_data_byte_length, image_data)

    def save(
        self, 
        sub_mesh_data_list: List[SubMeshData], 
        tex_to_mat_map: Dict[str,List[str]],
        verts,
        faces,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        all_texture_maps: List[torch.Tensor],
        normals: Optional[torch.Tensor] = None,
    ):
        # initialize json with one scene and one node
        scene_index = 0
        # pyre-fixme[6]: Incompatible parameter type
        self._json_data["asset"] = {"version": "2.0"}
        self._json_data["scene"] = scene_index
        self._json_data["scenes"].append({"nodes": [len(sub_mesh_data_list)]})

        material_json_index = 0
        byte_offset = 0
        bufferview_index = 0
        all_image_data = []
        for i, (texture_name, material_names) in enumerate(tex_to_mat_map.items()):
            # Check if texture_name is empty string
            if not texture_name:
                texture_name = f"texture_{i}.png"
            # add image json & bufferView
            image_byte, image_data = self._write_image_buffer(all_texture_maps[i], texture_name, bufferview_index, offset=byte_offset)
            byte_offset += image_byte
            all_image_data.append(image_data)

            bufferview_index += 1

            # add material json
            for material_name in material_names:
                # add texture json, one texture json per material json
                texture = {"sampler": 0, "source": i}
                self._json_data["textures"].append(texture)

                material = {
                    "name": material_name,
                    "pbrMetallicRoughness": {
                        "baseColorTexture": {"index": material_json_index},
                        "baseColorFactor": [1, 1, 1, 1],
                        "metallicFactor": 0,
                        "roughnessFactor": 0.99,
                    },
                    "emissiveFactor": [0, 0, 0],
                    "alphaMode": "OPAQUE",
                }
                self._json_data["materials"].append(material)
                material_json_index += 1
                
        image_byte_add_offset = 4 - byte_offset % 4
        byte_offset += image_byte_add_offset

        zero_map = torch.zeros((16, 16, 3), device=verts.device)
        add_i = 4 if normals is not None else 3
        accessor_index = 0
        all_mesh_data = []
        for i, sub_mesh_data in enumerate(sub_mesh_data_list):
            # one node per mesh
            node = {"name": sub_mesh_data.name, "mesh": i}
            self._json_data["nodes"].append(node)
            
            # get material json index for mesh json
            material_json_index = None
            for m_i, material_json in enumerate(self._json_data["materials"]):
                if material_json["name"] == sub_mesh_data.material:
                    material_json_index = m_i
                    break

            assert material_json_index is not None, f"Material {sub_mesh_data.material} not found in materials list"

            # add mesh json
            mesh_json = defaultdict(list)
            # pyre-fixme[6]: Incompatible parameter type
            mesh_json["name"] = sub_mesh_data.name
            attributes = {
                "POSITION": accessor_index, 
                "NORMAL": accessor_index + 3, 
                "TEXCOORD_0": accessor_index + 2
            } if normals is not None else {
                "POSITION": accessor_index, 
                "TEXCOORD_0": accessor_index + 2
            }
            primitives = {
                "attributes": attributes,
                "indices": accessor_index + 1, # faces
                "mode": _PrimitiveMode.TRIANGLES,
                "material": material_json_index,
            }

            mesh_json["primitives"].append(primitives)
            self._json_data["meshes"].append(mesh_json)

            # write accessor and buffer view for positions, indices, texcoords and normals
            v_i_start, v_i_end = sub_mesh_data.v_i_start2end
            vt_i_start, vt_i_end = sub_mesh_data.vt_i_start2end
            vn_i_start, vn_i_end = sub_mesh_data.vn_i_start2end
            f_i_start, f_i_end = sub_mesh_data.f_i_start2end

            sub_verts = verts[v_i_start:v_i_end]
            sub_faces = faces[f_i_start:f_i_end] - v_i_start
            sub_verts_uvs = verts_uvs[vt_i_start:vt_i_end]
            sub_faces_uvs = faces_uvs[f_i_start:f_i_end] - vt_i_start

            if normals is not None:
                sub_normals = normals[vn_i_start:vn_i_end]

            mesh = Meshes(verts=[sub_verts], faces=[sub_faces], verts_normals=[sub_normals] if normals is not None else None, 
                          textures=TexturesUV(verts_uvs=[sub_verts_uvs], faces_uvs=[sub_faces_uvs], maps=[zero_map]))

            # check validity of mesh
            if mesh.verts_packed() is None or mesh.faces_packed() is None:
                raise ValueError("invalid mesh to save, verts or face indices are empty")

            # accessors for positions, face indices, texture uvs and vertex normals
            pos_byte, pos_data = self._write_accessor_json(mesh, "positions", bufferview_index)
            idx_byte, idx_data = self._write_accessor_json(mesh, "indices", bufferview_index)
            if mesh.textures.verts_uvs_list()[0] is not None:
                tex_byte, tex_data = self._write_accessor_json(mesh, "texcoords", bufferview_index)
                texcoords = True
            else:
                raise ValueError("invalid mesh to save, texture uvs are empty")
            
            if normals is not None:
                nom_byte, nom_data = self._write_accessor_json(mesh, "normals", bufferview_index)

            # bufferViews for positions, indices, texcoords and normals
            self._write_bufferview("positions", buffer_view=bufferview_index, byte_length=pos_byte, offset=byte_offset)
            byte_offset += pos_byte

            self._write_bufferview("indices", buffer_view=bufferview_index+1, byte_length=idx_byte, offset=byte_offset)
            byte_offset += idx_byte
                
            if texcoords:
                self._write_bufferview(
                    "texcoords", buffer_view=bufferview_index+2, byte_length=tex_byte, offset=byte_offset
                )
            else:
                self._write_bufferview(
                    "texvertices", buffer_view=bufferview_index+2, byte_length=tex_byte, offset=byte_offset
                )
            byte_offset += tex_byte

            if normals is not None:
                self._write_bufferview("normals", buffer_view=bufferview_index+3, byte_length=nom_byte, offset=byte_offset)
                byte_offset += nom_byte

            accessor_index += add_i
            bufferview_index += add_i

            mesh_data = [pos_data, idx_data, tex_data, nom_data] if normals is not None else [pos_data, idx_data, tex_data]
            all_mesh_data.extend(mesh_data)

        node = {"name": "object_root", "children": list(range(len(sub_mesh_data_list)))}
        self._json_data["nodes"].append(node)

        # default sampler
        sampler = {"magFilter": 9729, "minFilter": 9986, "wrapS": 10497, "wrapT": 10497}
        self._json_data["samplers"].append(sampler)

        # buffers
        self._json_data["buffers"].append({"byteLength": int(byte_offset)})

        # organize into a glb
        json_bytes = bytes(json.dumps(self._json_data, cls=OurEncoder), "utf-8")
        json_length = len(json_bytes)

        # write header
        version = 2
        total_header_length = 28  # (file header = 12) + 2 * (chunk header = 8)
        file_length = json_length + byte_offset + total_header_length
        header = struct.pack("<III", _GLTF_MAGIC, version, file_length)
        self.buffer_stream.write(header)

        # write json
        self.buffer_stream.write(struct.pack("<II", json_length, _JSON_CHUNK_TYPE))
        self.buffer_stream.write(json_bytes)

        # write binary data
        self.buffer_stream.write(struct.pack("<II", byte_offset, _BINARY_CHUNK_TYPE))
        for image_data in all_image_data:
            self.buffer_stream.write(image_data)

        self.buffer_stream.write(b"\x00" * image_byte_add_offset)

        for mesh_data in all_mesh_data:
            self.buffer_stream.write(mesh_data)


# Pytorch3D based renderering functions, managed in a class
# Render size is recommended to be the same as your latent view size
# DO NOT USE "bilinear" sampling when you are handling latents.
# Stable Diffusion has 4 latent channels so use channels=4

class UVProjection():

    def __init__(self, texture_size=96, w2h_ratio=1, render_size=64, sampling_mode="nearest", channels=3, device=None):
        self.channels = channels
        self.device = device or torch.device("cpu")
        #self.device = torch.device("cpu")
        self.lights = AmbientLights(ambient_color=((1.0,)*channels,), device=self.device)
        self.max_texture_height = texture_size
        self.target_size = (texture_size, texture_size*w2h_ratio) # (H, W)
        self.w2h_ratio = w2h_ratio
        self.render_size = render_size
        self.sampling_mode = sampling_mode

    # Load mesh, rescale the mesh to fit into the bounding box
    def load_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=False):
        if mesh_path.lower().endswith(".obj"):
            (mesh, verts_uvs, faces_uvs,
            tex_height, tex_width, all_tex_resized_widths, 
            tex_to_vt_map, tex_to_mat_map) = self.load_obj_as_mesh(mesh_path, device=self.device)
            sub_mesh_data_list = self.get_obj_sub_mesh_data_list(mesh_path)
        elif mesh_path.lower().endswith(".glb"):
            (mesh, verts_uvs, faces_uvs,
            tex_height, tex_width, all_tex_resized_widths, 
            sub_mesh_data_list, tex_to_vt_map, tex_to_mat_map) = self.load_glb_meshes(mesh_path, device=self.device)
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."

        self.center_and_scale_mesh(mesh, scale_factor, auto_center)

        if autouv or (mesh.textures is None):
            mesh = self.uv_unwrap(mesh, verts_uvs, faces_uvs)

        self.mesh = mesh
        self.target_size = (tex_height, tex_width)
        self.w2h_ratio = tex_width / tex_height
        self.all_tex_resized_widths = all_tex_resized_widths
        # Used for saving the mesh
        self.sub_mesh_data_list = sub_mesh_data_list
        self.tex_to_vt_map = tex_to_vt_map
        self.tex_to_mat_map = tex_to_mat_map

    def load_obj_as_mesh(
        self,
        obj_path,
        device: Optional[Device] = None,
        load_textures: bool = True,
        texture_atlas_size: int = 4,
        texture_wrap: Optional[str] = "repeat",
        path_manager: Optional[PathManager] = None,
    ):
        """
        Load meshes from a .obj file using the load_obj function, and
        return them as a Meshes object. This only works for meshes which have a
        single texture image for the whole mesh. See the load_obj function for more
        details. material_colors and normals are not stored.

        Args:
            obj_path: file-like object (with methods read, readline, tell,
                and seek), pathlib path or string containing file name.
            device: Desired device of returned Meshes. Default:
                uses the current device for the default tensor type.
            load_textures: Boolean indicating whether material files are loaded
            create_texture_atlas, texture_atlas_size, texture_wrap: as for load_obj.
            path_manager: optionally a PathManager object to interpret paths.
        """

        verts, faces, aux = load_obj(
            obj_path,
            load_textures=load_textures,
            create_texture_atlas=False,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
        )

        verts_uvs = None
        faces_uvs = None
        if load_textures:
            verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
            faces_uvs = faces.textures_idx.to(device)  # (F, 3)

        # TexturesUV type, supports multiple UVs, need vt prenset in obj file		
        tex_to_vt_map, tex_to_mat_map = self.get_obj_tex_to_vt_map(obj_path)

        tex_maps = aux.texture_images # Dict[material name, image tensor]
        if tex_maps is not None:
            all_tex_maps = {tex_name: tex_maps[mats[0]] for tex_name, mats in tex_to_mat_map.items() if mats[0] in tex_maps} # Dict[texture image name, image tensor]


        mesh, tex_height, tex_width, resized_widths = self.construct_texture_mesh(tex_to_vt_map, all_tex_maps, verts, faces.verts_idx, aux.normals, verts_uvs, faces_uvs, device)

        return mesh, verts_uvs, faces_uvs, tex_height, tex_width, resized_widths, tex_to_vt_map, tex_to_mat_map

    def get_obj_tex_to_vt_map(self, obj_path):
        image_name: str = ""
        try:
            data_dir = "./"
            if isinstance(obj_path, (str, bytes)):
                # pyre-fixme[6]: For 1st argument expected `PathLike[Variable[AnyStr <:
                #  [str, bytes]]]` but got `Union[Path, bytes, str]`.
                data_dir = os.path.dirname(obj_path)

            # Parse the .obj file to get the material to vertex uv map.
            with open(obj_path, "r") as f:
                mat_to_vt_map = {}
                empty_mat_vt = []
                last_vt_start_index = None
                lines = [line.strip() for line in f]
                # startswith expects each line to be a string. If the file is read in as
                # bytes then first decode to strings.
                if lines and isinstance(lines[0], bytes):
                    lines = [el.decode("utf-8") for el in lines]

                vt_i = 0
                before_vt = True
                for line in lines:
                    tokens = line.strip().split()
                    if line.startswith("mtllib"):
                        if len(tokens) < 2:
                            mtl_path = None
                        # NOTE: only allow one .mtl file per .obj.
                        # Definitions for multiple materials can be included
                        # in this one .mtl file.
                        mtl_path = line[len(tokens[0]) :].strip()  # Take the remainder of the line
                        mtl_path = os.path.join(data_dir, mtl_path)
                    elif line.startswith("vt "):
                        if before_vt:
                            if last_vt_start_index is not None:
                                empty_mat_vt.append((last_vt_start_index, vt_i))
                            last_vt_start_index = vt_i
                            before_vt = False
                        vt_i += 1
                    elif len(tokens) and tokens[0] == "usemtl":
                        material_name = tokens[1]
                        if material_name not in mat_to_vt_map:
                            mat_to_vt_map[material_name] = []
                        mat_to_vt_map[material_name].append((last_vt_start_index, vt_i))
                        last_vt_start_index = None
                        before_vt = True
                    elif not before_vt:
                        before_vt = True

            if vt_i > 0:
                if not mat_to_vt_map:
                    tex_to_vt_map = OrderedDict([("new_texture.png", [(0, vt_i)])])
                    tex_to_mat_map = OrderedDict([("new_texture.png", ["new_material"])])

                elif mtl_path is not None:
                    # Parse the .mtl file to get the texture to material map.
                    with open(mtl_path, "r") as f:
                        tex_to_mat_map = OrderedDict()
                        all_materials = []
                        last_material_name = None
                        for line in f:
                            tokens = line.strip().split()
                            if not tokens:
                                continue
                            if tokens[0] == "newmtl":
                                last_material_name = tokens[1]
                                all_materials.append(last_material_name)
                            elif tokens[0] == "map_Kd":
                                # Diffuse texture map
                                # Account for the case where filenames might have spaces
                                image_name = os.path.basename(line.strip()[7:])
                                if image_name not in tex_to_mat_map:
                                    tex_to_mat_map[image_name] = []
                                if last_material_name not in tex_to_mat_map[image_name]:
                                    tex_to_mat_map[image_name].append(last_material_name)
                                all_materials.pop()

                        for material in all_materials:
                            tex_to_mat_map[material + "_new_texture.png"] = [material]

                    # Combine the two maps to get the texture to vertex uv map.
                    tex_to_vt_map = OrderedDict()
                    for tex, mats in tex_to_mat_map.items():
                        all_vts = []
                        for mat in mats:
                            if mat in mat_to_vt_map and mat_to_vt_map[mat]:
                                all_vts += mat_to_vt_map[mat]
                        if all_vts:
                            tex_to_vt_map[tex] = all_vts

                    if empty_mat_vt:
                        tex_to_mat_map["new_texture.png"] = ["new_material"]
                        tex_to_vt_map["new_texture.png"] = empty_mat_vt

                else:
                    raise ValueError("No .mtl file found for the .obj file.")
            else:
                raise ValueError("No vertex texture coordinates found in the .obj file.")

        except Exception as e:
            print(f"Error in proecssing UVs and materials in .obj file: {e}")
            traceback.print_exc()
            
            tex_to_vt_map = None
            tex_to_mat_map = None

        return tex_to_vt_map, tex_to_mat_map

    def get_obj_sub_mesh_data_list(self, obj_path):
        # Parse the .obj file to get the material to vertex uv map.
        with open(obj_path, "r") as f:
            sub_mesh_data_list = []
            sub_mesh_data = None
            v_i = 0
            vt_i = 0
            vn_i = 0
            f_i = 0

            lines = [line.strip() for line in f]
            # startswith expects each line to be a string. If the file is read in as
            # bytes then first decode to strings.
            if lines and isinstance(lines[0], bytes):
                lines = [el.decode("utf-8") for el in lines]
            for line in lines:
                tokens = line.strip().split()
                if len(tokens) and tokens[0] == "o":
                    sub_mesh_name = tokens[1]
                    if sub_mesh_data is not None:
                        sub_mesh_data.set_end_index(v_i, v_i if vt_i == 0 else vt_i, v_i if vn_i == 0 else vn_i, f_i)
                        sub_mesh_data_list.append(sub_mesh_data)
                    
                    sub_mesh_data = SubMeshData(sub_mesh_name)
                    sub_mesh_data.set_start_index(v_i, v_i if vt_i == 0 else vt_i, v_i if vn_i == 0 else vn_i, f_i)

                elif line.startswith("v "):
                    v_i += 1
                elif line.startswith("vt "):
                    vt_i += 1
                elif line.startswith("vn "):
                    vn_i += 1
                elif line.startswith("f "):
                    f_i += 1
                elif len(tokens) and tokens[0] == "usemtl":
                    material_name = tokens[1]
                    sub_mesh_data.material = material_name

            if sub_mesh_data is None:
                sub_mesh_data = SubMeshData("default")
                sub_mesh_data.set_start_index(0, 0, 0, 0)
                sub_mesh_data.set_end_index(v_i, v_i if vt_i == 0 else vt_i, v_i if vn_i == 0 else vn_i, f_i)
                sub_mesh_data.material = "new_material"
                sub_mesh_data_list.append(sub_mesh_data)
            else:
                sub_mesh_data.set_end_index(v_i, v_i if vt_i == 0 else vt_i, v_i if vn_i == 0 else vn_i, f_i)
                sub_mesh_data_list.append(sub_mesh_data)

        return sub_mesh_data_list

    def load_glb_meshes(
            self, 
            mesh_path, 
            include_textures=True, 
            device: Optional[Device] = None
        ):
        """
        Loads all the meshes from the default scene in the given GLB file.
        and returns them at one mesh object.

        Args:
            path: path to read from
            include_textures: whether to load textures
        """
        with open(mesh_path, "rb") as f:
            loader = _GLTFLoader(cast(BinaryIO, f))
        names_meshes_list = loader.load(include_textures=include_textures)

        (tex_to_vt_map, tex_to_mat_map, all_tex_maps, sub_mesh_data_list,
        verts, faces, normals, verts_uvs, faces_uvs) = self.get_glb_mesh_data(loader, names_meshes_list)

        mesh, tex_height, tex_width, resized_widths = self.construct_texture_mesh(tex_to_vt_map, all_tex_maps, verts, faces, normals, verts_uvs, faces_uvs, device)

        return mesh, verts_uvs, faces_uvs, tex_height, tex_width, resized_widths, sub_mesh_data_list, tex_to_vt_map, tex_to_mat_map
    
    def get_glb_mesh_data(self, loader: _GLTFLoader, names_meshes_list):
        tex_to_mat_map = OrderedDict()
        all_tex_maps = {}
        sub_mesh_data_list = []
        verts = None
        faces = None
        normals = None
        verts_uvs = None
        faces_uvs = None

        try:
            tex_to_vt_map = OrderedDict()

            image_name: str = ""

            nodes = loader._json_data.get("nodes", [])
            meshes = loader._json_data.get("meshes", [])
            materials = loader._json_data.get("materials", [])
            textures = loader._json_data.get("textures", [])
            images = loader._json_data.get("images", [])

            i = 0
            v_i_start = 0
            v_i_end = 0
            vt_i_end = 0
            vn_i_end = 0
            f_i_end = 0
            for current_node in nodes:
                if "mesh" in current_node:
                    mesh_index = current_node["mesh"]
                    mesh_json = meshes[mesh_index]
                    for primitive in mesh_json["primitives"]:
                        # Get the sub-mesh name
                        sub_mesh_name, sub_mesh = names_meshes_list[i]
                        if not sub_mesh_name:
                            sub_mesh_name = f"mesh_{i}"
                        
                        sub_mesh_data = SubMeshData(sub_mesh_name)

                        v_i_start = v_i_end

                        sub_verts = sub_mesh.verts_packed()
                        sub_faces = sub_mesh.faces_packed()
                        sub_normals = sub_mesh.verts_normals_packed()
                        sub_verts_uvs = torch.cat(sub_mesh.textures.verts_uvs_list())
                        sub_faces_uvs = torch.cat(sub_mesh.textures.faces_uvs_list())

                        # Concatenate the sub-mesh data
                        if verts is None:
                            verts = sub_verts
                            faces = sub_faces
                            normals = sub_normals
                            verts_uvs = sub_verts_uvs
                            faces_uvs = sub_faces_uvs
                        else:
                            verts = torch.cat((verts, sub_verts))
                            faces = torch.cat((faces, sub_faces + v_i_end))
                            normals = torch.cat((normals, sub_normals))
                            verts_uvs = torch.cat((verts_uvs, sub_verts_uvs))
                            faces_uvs = torch.cat((faces_uvs, sub_faces_uvs + vt_i_end))

                        # Calculate and set the sub-mesh data
                        sub_mesh_data.set_start_index(v_i_end, vt_i_end, vn_i_end, f_i_end)
                        v_i_end += sub_verts.shape[0]
                        vt_i_end += torch.cat(sub_mesh.textures.verts_uvs_list()).shape[0]
                        vn_i_end += sub_normals.shape[0]
                        f_i_end += sub_faces.shape[0]
                        sub_mesh_data.set_end_index(v_i_end, vt_i_end, vn_i_end, f_i_end)
                        
                        material_name = None
                        image_name = None

                        # Get the sub-mesh material and texture
                        if "material" in primitive:
                            material_index = primitive["material"]
                            material_json = materials[material_index]
                            if "name" in material_json:
                                material_name = material_json["name"]

                            if "pbrMetallicRoughness" in material_json and "baseColorTexture" in material_json["pbrMetallicRoughness"]:
                                texture_index = material_json["pbrMetallicRoughness"]["baseColorTexture"]["index"]
                                texture_json = textures[texture_index]
                                image_index = texture_json["source"]
                                image_json = images[image_index]
                                if "name" in image_json:
                                    image_name = image_json["name"]
                                    if image_json["mimeType"] == "image/png":
                                        image_name += ".png"
                                    elif image_json["mimeType"] == "image/jpeg":
                                        image_name += ".jpg"
                                    else:
                                        raise ValueError(f"Texture image format not supported for {image_json['mimeType']}, use image/png or image/jpeg")

                                    image_tensor = loader._get_texture_map_image(image_index)
                                    all_tex_maps[image_name] = image_tensor
                        
                        if material_name is None:
                            material_name = f"new_material_{mesh_index}"
                        if image_name is None:
                            image_name = f"new_texture_{mesh_index}.png"

                        # Store the output
                        if image_name not in tex_to_vt_map:
                            tex_to_vt_map[image_name] = []
                        tex_to_vt_map[image_name].append((v_i_start, v_i_end))

                        if image_name not in tex_to_mat_map:
                            tex_to_mat_map[image_name] = []
                        if material_name not in tex_to_mat_map[image_name]:
                            tex_to_mat_map[image_name].append(material_name)

                        sub_mesh_data.material = material_name
                        sub_mesh_data_list.append(sub_mesh_data)

                        i += 1

        except Exception as e:
            print(f"Error in proecssing mesh, UVs and materials in .glb file: {e}")
            traceback.print_exc()

            tex_to_vt_map = None

            v_i_end = 0
            vn_i_end = 0
            for mesh_index, (sub_mesh_name, sub_mesh) in enumerate(names_meshes_list):
                if not sub_mesh_name:
                    sub_mesh_name = f"mesh_{mesh_index}"
                
                sub_mesh_data = SubMeshData(sub_mesh_name)

                sub_verts = sub_mesh.verts_packed()
                sub_faces = sub_mesh.faces_packed()
                sub_normals = sub_mesh.verts_normals_packed()

                # Concatenate the sub-mesh data
                if verts is None:
                    verts = sub_verts
                    faces = sub_faces
                    normals = sub_normals
                else:
                    verts = torch.cat((verts, sub_verts))
                    faces = torch.cat((faces, sub_faces + v_i_end))
                    normals = torch.cat((normals, sub_normals))

                # Calculate and set the sub-mesh data
                sub_mesh_data.set_start_index(v_i_end, v_i_end, vn_i_end, f_i_end)
                v_i_end += sub_verts.shape[0]
                vn_i_end += sub_normals.shape[0]
                f_i_end += sub_faces.shape[0]
                sub_mesh_data.set_end_index(v_i_end, v_i_end, vn_i_end, f_i_end)

                material_name = f"new_material_{mesh_index}"
                image_name = f"new_texture_{mesh_index}.png"

                tex_to_mat_map[image_name] = [material_name]

                sub_mesh_data.material = material_name
                sub_mesh_data_list.append(sub_mesh_data)

        return tex_to_vt_map, tex_to_mat_map, all_tex_maps, sub_mesh_data_list, verts, faces, normals, verts_uvs, faces_uvs

    # Construct mesh and one wide texture map
    def construct_texture_mesh(self, tex_to_vt_map, all_tex_maps, verts, faces, normals, verts_uvs, faces_uvs, device):
        if tex_to_vt_map is not None:
            all_tex_heights = [tex_map.shape[0] for tex_map in all_tex_maps.values()]
            tex_height = min(max(all_tex_heights), self.max_texture_height) if all_tex_heights else self.max_texture_height  # max height of all textures
            tex_width = 0
            cumulative_w2h_ratio = 0
            resized_widths = []
            resized_images = []
            for tex_name, vt_list in tex_to_vt_map.items():
                if tex_name in all_tex_maps:
                    tex_map = all_tex_maps[tex_name]
                    height, width, channels = tex_map.shape
                    w2h_ratio = width / height
                    new_width = int(width * (tex_height / height))
                else:
                    tex_map = torch.zeros((tex_height, tex_height, 3), device=device)
                    w2h_ratio = 1
                    new_width = tex_height

                # scale the u coordinate in UV to match the texture aspect ratio, also horizontally concatenate the UVs
                for start_vt_i, end_vt_i in vt_list:
                    verts_uvs[start_vt_i:end_vt_i, 0] = verts_uvs[start_vt_i:end_vt_i, 0] * w2h_ratio + cumulative_w2h_ratio

                resized_image = F.interpolate(tex_map.permute(2, 0, 1).unsqueeze(0), size=(tex_height, new_width), mode='bilinear', align_corners=False)

                tex_width += new_width
                cumulative_w2h_ratio += w2h_ratio
                resized_widths.append(new_width)
                resized_images.append(resized_image.squeeze(0).permute(1, 2, 0))
                
            # Normalize the concatenated UVs to [0, 1]
            verts_uvs[...,0] /= cumulative_w2h_ratio

            concatenated_image = torch.cat(resized_images, dim=1).to(device)[None]
            tex = TexturesUV(
                verts_uvs=[verts_uvs.to(device)], faces_uvs=[faces_uvs.to(device)], maps=concatenated_image
            )

        else:
            tex_height = self.max_texture_height
            tex_width = tex_height
            resized_widths = [tex_height]
            tex = None

        mesh = Meshes(
            verts=[verts.to(device)], faces=[faces.to(device)], verts_normals=[normals.to(device)] if normals is not None else None, textures=tex
        )

        return mesh, tex_height, tex_width, resized_widths
    
    def center_and_scale_mesh(self, mesh, scale_factor=2.0, auto_center=True):
        if auto_center:
            verts = mesh.verts_packed()
            max_bb = (verts - 0).max(0)[0]
            min_bb = (verts - 0).min(0)[0]
            scale = (max_bb - min_bb).max()/2 
            center = (max_bb+min_bb) /2
            mesh.offset_verts_(-center)
            mesh.scale_verts_((scale_factor / float(scale)))
        else:
            mesh.scale_verts_((scale_factor))

    # Code referred to TEXTure code (https://github.com/TEXTurePaper/TEXTurePaper.git)
    def uv_unwrap(self, mesh, vt=None, ft=None):
        if vt is None or ft is None:
            verts_list = mesh.verts_list()[0]
            faces_list = mesh.faces_list()[0]

            import xatlas
            import numpy as np
            v_np = verts_list.cpu().numpy()
            f_np = faces_list.int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).type(verts_list.dtype)
            ft = torch.from_numpy(ft_np.astype(np.int64)).type(faces_list.dtype)

            # remap vertex index range from vt, ft to v_np, f_np
            new_vt = torch.zeros((v_np.shape[0], vt.shape[1]), dtype=vt.dtype)
            new_ft = torch.zeros(ft.shape, dtype=ft.dtype)
            v_i_2_vt_i_map = {v_i: vt_i for vt_i, v_i in enumerate(vmapping)}
            
            for v_i in range(new_vt.shape[0]):
                new_vt[v_i] = vt[v_i_2_vt_i_map[v_i]]
            
            for f_i in range(new_ft.shape[0]):
                new_ft[f_i] = torch.tensor([vmapping[vt_i] for vt_i in ft[f_i]], dtype=ft.dtype)

            vt = new_vt
            ft = new_ft


        new_map = torch.zeros(self.target_size+(self.channels,), device=mesh.device)
        new_tex = TexturesUV(
            [new_map], 
            [ft.to(mesh.device)], 
            [vt.to(mesh.device)], 
            sampling_mode=self.sampling_mode
        )

        mesh.textures = new_tex
        return mesh

    # Save obj mesh
    def save_mesh(self, mesh_path, texture):
        verts_uvs = self.scale_back_to_original_uv(torch.cat(self.mesh.textures.verts_uvs_list()))
        face_uvs = torch.cat(self.mesh.textures.faces_uvs_list())
        # Spilt the texture map into individual texture maps
        split_texture_maps = torch.split(texture, self.all_tex_resized_widths, dim=1)
        if mesh_path.lower().endswith(".obj"):
            self.save_obj_mesh(
                mesh_path, 
                self.mesh.verts_packed(),
                self.mesh.faces_packed(),
                verts_uvs=verts_uvs,
                faces_uvs=face_uvs,
                all_texture_maps=split_texture_maps,
                normals=self.mesh.verts_normals_packed(),
                faces_normals_idx=self.mesh.faces_packed(),
            )
        elif mesh_path.lower().endswith(".glb"):
            with open(mesh_path, "wb") as f:
                writer = GLBWriter(cast(BinaryIO, f))
                writer.save(
                    self.sub_mesh_data_list,
                    self.tex_to_mat_map,
                    self.mesh.verts_packed(),
                    self.mesh.faces_packed(),
                    verts_uvs=verts_uvs,
                    faces_uvs=face_uvs,
                    all_texture_maps=split_texture_maps,
                    normals=self.mesh.verts_normals_packed(),
                )
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."

        #self.save_output_test_pickle(mesh_path.replace(".glb", ".pkl"), verts_uvs, face_uvs, split_texture_maps)

    def scale_back_to_original_uv(self, verts_uvs):
        if self.tex_to_vt_map is not None:
            # Denormalize the concatenated UVs from [0, 1] to [0, w2h_ratio]
            verts_uvs[...,0] *= self.w2h_ratio

            cumulative_w2h_ratio = 0
            for i, (tex_name, vt_list) in enumerate(self.tex_to_vt_map.items()):
                w2h_ratio = self.all_tex_resized_widths[i] / self.target_size[0]
                # scale the each u coordinate in each UV map to [0, 1]
                for start_vt_i, end_vt_i in vt_list:
                    verts_uvs[start_vt_i:end_vt_i, 0] = (verts_uvs[start_vt_i:end_vt_i, 0] - cumulative_w2h_ratio) / w2h_ratio

                cumulative_w2h_ratio += w2h_ratio

        return verts_uvs
    
    def save_output_test_pickle(self, output_path, verts_uvs, face_uvs, split_texture_maps):
        import pickle

        output_test_dict = {
            "sub_mesh_data_list": self.sub_mesh_data_list,
            "tex_to_mat_map": self.tex_to_mat_map,
            "verts": self.mesh.verts_packed(),
            "faces": self.mesh.faces_packed(),
            "verts_uvs": verts_uvs,
            "face_uvs": face_uvs,
            "split_texture_maps": split_texture_maps,
            "normals": self.mesh.verts_normals_packed(),
        }
        with open(output_path, "wb") as f:
            pickle.dump(output_test_dict, f)
        
    def save_obj_mesh(
        self,
        mesh_path: Union[Path, str],
        verts,
        faces,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        all_texture_maps: List[torch.Tensor],
        decimal_places: Optional[int] = None,
        normals: Optional[torch.Tensor] = None,
        faces_normals_idx: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Save a mesh to an .obj file.

        Args:
            mesh_path: File (str or path) to which the mesh should be written.
            verts: FloatTensor of shape (V, 3) giving vertex coordinates.
            faces: LongTensor of shape (F, 3) giving faces.
            decimal_places: Number of decimal places for saving.
            normals: FloatTensor of shape (V, 3) giving normals for faces_normals_idx
                to index into.
            faces_normals_idx: LongTensor of shape (F, 3) giving the index into
                normals for each vertex in the face.
            verts_uvs: FloatTensor of shape (V, 2) giving the uv coordinate per vertex.
            faces_uvs: LongTensor of shape (F, 3) giving the index into verts_uvs for
                each vertex in the face.
            texture_map: FloatTensor of shape (H, W, 3) representing the texture map
                for the mesh which will be saved as an image or multiple imahges depends on number of UVs. The values are expected
                to be in the range [0, 1],
        """
        if len(verts) and (verts.dim() != 2 or verts.size(1) != 3):
            message = "'verts' should either be empty or of shape (num_verts, 3)."
            raise ValueError(message)

        if len(faces) and (faces.dim() != 2 or faces.size(1) != 3):
            message = "'faces' should either be empty or of shape (num_faces, 3)."
            raise ValueError(message)
        
        if torch.any(faces >= verts.shape[0]) or torch.any(faces < 0):
            raise ValueError("Faces have invalid indices")

        if (normals is None) != (faces_normals_idx is None):
            message = "'normals' and 'faces_normals_idx' must both be None or neither."
            raise ValueError(message)

        if faces_normals_idx is not None and (
            faces_normals_idx.dim() != 2 or faces_normals_idx.size(1) != 3
        ):
            message = (
                "'faces_normals_idx' should either be empty or of shape (num_faces, 3)."
            )
            raise ValueError(message)

        if normals is not None and (normals.dim() != 2 or normals.size(1) != 3):
            message = "'normals' should either be empty or of shape (num_verts, 3)."
            raise ValueError(message)

        if faces_uvs is not None and (faces_uvs.dim() != 2 or faces_uvs.size(1) != 3):
            message = "'faces_uvs' should either be empty or of shape (num_faces, 3)."
            raise ValueError(message)

        if verts_uvs is not None and (verts_uvs.dim() != 2 or verts_uvs.size(1) != 2):
            message = "'verts_uvs' should either be empty or of shape (num_verts, 2)."
            raise ValueError(message)

        save_texture = all(t is not None for t in [faces_uvs, verts_uvs, all_texture_maps])
        save_normals = normals is not None and faces_normals_idx is not None
        output_path = Path(mesh_path)

        # Save the .obj file
        # pyre-fixme[9]: f has type `Union[Path, str]`; used as `IO[typing.Any]`.
        with open(mesh_path, "w") as f:
            if save_texture:
                # Add the header required for the texture info to be loaded correctly
                obj_header = "\nmtllib {0}.mtl\n\n".format(output_path.stem)
                # pyre-fixme[16]: Item `Path` of `Union[Path, str]` has no attribute
                #  `write`.
                f.write(obj_header)
            if len(verts) and (verts.dim() != 2 or verts.size(1) != 3):
                message = "'verts' should either be empty or of shape (num_verts, 3)."
                raise ValueError(message)

            if len(faces) and (faces.dim() != 2 or faces.size(1) != 3):
                message = "'faces' should either be empty or of shape (num_faces, 3)."
                raise ValueError(message)

            if not (len(verts) or len(faces)):
                raise ValueError("Empty 'verts' and 'faces' arguments provided")

            verts, faces = verts.cpu(), faces.cpu()
            verts_uvs, faces_uvs = verts_uvs.cpu(), faces_uvs.cpu()

            if decimal_places is None:
                float_str = "%f"
            else:
                float_str = "%" + ".%df" % decimal_places

            for sub_mesh_data in self.sub_mesh_data_list:
                v_i_start, v_i_end = sub_mesh_data.v_i_start2end
                vt_i_start, vt_i_end = sub_mesh_data.vt_i_start2end
                vn_i_start, vn_i_end = sub_mesh_data.vn_i_start2end
                f_i_start, f_i_end = sub_mesh_data.f_i_start2end

                sub_verts = verts[v_i_start:v_i_end]
                sub_faces = faces[f_i_start:f_i_end]

                sub_verts_uvs = verts_uvs[vt_i_start:vt_i_end]
                sub_faces_uvs = faces_uvs[f_i_start:f_i_end]

                if normals is not None:
                    sub_normals = normals[vn_i_start:vn_i_end]
                if faces_normals_idx is not None:
                    sub_faces_normals_idx = faces_normals_idx[f_i_start:f_i_end]

                lines = f"\no {sub_mesh_data.name}\n"

                if len(sub_verts):
                    V, D = sub_verts.shape
                    for i in range(V):
                        vert = [float_str % sub_verts[i, j] for j in range(D)]
                        lines += "v %s\n" % " ".join(vert)

                if save_normals:
                    lines += _write_normals(sub_normals, sub_faces_normals_idx, float_str)

                if save_texture:
                    # Save verts uvs after verts
                    if len(sub_verts_uvs):
                        uV, uD = sub_verts_uvs.shape
                        for i in range(uV):
                            uv = [float_str % sub_verts_uvs[i, j] for j in range(uD)]
                            lines += "vt %s\n" % " ".join(uv)

                        lines += "s 1\n"
                        lines += "usemtl %s\n" % sub_mesh_data.material

                f.write(lines)

                if len(faces):
                    _write_faces(
                        f,
                        sub_faces,
                        sub_faces_uvs if save_texture else None,
                        sub_faces_normals_idx if save_normals else None,
                    )

        # Save the .mtl and .png files associated with the texture
        if save_texture:
            mtl_path = output_path.with_suffix(".mtl")

            with open(mtl_path, "w") as f_mtl:

                for i, (texture_name, material_names) in enumerate(self.tex_to_mat_map.items()):
                    # Save texture map to output folder
                    texture_image = all_texture_maps[i]
                    texture_image = texture_image.detach().cpu() * 255.0
                    image = Image.fromarray(texture_image.numpy().astype(np.uint8))

                    image_path = output_path.with_name(texture_name)
                    with open(image_path, "wb") as im_f:
                        image.save(im_f)

                    for material_name in material_names:
                        # Create .mtl file with the material name and texture map filename
                        lines = f"newmtl {material_name}\n"
                        lines += "Ka 1.000 1.000 1.000\n"
                        lines += "Kd 1.000 1.000 1.000\n"
                        lines += "Ks 0.000 0.000 0.000\n"
                        lines += "Ke 0.000000 0.000000 0.000000\n"
                        lines += "Ni 1.000000\n"
                        lines += "d 1.0\n"
                        lines += "illum 2\n"
                        lines += "Pr 0.500000\n"
                        lines += "Pm 0.000000\n"
                        lines += f"map_Kd {image_path.name}\n\n"
                        f_mtl.write(lines)

    '''
        A functions that disconnect faces in the mesh according to
        its UV seams. The number of vertices are made equal to the
        number of unique vertices its UV layout, while the faces list
        is intact.
    '''
    def disconnect_faces(self):
        mesh = self.mesh
        verts_list = mesh.verts_list()
        faces_list = mesh.faces_list()
        verts_uvs_list = mesh.textures.verts_uvs_list()
        faces_uvs_list = mesh.textures.faces_uvs_list()
        packed_list = [v[f] for v,f in zip(verts_list, faces_list)]
        verts_disconnect_list = [
            torch.zeros(
                (verts_uvs_list[i].shape[0], 3), 
                dtype=verts_list[0].dtype, 
                device=verts_list[0].device
            ) 
            for i in range(len(verts_list))]
        for i in range(len(verts_list)):
            verts_disconnect_list[i][faces_uvs_list] = packed_list[i]
        #assert not mesh.has_verts_normals(), "Not implemented for vertex normals"
        self.mesh_d = Meshes(verts_disconnect_list, faces_uvs_list, mesh.textures)
        return self.mesh_d


    '''
        A function that construct a temp mesh for back-projection.
        Take a disconnected mesh and a rasterizer, the function calculates
        the projected faces as the UV, as use its original UV with pseudo
        z value as world space geometry.
    '''
    def construct_uv_mesh(self):
        mesh = self.mesh_d
        verts_list = mesh.verts_list()
        verts_uvs_list = mesh.textures.verts_uvs_list()
        # faces_list = [torch.flip(faces, [-1]) for faces in mesh.faces_list()]
        new_verts_list = []
        for i, (verts, verts_uv) in enumerate(zip(verts_list, verts_uvs_list)):
            verts = verts.clone()
            verts_uv = verts_uv.clone()
            # Horizontally scale & move the concatenated UV, since by default, UV will be placed at the center of the texture image's horizontal axis
            verts_uv[...,0] = verts_uv[...,0] * self.w2h_ratio - (self.w2h_ratio / 2 - 0.5)
            verts[...,0:2] = verts_uv[...,:]
            verts = (verts - 0.5) * 2
            verts[...,2] *= 1
            new_verts_list.append(verts)
        textures_uv = mesh.textures.clone()
        self.mesh_uv = Meshes(new_verts_list, mesh.faces_list(), textures_uv)
        return self.mesh_uv


    # Set texture for the current mesh.
    def set_texture_map(self, texture):
        new_map = texture.permute(1, 2, 0)
        new_map = new_map.to(self.device)
        new_tex = TexturesUV(
            [new_map], 
            self.mesh.textures.faces_uvs_padded(), 
            self.mesh.textures.verts_uvs_padded(), 
            sampling_mode=self.sampling_mode
            )
        self.mesh.textures = new_tex


    # Set the initial normal noise texture
    # No generator here for replication of the experiment result. Add one as you wish
    def set_noise_texture(self, channels=None):
        if not channels:
            channels = self.channels
        noise_texture = torch.normal(0, 1, (channels,) + self.target_size, device=self.device)
        self.set_texture_map(noise_texture)
        return noise_texture


    # Set the cameras given the camera poses and centers
    def set_cameras(self, camera_poses, centers=None, camera_distance=2.7, scale=None):
        elev = torch.FloatTensor([pose[0] for pose in camera_poses])
        azim = torch.FloatTensor([pose[1] for pose in camera_poses])
        R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim, at=centers or ((0,0,0),))
        self.cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, scale_xyz=scale or ((1,1,1),))


    # Set all necessary internal data for rendering and texture baking
    # Can be used to refresh after changing camera positions
    def set_cameras_and_render_settings(self, camera_poses, centers=None, camera_distance=2.7, render_size=None, scale=None):
        self.set_cameras(camera_poses, centers, camera_distance, scale=scale)
        if render_size is None:
            render_size = self.render_size
        if not hasattr(self, "renderer"):
            self.setup_renderer(size=render_size)
        if not hasattr(self, "mesh_d"):
            self.disconnect_faces()
        if not hasattr(self, "mesh_uv"):
            self.construct_uv_mesh()
        self.calculate_tex_gradient()
        self.calculate_visible_triangle_mask()
        _,_,_,cos_maps,_, _ = self.render_geometry()
        self.calculate_cos_angle_weights(cos_maps)


    # Setup renderers for rendering
    # max faces per bin set to 30000 to avoid overflow in many test cases.
    # You can use default value to let pytorch3d handle that for you.
    def setup_renderer(self, size=64, blur=0.0, face_per_pix=1, perspective_correct=False, channels=None):
        if not channels:
            channels = self.channels

        self.raster_settings = RasterizationSettings(
            image_size=size, 
            blur_radius=blur, 
            faces_per_pixel=face_per_pix,
            perspective_correct=perspective_correct,
            cull_backfaces=True,
            max_faces_per_bin=None,
            bin_size=-1,
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings,

            ),
            shader=HardNChannelFlatShader(
                device=self.device, 
                cameras=self.cameras,
                lights=self.lights,
                channels=channels,
                # materials=materials
            )
        )


    # Bake screen-space cosine weights to UV space
    # May be able to reimplement using the generic "bake_texture" function, but it works so leave it here for now
    @torch.enable_grad()
    def calculate_cos_angle_weights(self, cos_angles, fill=True, channels=None):
        if not channels:
            channels = self.channels
        cos_maps = []
        tmp_mesh = self.mesh.clone()
        for i in range(len(self.cameras)):
            
            zero_map = torch.zeros(self.target_size+(channels,), device=self.device, requires_grad=True)
            optimizer = torch.optim.SGD([zero_map], lr=1, momentum=0)
            optimizer.zero_grad()
            zero_tex = TexturesUV([zero_map], self.mesh.textures.faces_uvs_padded(), self.mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
            tmp_mesh.textures = zero_tex

            images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights)

            loss = torch.sum((cos_angles[i,:,:,0:1]**1 - images_predicted)**2)
            loss.backward()
            optimizer.step()

            if fill:
                zero_map = zero_map.detach() / (self.gradient_maps[i] + 1E-8)
                zero_map = voronoi_solve(zero_map, self.gradient_maps[i][...,0])
            else:
                zero_map = zero_map.detach() / (self.gradient_maps[i]+1E-8)
            cos_maps.append(zero_map)
        self.cos_maps = cos_maps

        
    # Get geometric info from fragment shader
    # Can be used for generating conditioning image and cosine weights
    # Returns some information you may not need, remember to release them for memory saving
    @torch.no_grad()
    def render_geometry(self, image_size=None, render_cam_index=None):
        if image_size:
            size = self.renderer.rasterizer.raster_settings.image_size
            self.renderer.rasterizer.raster_settings.image_size = image_size
        shader = self.renderer.shader
        self.renderer.shader = HardGeometryShader(device=self.device, cameras=self.cameras[0], lights=self.lights, blend_params=BlendParams(background_color=(0,)*self.channels))
        tmp_mesh = self.mesh.clone()
        
        render_cameras = self.cameras[render_cam_index] if render_cam_index is not None else self.cameras
        verts, normals, depths, cos_angles, texels, fragments = self.renderer(tmp_mesh.extend(len(render_cameras)), cameras=render_cameras, lights=self.lights)
        self.renderer.shader = shader

        if image_size:
            self.renderer.rasterizer.raster_settings.image_size = size

        return verts, normals, depths, cos_angles, texels, fragments


    # Project world normal to view space and normalize
    @torch.no_grad()
    def decode_view_normal(self, normals):
        w2v_mat = self.cameras.get_full_projection_transform()
        normals_view = torch.clone(normals)[:,:,:,0:3]
        normals_view = normals_view.reshape(normals_view.shape[0], -1, 3)
        normals_view = w2v_mat.transform_normals(normals_view)
        normals_view = normals_view.reshape(normals.shape[0:3]+(3,))
        normals_view[:,:,:,2] *= -1
        normals = (normals_view[...,0:3]+1) * normals[...,3:] / 2 + torch.FloatTensor(((((0.5,0.5,1))))).to(self.device) * (1 - normals[...,3:])
        # normals = torch.cat([normal for normal in normals], dim=1)
        normals = normals.clamp(0, 1).permute(0,3,1,2)
        return normals


    # Normalize absolute depth to inverse depth
    @torch.no_grad()
    def decode_normalized_depth(self, depths, batched_norm=False):
        view_z, mask = depths.unbind(-1)
        view_z = view_z * mask + 100 * (1-mask)
        inv_z = 1 / view_z
        inv_z_min = inv_z * mask + 100 * (1-mask)
        if not batched_norm:
            max_ = torch.max(inv_z, 1, keepdim=True)
            max_ = torch.max(max_[0], 2, keepdim=True)[0]

            min_ = torch.min(inv_z_min, 1, keepdim=True)
            min_ = torch.min(min_[0], 2, keepdim=True)[0]
        else:
            max_ = torch.max(inv_z)
            min_ = torch.min(inv_z_min)
        inv_z = (inv_z - min_) / (max_ - min_)
        inv_z = inv_z.clamp(0,1)
        inv_z = inv_z[...,None].repeat(1,1,1,3).permute(0,3,1,2)

        return inv_z


    # Multiple screen pixels could pass gradient to a same texel
    # We can precalculate this gradient strength and use it to normalize gradients when we bake textures
    @torch.enable_grad()
    def calculate_tex_gradient(self, channels=None):
        if not channels:
            channels = self.channels
        tmp_mesh = self.mesh.clone()
        gradient_maps = []
        for i in range(len(self.cameras)):
            zero_map = torch.zeros(self.target_size+(channels,), device=self.device, requires_grad=True)
            optimizer = torch.optim.SGD([zero_map], lr=1, momentum=0)
            optimizer.zero_grad()
            zero_tex = TexturesUV([zero_map], self.mesh.textures.faces_uvs_padded(), self.mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
            tmp_mesh.textures = zero_tex
            images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights)
            loss = torch.sum((1 - images_predicted)**2)
            loss.backward()
            optimizer.step()

            gradient_maps.append(zero_map.detach())

        self.gradient_maps = gradient_maps


    # Get the UV space masks of triangles visible in each view
    # First get face ids from each view, then filter pixels on UV space to generate masks
    @torch.no_grad()
    def calculate_visible_triangle_mask(self, blur_radius=0, faces_per_pixel=5, channels=None, image_size=(512,512)):
        if not channels:
            channels = self.channels

        pix2face_list = []
        for i in range(len(self.cameras)):
            self.renderer.rasterizer.raster_settings.image_size=image_size
            pix2face = self.renderer.rasterizer(self.mesh_d, cameras=self.cameras[i]).pix_to_face
            self.renderer.rasterizer.raster_settings.image_size=self.render_size
            pix2face_list.append(pix2face)

        if not hasattr(self, "mesh_uv"):
            self.construct_uv_mesh()

        raster_settings = RasterizationSettings(
            image_size=self.target_size, 
            blur_radius=blur_radius, 
            faces_per_pixel=faces_per_pixel,
            perspective_correct=False,
            cull_backfaces=False,
            max_faces_per_bin=None,
            bin_size=-1,
        )

        R, T = look_at_view_transform(dist=2, elev=0, azim=0)
        cameras = FoVOrthographicCameras(device=self.device, R=R, T=T)

        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
        uv_pix2face = rasterizer(self.mesh_uv).pix_to_face

        visible_triangles = []
        for i in range(len(pix2face_list)):
            valid_faceid = torch.unique(pix2face_list[i])
            valid_faceid = valid_faceid[1:] if valid_faceid[0]==-1 else valid_faceid
            mask = torch.isin(uv_pix2face[0], valid_faceid, assume_unique=False)
            # uv_pix2face[0][~mask] = -1
            triangle_mask = torch.ones(self.target_size+(faces_per_pixel,), device=self.device)
            triangle_mask[~mask] = 0
            triangle_mask, _ = torch.max(triangle_mask, dim=-1, keepdim=True)
            triangle_mask[:,1:][triangle_mask[:,:-1] > 0] = 1
            triangle_mask[:,:-1][triangle_mask[:,1:] > 0] = 1
            triangle_mask[1:,:][triangle_mask[:-1,:] > 0] = 1
            triangle_mask[:-1,:][triangle_mask[1:,:] > 0] = 1
            visible_triangles.append(triangle_mask)

        self.visible_triangles = visible_triangles
        # self.save_visible_masks("./")

    def save_visible_masks(self, path):
        combined_mask = torch.zeros_like(self.visible_triangles[0])
        for mask in self.visible_triangles:
            combined_mask += mask

        combined_mask = combined_mask.clamp(0, 1)  # Ensure the mask values are either 0 or 1

        # Save the combined mask as an image
        combined_mask_image = (combined_mask.squeeze() * 255).cpu().numpy().astype(np.uint8)
        combined_mask_image = Image.fromarray(combined_mask_image)
        combined_mask_image.save(os.path.join(path,"combined_UV_mask.png"))

    # Render the current mesh and texture from current cameras
    def render_textured_views(self, return_tensor=False):
        meshes = self.mesh.extend(len(self.cameras))
        images_predicted = self.renderer(meshes, cameras=self.cameras, lights=self.lights)
        if return_tensor:
            return images_predicted.permute(0,3,1,2)
        return [image.permute(2, 0, 1) for image in images_predicted]


    # Bake views into a texture
    # First bake into individual textures then combine based on cosine weight
    @torch.enable_grad()
    def bake_texture(self, views=None, main_views=[], cos_weighted=True, channels=None, exp=None, noisy=False, generator=None):
        if not exp:
            exp=1
        if not channels:
            channels = self.channels
        views = [view.permute(1, 2, 0) for view in views]

        tmp_mesh = self.mesh
        bake_maps = [torch.zeros(self.target_size+(views[0].shape[2],), device=self.device, requires_grad=True) for view in views]
        optimizer = torch.optim.SGD(bake_maps, lr=1, momentum=0)
        optimizer.zero_grad()
        loss = 0
        for i in range(len(self.cameras)):    
            bake_tex = TexturesUV([bake_maps[i]], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
            tmp_mesh.textures = bake_tex
            images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights, device=self.device)
            predicted_rgb = images_predicted[..., :-1]
            loss += (((predicted_rgb[...] - views[i]))**2).sum()
        loss.backward(retain_graph=False)
        optimizer.step()

        total_weights = 0
        baked = 0
        for i in range(len(bake_maps)):
            normalized_baked_map = bake_maps[i].detach() / (self.gradient_maps[i] + 1E-8)
            bake_map = voronoi_solve(normalized_baked_map, self.gradient_maps[i][...,0])
            weight = self.visible_triangles[i] * (self.cos_maps[i]) ** exp
            if noisy:
                noise = torch.rand(weight.shape[:-1]+(1,), generator=generator).type(weight.dtype).to(weight.device)
                weight *= noise
            total_weights += weight
            baked += bake_map * weight
        baked /= total_weights + 1E-8
        baked = voronoi_solve(baked, total_weights[...,0])

        bake_tex = TexturesUV([baked], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
        tmp_mesh.textures = bake_tex
        extended_mesh = tmp_mesh.extend(len(self.cameras))
        images_predicted = self.renderer(extended_mesh, cameras=self.cameras, lights=self.lights)
        learned_views = [image.permute(2, 0, 1) for image in images_predicted]

        return learned_views, baked.permute(2, 0, 1), total_weights.permute(2, 0, 1)


    # Move the internel data to a specific device
    def to(self, device):
        for mesh_name in ["mesh", "mesh_d", "mesh_uv"]:
            if hasattr(self, mesh_name):
                mesh = getattr(self, mesh_name)
                setattr(self, mesh_name, mesh.to(device))
        for list_name in ["visible_triangles", "visibility_maps", "cos_maps"]:
            if hasattr(self, list_name):
                map_list = getattr(self, list_name)
                for i in range(len(map_list)):
                    map_list[i] = map_list[i].to(device)
