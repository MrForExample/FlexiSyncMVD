from pytorch3d.io.experimental_gltf_io import _GLTFLoader, MeshGlbFormat, load_meshes
from pytorch3d.io import IO
import torch

from typing import Any, BinaryIO, cast
from collections import OrderedDict

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

def load_glb_mesh(mesh_path):
	io = IO()
	io.register_meshes_format(MeshGlbFormat())
	with open(mesh_path, "rb") as f:
		mesh = io.load_mesh(f, include_textures=True, device="cpu")

	return mesh

def load_glb_meshes(mesh_path, include_textures=True):
	with open(mesh_path, "rb") as f:
		loader = _GLTFLoader(cast(BinaryIO, f))
	names_meshes_list = loader.load(include_textures=include_textures)

	(tex_to_vt_map, tex_to_mat_map, all_tex_maps, sub_mesh_data_list,
	verts, faces, verts_uvs, faces_uvs) = get_glb_mesh_data(loader, names_meshes_list)

	print(f"tex_to_vt_map: {tex_to_vt_map}\n\n tex_to_mat_map: {tex_to_mat_map}\n\n")
	print(f"all_tex_maps: {[(image_name, image_tensor.shape) for image_name, image_tensor in all_tex_maps.items()]}\n\n")
	print(f"sub_mesh_data_list: {[(sub_mesh_data.name, sub_mesh_data.v_i_start2end, sub_mesh_data.vt_i_start2end, sub_mesh_data.vn_i_start2end, sub_mesh_data.f_i_start2end, sub_mesh_data.material) for sub_mesh_data in sub_mesh_data_list]}\n\n")
	print(f"verts: {verts.shape}\n\n faces: {faces.shape}\n\n verts_uvs: {verts_uvs.shape}\n\n faces_uvs: {faces_uvs.shape}\n\n")
	#return mesh

def get_glb_mesh_data(loader: _GLTFLoader, names_meshes_list):
	tex_to_vt_map = OrderedDict()
	tex_to_mat_map = OrderedDict()
	all_tex_maps = {}
	sub_mesh_data_list = []
	verts = None
	faces = None
	verts_uvs = None
	faces_uvs = None

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
				sub_verts_uvs = torch.cat(sub_mesh.textures.verts_uvs_list())
				sub_faces_uvs = torch.cat(sub_mesh.textures.faces_uvs_list())

				# Concatenate the sub-mesh data
				if verts is None:
					verts = sub_verts
					faces = sub_faces
					verts_uvs = sub_verts_uvs
					faces_uvs = sub_faces_uvs
				else:
					verts = torch.cat((verts, sub_verts))
					faces = torch.cat((faces, sub_faces + v_i_end))
					verts_uvs = torch.cat((verts_uvs, sub_verts_uvs))
					faces_uvs = torch.cat((faces_uvs, sub_faces_uvs + vt_i_end))

				# Calculate and set the sub-mesh data
				sub_mesh_data.set_start_index(v_i_end, vt_i_end, vn_i_end, f_i_end)
				v_i_end += sub_verts.shape[0]
				vt_i_end += torch.cat(sub_mesh.textures.verts_uvs_list()).shape[0]
				vn_i_end += sub_mesh.verts_normals_packed().shape[0]
				f_i_end += sub_mesh.faces_packed().shape[0]
				sub_mesh_data.set_end_index(v_i_end, vt_i_end, vn_i_end, f_i_end)

				# Get the sub-mesh material and texture
				if "material" in primitive:
					material_index = primitive["material"]
					material_json = materials[material_index]
					material_name = material_json["name"]

					material_roughness = material_json["pbrMetallicRoughness"]
					if "baseColorTexture" in material_roughness:
						texture_index = material_roughness["baseColorTexture"]["index"]
						texture_json = textures[texture_index]
						image_index = texture_json["source"]
						image_json = images[image_index]	
						image_name = image_json["name"]
						if image_json["mimeType"] == "image/png":
							image_name += ".png"
						elif image_json["mimeType"] == "image/jpeg":
							image_name += ".jpg"

						image_tensor = loader._get_texture_map_image(image_index)
						all_tex_maps[image_name] = image_tensor
				else:
					material_name = "new_material"
					image_name = "new_texture.png"

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

	return tex_to_vt_map, tex_to_mat_map, all_tex_maps, sub_mesh_data_list, verts, faces, verts_uvs, faces_uvs

def test_load_glb_mesh():
	#mesh_path = "/home/li/Softwares/FlexiSyncMVD/FlexiSyncMVD/data/marcille/Marcille_All_UVs.glb"
	mesh_path = "/home/li/Softwares/FlexiSyncMVD/FlexiSyncMVD/data/marcille/_exp/MVD_01Dec2024-140042/test.glb"
	#mesh = load_glb_mesh(mesh_path)
	#assert mesh.verts_packed().shape[1] == 3
	#assert mesh.faces_packed().shape[1] == 3
	#assert mesh.textures is not None

	load_glb_meshes(mesh_path)

if __name__ == "__main__":
	test_load_glb_mesh()
