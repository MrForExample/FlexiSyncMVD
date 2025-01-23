import sys
sys.path.append("/home/li/Softwares/FlexiSyncMVD/FlexiSyncMVD")

from src.renderer.project import GLBWriter
from typing import BinaryIO, cast
import pickle
import json
from pytorch3d.io.experimental_gltf_io import OurEncoder

# We can debug the GLB file using https://github.khronos.org/glTF-Validator/ and https://modelviewer.dev/editor/, both shows detailed error messages
def test_glb_save(mesh_path, json_path, pkl_path):
	with open(pkl_path, 'rb') as inp:
		output_test_dict = pickle.load(inp)

	with open(mesh_path, "wb") as f:
		writer = GLBWriter(cast(BinaryIO, f))
		writer.save(
			output_test_dict["sub_mesh_data_list"],
			output_test_dict["tex_to_mat_map"],
			output_test_dict["verts"],
			output_test_dict["faces"],
			verts_uvs=output_test_dict["verts_uvs"],
			faces_uvs=output_test_dict["face_uvs"],
			all_texture_maps=output_test_dict["split_texture_maps"],
			normals=output_test_dict["normals"],
		)
		open(json_path, "w").write(json.dumps(writer._json_data, cls=OurEncoder, indent=4))

if __name__ == "__main__":
	#mesh_path = "/home/li/Softwares/FlexiSyncMVD/FlexiSyncMVD/data/marcille/_exp/MVD_01Dec2024-140042/test.glb"
	#json_path = "/home/li/Softwares/FlexiSyncMVD/FlexiSyncMVD/data/marcille/_exp/MVD_01Dec2024-140042/test.json"
	#pkl_path = "/home/li/Softwares/FlexiSyncMVD/FlexiSyncMVD/data/marcille/_exp/MVD_01Dec2024-140042/results/textured.pkl"
	mesh_path = "/home/li/Softwares/FlexiSyncMVD/FlexiSyncMVD/data/cammy/_exp/MVD_02Dec2024-185106/test.glb"
	json_path = "/home/li/Softwares/FlexiSyncMVD/FlexiSyncMVD/data/cammy/_exp/MVD_02Dec2024-185106/test.json"
	pkl_path = "/home/li/Softwares/FlexiSyncMVD/FlexiSyncMVD/data/cammy/_exp/MVD_02Dec2024-185106/results/textured.pkl"
	test_glb_save(mesh_path, json_path, pkl_path)