import os

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


def get_obj_sub_mesh_data(obj_path):
	data_dir = "./"
	if isinstance(obj_path, (str, bytes)):
		# pyre-fixme[6]: For 1st argument expected `PathLike[Variable[AnyStr <:
		#  [str, bytes]]]` but got `Union[Path, bytes, str]`.
		data_dir = os.path.dirname(obj_path)

	# Parse the .obj file to get the material to vertex uv map.
	with open(obj_path, "r") as f:
		mesh_data_list = []
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
					sub_mesh_data.set_end_index(v_i, vt_i, vn_i, f_i)
					mesh_data_list.append(sub_mesh_data)
				
				sub_mesh_data = SubMeshData(sub_mesh_name)
				sub_mesh_data.set_start_index(v_i, vt_i, vn_i, f_i)

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
			sub_mesh_data.set_end_index(v_i, vt_i, vn_i, f_i)
			sub_mesh_data.material = "new_material"
			mesh_data_list.append(sub_mesh_data)

	return mesh_data_list

def test():
	obj_path = "/home/li/Softwares/FlexiSyncMVD/FlexiSyncMVD/data/marcille/Marcille_all_UVs.obj"
	mesh_data_list = get_obj_sub_mesh_data(obj_path)
	for mesh_data in mesh_data_list:
		print(f"SubMeshData: {mesh_data.name}")
		print(f"Vertex start2end: {mesh_data.v_i_start2end}")
		print(f"Vertex texture start2end: {mesh_data.vt_i_start2end}")
		print(f"Vertex normal start2end: {mesh_data.vn_i_start2end}")
		print(f"Face start2end: {mesh_data.f_i_start2end}")
		print(f"Material: {mesh_data.material}")
		print(f"---------------------------------\n\n")

if __name__ == "__main__":
	test()