import os

def get_obj_tex_to_vt_map(self, obj_path):
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
				tex_to_vt_map = {"new_texture.png": [(0, vt_i)]}
				tex_to_mat_map = {"new_texture.png": ["new_material"]}

			elif mtl_path is not None:
				# Parse the .mtl file to get the texture to material map.
				with open(mtl_path, "r") as f:
					tex_to_mat_map = {}
					all_materials = set()
					last_material_name = None
					for line in f:
						tokens = line.strip().split()
						if not tokens:
							continue
						if tokens[0] == "newmtl":
							last_material_name = tokens[1]
							all_materials.add(last_material_name)
						elif tokens[0] == "map_Kd":
							# Diffuse texture map
							# Account for the case where filenames might have spaces
							image_name = os.path.basename(line.strip()[7:])
							if image_name not in tex_to_mat_map:
								tex_to_mat_map[image_name] = []
							tex_to_mat_map[image_name].append(last_material_name)
							all_materials.remove(last_material_name)

					for material in all_materials:
						tex_to_mat_map[material + "_new_texture.png"] = [material]

				# Combine the two maps to get the texture to vertex uv map.
				tex_to_vt_map = {}
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
		tex_to_vt_map = None
		tex_to_mat_map = None

	return tex_to_vt_map, tex_to_mat_map

def test():
	obj_path = "test.obj"
	with open(obj_path, "w") as f:
		f.write("# www.blender.org\n")
		f.write("mtllib test.mtl\n")
		f.write("vt 1 2\n")
		f.write("vt 3 4\n")
		f.write("v 5 6 7\n")
		f.write("vt 8 9\n") # 2
		f.write("vt 10 11\n")
		f.write("vt 12 13\n") # 4
		f.write("s 1\n")
		#f.write("usemtl MarcilleSclera\n")
		f.write("f 1/1 2/2 3/3\n")
		f.write("f 4/4 5/5 6/6\n")
		f.write("f 7/7 8/8 9/9\n")
		f.write("vt 1 2\n") # 5
		f.write("vt 3 4\n") # 6
		#f.write("usemtl MarcilleSclera_01\n")
		f.write("vt 10 11\n") # 7
		f.write("s 1\n")
		#f.write("usemtl MarcilleSclera_02\n")
		f.write("v 5 6 7\n")
		f.write("v 5 6 7\n")
		f.write("v 9 1 2\n")
		f.write("vt 8 9\n") # 8
		f.write("vt 10 11\n")
		f.write("vt 12 13\n") # 10
		f.write("s 1\n")
		#f.write("usemtl MarcilleSclera\n")
		f.write("f 1/1 2/2 3/3\n")
		f.write("f 4/4 5/5 6/6\n")
		f.write("f 7/7 8/8 9/9\n")
		f.write("vt 8 9\n") # 11
		f.write("vt 10 11\n")
		f.write("vt 12 13\n") # 13
		f.write("s 1\n")
		f.write("usemtl MarcilleSclera_02\n")

	mtl_path = "test.mtl"
	with open(mtl_path, "w") as f:
		f.write("newmtl MarcilleSclera\n")
		f.write("Kd 0.800000 0.800000 0.800000\n")
		f.write("map_Kd Marcille_all_UVs_textures/Image_0.png\n")
		f.write("newmtl MarcilleSclera_02\n")
		f.write("Kd 0.800000 0.800000 0.800000\n")
		f.write("newmtl MarcilleSclera_01\n")
		f.write("Kd 0.800000 0.800000 0.800000\n")
		f.write("map_Kd Marcille_all_UVs_textures/Image_0.png\n")
		f.write("newmtl MarcilleSclera_03\n")
		f.write("Kd 0.800000 0.800000 0.800000\n")
	

	tex_to_vt_map, tex_to_mat_map = get_obj_tex_to_vt_map(None, obj_path)
	print(f"tex_to_vt_map:\n{tex_to_vt_map};\n\ntex_to_mat_map:\n{tex_to_mat_map}")

	os.remove(obj_path)
	os.remove(mtl_path)

test()