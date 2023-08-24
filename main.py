import numpy as np
import pymeshlab
import laspy

# Config these variable to run
path = "Input.obj" # path to obj file
percent = pymeshlab.Percentage(0.5) # Use in line 16 and 60, no need to change
face_with_same_normal = 500 # minimum number of face with same normal to create a new mesh
point_multiply = 25 # number of point you want to create

# Create and load mesh
ms = pymeshlab.MeshSet()
ms.load_new_mesh(path)
ms.set_current_mesh(0)
mss = ms.current_mesh()
ms.transfer_texture_to_color_per_vertex(sourcemesh = 0,targetmesh = 0,upperbound = percent)

# Read data in mesh
original_colour = mss.vertex_color_matrix()
original_vertex = mss.vertex_matrix()

# Collect normal that have many face_with_same_normal and divive them into different mash
face_normal = mss.face_normal_matrix()
unique_face_normal,count = np.unique(face_normal,axis = 0 ,return_counts=True)
same_normal = unique_face_normal[count>face_with_same_normal]

for normal in same_normal:
    condition = 'fnx == ' + str(normal[0]) + ' && fny == ' + str(normal[1]) + ' && fnz == ' + str(normal[2]) +''
    ms.compute_selection_by_condition_per_face(condselect = condition)
    ms.generate_from_selected_faces()
    ms.set_current_mesh(0)


# Sampling the mashes to create more point
for i in range(0,ms.__len__()):
    print("===================================")
    if (i>0):
        print("Mesh "+str(i)+" contain faces with normal:",end = " ")
        print(same_normal[i-1])
    else:
        print("This mesh contain faces that has small number of face per normal: ")
    print("This mesh has:",end = ' ')
    ms.set_current_mesh(i)
    sample_num = ms.current_mesh().vertex_number() * point_multiply

    print(ms.current_mesh().vertex_number(), "vertices", ms.current_mesh().face_number(), "faces")
    print("After sampling, this mesh has:", end=" ")
    ms.generate_sampling_poisson_disk(samplenum=sample_num, refineflag=True)
    print(ms.current_mesh().vertex_number(), "vertices", ms.current_mesh().face_number(), "faces")

# Merge meshes into one final mesh
ms.generate_by_merging_visible_meshes(mergevisible = False)
submesh_count = int(same_normal.size / 3) * 2 + 2
ms.set_current_mesh(submesh_count)
print("After merging all the mesh, the final mesh has:",end = " ")
print(ms.current_mesh().vertex_number(), "vertices", ms.current_mesh().face_number(), "faces")

# Transfer the texture from faces to points
ms.load_new_mesh(path)
ms.transfer_texture_to_color_per_vertex(sourcemesh = submesh_count+1,targetmesh = submesh_count,upperbound = percent)

# Add classification to know which point is the original point. Original point will have classification value of 19
# created point will have value 0
mss = ms.mesh(submesh_count)
colour = mss.vertex_color_matrix()
vertex = mss.vertex_matrix()

colour = np.concatenate((original_colour,colour),axis = 0)
vertex = np.concatenate((original_vertex,vertex),axis = 0)

indexes = np.unique(vertex, axis = 0, return_index=True)[1]
vertex = [vertex[index,:] for index in sorted(indexes)]
vertex = np.array(vertex)

colour = [colour[index,:] for index in sorted(indexes)]
colour = np.array(colour)

offset = np.min(vertex,axis = 0)

vertex = np.c_[vertex,np.zeros(vertex.shape[0])]
vertex[:original_vertex.shape[0],3] = 19

colour = (np.floor(colour * 255)).astype(int)

# Create a laspy and save points into las file
header = laspy.LasHeader(point_format=7, version="1.4")
header.offsets = offset
header.scales = np.array([0.01, 0.01, 0.01])

las = laspy.LasData(header)

las.x = vertex[:, 0]
las.y = vertex[:, 1]
las.z = vertex[:, 2]

las.classification = vertex[:,3]

las.red = colour[:, 0]
las.green = colour[:, 1]
las.blue = colour[:, 2]

las.write("Output.las")