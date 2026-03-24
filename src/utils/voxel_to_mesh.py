import numpy as np
from skimage import measure
import trimesh


def voxel_to_mesh(voxel_grid, threshold=0.5):
    """
    Convert voxel grid to mesh using marching cubes
    """

    voxel_grid = voxel_grid.squeeze()  # [32,32,32]

    # Marching cubes
    verts, faces, normals, _ = measure.marching_cubes(voxel_grid, level=threshold)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    return mesh


def save_mesh(mesh, filename="output.obj"):
    mesh.export(filename)