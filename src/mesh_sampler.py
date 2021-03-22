import numpy as np
import random
import trimesh
import pandas as pd
from pathlib import Path
import open3d
from tqdm import tqdm
import torch

def iterative_furthest_point_sampling(points, K):
    i = random.randint(0, points.shape[0]-1)
    #  s = [points[i]]
    s = [i]
    dists = ((points[s[-1]].reshape(-1, 1, 3) - points.reshape(1, -1, 3))**2).sum(axis=-1)
    dists_to_s = np.min(dists, axis=0).squeeze()
    for i in range(1, K):
        # Calculate distance to all points
        # Shape: (len(s), points.shape[0])
        dists = ((points[s[-1]].reshape(-1, 1, 3) - points.reshape(1, -1, 3))**2).sum(axis=-1)
        dist_to_s = np.min(dists, axis=0).squeeze()
        dists_to_s = np.minimum(dists_to_s, dist_to_s)
        ind = np.argmax(dists_to_s)
        s.append(ind)
    return np.stack(s, axis=0)

def _iterative_furthest_point_sampling(points, K, colors=None, normals=None):
    i = random.randint(0, points.shape[0]-1)
    s = [points[i]]
    if colors is not None:
        cols = [colors[i]]
    if normals is not None:
        norms = [normals[i]]
    dists = ((s[-1].reshape(-1, 1, 3) - points.reshape(1, -1, 3))**2).sum(axis=-1)
    dists_to_s = np.min(dists, axis=0).squeeze()
    for i in range(1, K):
        # Calculate distance to all points
        # Shape: (len(s), points.shape[0])
        dists = ((s[-1].reshape(-1, 1, 3) - points.reshape(1, -1, 3))**2).sum(axis=-1)
        dist_to_s = np.min(dists, axis=0).squeeze()
        dists_to_s = np.minimum(dists_to_s, dist_to_s)
        ind = np.argmax(dists_to_s)
        s.append(points[ind])
        if colors is not None:
            cols.append(colors[ind])
        if normals is not None:
            norms.append(normals[ind])
    out = [np.stack(s, axis=0)]
    if colors is not None:
        out.append(np.stack(cols, axis=0))
    if normals is not None:
        out.append(np.stack(norms, axis=0))
    return out

def voxel_down_sample(points, voxel_size, colors=None, normals=None):
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points)
    if colors is not None:
        pc.colors = open3d.utility.Vector3dVector(colors)
    if normals is not None:
        pc.normals = open3d.utility.Vector3dVector(normals)
    vpc = pc.voxel_down_sample(voxel_size)
    out = [np.asarray(vpc.points)]
    if colors is not None:
        out.append(np.asarray(vpc.colors))
    if normals is not None:
        out.append(np.asarray(vpc.normals))
    return out


class MeshSampler:
    def __init__(self, base_path, csv_path):
        self.metadata = pd.read_csv(csv_path)
        self.base_path = base_path

    def build(self, num_points):
        # load objects
        self.cloud_pts = {}
        self.cloud_norms = {}
        self.cloud_cols = {}
        for (model_name, model_path) in tqdm(zip(self.metadata["object"], self.metadata["location"])):
            print(model_name)
            path = str(self.base_path / model_path / "visual_meshes" / "visual.dae")
            kwargs = trimesh.exchange.dae.load_collada(path)
            mesh = trimesh.load(path, **kwargs)
            name = list(mesh.geometry.keys())[0]
            mesh = mesh.geometry[name]
            sample_pts, inds = trimesh.sample.sample_surface(mesh, num_points*100)
            try:
                # retrieve color
                colvis = mesh.visual.to_color()
                colvis.mesh = mesh
                colors = np.asarray(colvis.face_colors[inds])[:, :3]/255
            except:
                colors = np.zeros((sample_pts.shape[0], 3))
                colors[:, 0] = 1
            # retrieve normals
            normals = mesh.face_normals[inds]

            #  subsamp, subcolor, subnorm = iterative_furthest_point_sampling(sample_pts, num_points, colors=colors, normals=normals)
            subsamp, subcolor, subnorm = voxel_down_sample(sample_pts, 0.001, colors=colors, normals=normals)
            self.cloud_pts[model_name] = subsamp
            self.cloud_cols[model_name] = subcolor
            self.cloud_norms[model_name] = subnorm

    def __len__(self):
        return self.metadata.shape[0]

    def idx_to_name(self, i):
        return self.metadata.iloc[i]['object'].item()

    def get_meta(self, name):
        return self.metadata[self.metadata['object']==name]

    def get_center(self, name):
        return torch.tensor([b - a for a, b in self.get_bounds(name)]).reshape(1, 3)

    def get_bounds(self, name):
        meta = self.get_meta(name)
        bounds = [(meta['min_x'], meta['max_x']),
                  (meta['min_y'], meta['max_y']),
                  (meta['min_z'], meta['max_z'])]
        return [(float(a), float(b)) for a, b, in bounds]

    def iter(self):
        return self.cloud_pts.keys()

    def get_symmetries(self, name):
        meta = self.get_meta(name)
        return meta['geometric_symmetry'].item()

    def save_cache(self):
        np.savez("caches/model_point_cache.npz", **self.cloud_pts)
        np.savez("caches/model_color_cache.npz", **self.cloud_cols)
        np.savez("caches/model_normal_cache.npz", **self.cloud_norms)

    def load_cache(self):
        self.cloud_pts = np.load("caches/model_point_cache.npz")
        self.cloud_cols = np.load("caches/model_color_cache.npz")
        self.cloud_norms = np.load("caches/model_normal_cache.npz")

if __name__ == "__main__":
    base_path = Path("/data/haosu")
    meshes = MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
    #  meshes.build(1000)
    meshes.build(1000)
    meshes.save_cache()
    meshes.load_cache()
