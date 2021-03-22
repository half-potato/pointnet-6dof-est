import loader
import mesh_sampler
import numpy as np
import open3d
from pathlib import Path
import cpd
import icp
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

base_path = Path("/data/haosu")
#  dataset = loader.PCDDataset(base_path, "test")
meshes = mesh_sampler.MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
dataset = loader.PCDDataset(base_path, 'test_final', use_full=True, meshes=meshes)
#  meshes.build(1000)
#  meshes.save_cache()
meshes.load_cache()
data = {}
it_sampler = mesh_sampler._iterative_furthest_point_sampling
sampler = mesh_sampler.voxel_down_sample

def estimate(ref, ref_cols, scan, scan_cols, N, config, viz=False):
    scan, scan_cols = sampler(scan, N, scan_cols)
    ref, ref_cols = sampler(ref, N, ref_cols)
    max_N = 1000
    #  print(scan.shape[0], ref.shape[0])
    if scan.shape[0] > max_N:
        scan, scan_cols = it_sampler(scan, max_N, scan_cols)
    if ref.shape[0] > max_N:
        ref, ref_cols = it_sampler(ref, max_N, ref_cols)


    algo = cpd.CPD(scan, ref, config, featX=scan_cols, featY=ref_cols)
    loss = algo.estimate(viz)
    return algo, loss

for objects, prefix in tqdm(dataset, total=len(dataset)):
    data[prefix] = {
        "poses_world": [None for i in range(loader.NUM_OBJECTS)]
    }
    for obj in objects:
        ref = meshes.cloud_pts[obj['object_name']] * obj['scale']
        ref_cols = meshes.cloud_cols[obj['object_name']][:, :3]
        ref_norms = meshes.cloud_norms[obj['object_name']]
        scan = obj['points']
        scan_cols = obj['colors']
        if scan.shape[0] == 0:
            continue
        # Get size of cloud for voxel down sample
        bounds = np.array(meshes.get_bounds(obj['object_name'])) * obj['scale'].reshape(-1, 1)
        max_dim = max([abs(max_v - min_v) for min_v, max_v in bounds])
        # estimate norms using open3d
        #  scan_pc = open3d.geometry.PointCloud()
        #  scan_pc.points = open3d.utility.Vector3dVector(scan)
        #  scan_pc.estimate_normals()
        #  scan_norms = np.asarray(scan_pc.normals)

        algo, loss = estimate(ref, ref_cols, scan, scan_cols, max_dim/6, cpd.fast_config)
        #  algo.visualize_points()
        #  if abs(algo.s-1) > 0.15 or loss < 10000:
        #      print(f"s: {algo.s}, Trying symmetric config")
        #      # bad estimate, try again with more points
        #      algo, loss = estimate(ref, ref_cols, scan, scan_cols, 0.01, cpd.symmetric_config)
        #      #  algo.visualize_points()
        if abs(algo.s-1) > 0.15 or loss < 10000:
            #  print(f"s: {algo.s}, loss: {loss} Trying color config")
            # bad estimate, try again with more points
            new_algo, new_loss = estimate(ref, ref_cols, scan, scan_cols, max_dim/8, cpd.color_config)
            #  algo.visualize_points()
            algo = algo if new_loss < loss else new_algo
        elif abs(algo.s-1) > 0.15 or loss < 10000:
            #  print(f"s: {algo.s}, Everything failed")
            pass
        else:
            #  print("Fast config succeeded")
            pass
            #  algo.visualize_points()
            #  algo = estimate(ref, ref_cols, scan, scan_cols, 1000, cpd.color_config, viz=True)
        algo.refine()
        #  algo.visualize_points()
        RT = algo.get_transform()
        data[prefix]['poses_world'][obj['object_id']] = RT.tolist()

        #  print(RT)
        #  print(obj['pose'])

        #  sample2 = open3d.geometry.PointCloud()
        #  sample2.points = open3d.utility.Vector3dVector(ref)
        #  sample2.estimate_normals()
        #
        #  sample1 = open3d.geometry.PointCloud()
        #  sample1.points = open3d.utility.Vector3dVector(scan)
        #  sample1.estimate_normals()
        #  sample1.transform(RT)
        #  open3d.visualization.draw_geometries([sample1, sample2])
        #
        #  algo = icp.ICP(sample1, sample2)
        #  RT = algo.estimate()
        #  open3d.visualization.draw_geometries([sample1, sample2])
        #

with open("cpd.json", "w") as f:
    json.dump(data, f)
