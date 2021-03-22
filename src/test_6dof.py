import math
import loader
import numpy as np
import mesh_sampler
import open3d
from pathlib import Path
from models.simple_6dof import Simple6DOF, rotation_from_vectors
import loss as losses
import torch
from icecream import ic
from tqdm import tqdm
import json
import argparse
#  np.set_printoptions(precision=4)

base_path = Path("/data/haosu")

parser = argparse.ArgumentParser()
parser.add_argument('--use_full', action='store_true')
parser.add_argument('--display', action='store_true')
args = parser.parse_args()

device = torch.device("cuda")
#  pc_loader = loader.TrainLoader(base_path, "test", [20, 500, 1000], device=device)
#  dataset = torch.utils.data.DataLoader(pc_loader, num_workers=1, batch_size=32, collate_fn=loader.collate_fn, shuffle=True)
#  dataset = torch.utils.data.DataLoader(pc_loader, num_workers=1, batch_size=1, collate_fn=loader.collate_fn, shuffle=True)
#  dataset = loader.PCDDataset(base_path, 'train', use_full=args.use_full)
meshes = mesh_sampler.MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
meshes.load_cache()

#  dataset = loader.PCDDataset(base_path, 'train', use_full=args.use_full, meshes=meshes)
dataset = loader.PCDDataset(base_path, 'test_final', use_full=args.use_full, meshes=meshes)
point_number_groups = [1024, 3000, 5000, 7000, 9000, 11000, 15000, 20000, 50000, 100000]

NUM_OBJECTS = len(meshes)

model = Simple6DOF(NUM_OBJECTS)
model.to(device)
#  ckpt = torch.load("pose_ckpt.pth")
ckpt = torch.load("checkpoint.pth")
model.load_state_dict(ckpt["model"])
model.eval()

data = {}
with torch.no_grad():
    for i in tqdm(range(0, len(dataset))):
        # Get frustrums
        objects, prefix = dataset[i]
        B = 1
        geoms = []
        for obj in objects:
            # Get point number group
            scan = obj['points']/obj['scale']
            scan_cols = obj['colors']
            idx = torch.tensor(obj["object_id"]).long().to(device)

            pc = np.concatenate([scan, scan_cols], axis=1)
            M, D = pc.shape
            G = [i for i, p in enumerate(point_number_groups) if M <= p]
            if len(G) == 0:
                continue
            G = G[0]
            N = point_number_groups[G]

            inds = np.random.choice(M, N-M, replace=True)
            #  pc = np.concatenate([pc, pc[inds]], axis=0)
            points = torch.tensor(pc.T).unsqueeze(0).float().to(device)

            centers = points[:, :3].mean(dim=2, keepdim=True)
            points[:, :3] -= centers
            #  centers = torch.cat([meshes.get_center(name) for name in names], dim=0).reshape(-1, 3, 1).to(device)
            a1, a2, ts = model(points[:, :3])
            rots = rotation_from_vectors(a1, a2, NUM_OBJECTS)
            inds = torch.arange(B)
            Rpred = rots[inds, idx].cpu()
            tpred = (ts[inds, idx]+centers.squeeze(2)).cpu()

            #  RTg = obj['pose']
            #  sample2 = open3d.geometry.PointCloud()
            #  sample2.points = open3d.utility.Vector3dVector(ref)
            #  sample2.transform(RTg)

            RT = np.eye(4)
            RT[:3, :3] = Rpred[0].cpu().numpy()
            RT[:3, 3] = tpred[0].cpu().numpy()
            if args.display:
                ref = meshes.cloud_pts[obj['object_name']] * obj['scale']
                sample1 = open3d.geometry.PointCloud()
                sample1.points = open3d.utility.Vector3dVector(ref)
                sample1.transform(RT)
                geoms.append(sample1)

            if prefix not in data:
                data[prefix] = {
                    "poses_world": [None for l in range(loader.NUM_OBJECTS)]
                }
            data[prefix]['poses_world'][obj['object_id']] = RT.tolist()
        if args.display:
            open3d.visualization.draw_geometries(geoms)

    with open("pointnet.json", "w") as f:
        json.dump(data, f)
