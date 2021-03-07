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

base_path = Path("/data/haosu")


device = torch.device("cuda")
pc_loader = loader.TrainLoader(base_path, "test", [20, 500, 1000])
dataset = torch.utils.data.DataLoader(pc_loader, num_workers=16, batch_size=32, collate_fn=loader.collate_fn, shuffle=True)

meshes = mesh_sampler.MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
meshes.load_cache()
NUM_OBJECTS = len(meshes)

model = Simple6DOF(NUM_OBJECTS)
model.to(device)
ckpt = torch.load("checkpoint.pth")
model.load_state_dict(ckpt["model"])
model.eval()

data = {}
with torch.no_grad():
    #  for i in tqdm(range(len(pc_loader))):
    for i, batches in enumerate(dataset):
        #  batches = pc_loader[i]
        for j in range(0, len(batches), pc_loader.num_g):
            (points, _, _, idx, names, scales, prefixes) = batches[j:j+pc_loader.num_g]
            if points == []:
                continue
            #  for
            B = points.shape[0]
            points = points.to(device)

            centers = points.mean(dim=2, keepdim=True)
            #  centers = torch.cat([meshes.get_center(name) for name in names], dim=0).reshape(-1, 3, 1).to(device)
            a1, a2, ts = model(points-centers)
            rots = rotation_from_vectors(a1, a2, NUM_OBJECTS)
            inds = torch.arange(B)
            Rpred = rots[inds, idx].cpu()
            tpred = (ts[inds, idx]+centers.squeeze(2)).cpu()

            for k, (R, t, prefix) in enumerate(zip(Rpred, tpred, prefixes)):
                if prefix not in data:
                    data[prefix] = {
                        "poses_world": [None for l in range(loader.NUM_OBJECTS)]
                    }
                RT = torch.eye(4)
                RT[:3, :3] = R
                RT[:3, 3] = t
                data[prefix]['poses_world'][idx[k]] = RT.tolist()

    with open("pointnet.json", "w") as f:
        json.dump(data, f)
