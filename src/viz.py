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
from torch.autograd import Variable

base_path = Path("/data/haosu")


device = torch.device("cuda:0")
pc_loader = loader.TrainLoader(base_path, "train", [20, 500, 1000])
#  batch_accum = loader.BatchAccumulator(pc_loader, 128)

meshes = mesh_sampler.MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
meshes.load_cache()
NUM_OBJECTS = len(meshes)

model = Simple6DOF(NUM_OBJECTS)
model.to(device)
ckpt = torch.load("checkpoint.pth")
model.load_state_dict(ckpt["model"])
#  model.load_state_dict(torch.load("malfn.pth"))
loss_fn = losses.geodesic_distance
loss_fn = losses.shape_aware_loss
model.eval()

total_loss = 0
rng = range(1070*20,1070*20+10)
#  rng = range(0, 40)
for i in rng:
    batches = pc_loader[i]
    geoms = []
    for j in range(0, len(batches), pc_loader.num_g):
        (points, Rgt, tgt, idx, names, scales) = batches[j:j+6]
        if points == []:
            continue
        B = points.shape[0]
        points = points.to(device)
        Rgt = Rgt.to(device)
        tgt = tgt.to(device)

        centers = points.mean(dim=2, keepdim=True)
        #  centers = torch.cat([meshes.get_center(name) for name in names], dim=0).reshape(-1, 3, 1).to(device)
        a1, a2, ts = model(points-centers)
        rots = rotation_from_vectors(a1, a2, NUM_OBJECTS)
        inds = torch.arange(B)
        Rpred = rots[inds, idx]
        tpred = ts[inds, idx]+centers.squeeze(2)
        #  print(Rgt, Rpred)
        #  print(tpred, tgt)

        #  break
        # Draw prediction
        for j, pc in enumerate(points.cpu().numpy()):
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(pc.T)
            geoms.append(pcd)

            trans = torch.eye(4)
            #  trans[:3, :3] = Rpred[j]
            #  trans[:3, 3] = tpred[j]
            syms = meshes.get_symmetries(names[j])
            #  print(Rgt[j], Rpred[j])
            for rot in losses.gen_group_products(syms):
                #  trans[:3, :3] = rot.cpu() @ Rgt[j].cpu()
                #  trans[:3, :3] = Rgt[j].cpu() @ rot.cpu()
                #  trans[:3, :3] = Rpred[j].cpu() @ rot.cpu()
                # Model predicts: Rt from base reference frame to points reference frame
                trans[:3, :3] = Rpred[j].cpu()
                trans[:3, 3] = tpred[j]
                pcd = open3d.geometry.PointCloud()

                obj_data, _ = pc_loader.pcddata[i]

                pcd.points = open3d.utility.Vector3dVector(meshes.cloud_pts[names[j]]*scales[j])
                pcd.transform(trans.detach().cpu().numpy())
                geoms.append(pcd)
    open3d.visualization.draw_geometries(geoms)


    rot_loss = losses.symmetric_loss(loss_fn, Rpred, Rgt, [meshes.get_symmetries(name) for name in names], X=torch.bmm(Rgt.transpose(1, 2), points))
    #  rot_loss = loss_fn(Rpred, Rgt, X=torch.bmm(Rgt.transpose(1, 2), points)).sum()
    trans_loss = torch.norm(tpred-tgt, dim=1).mean()
    norm_loss = losses.margin_regularizer(a1) + losses.margin_regularizer(a2)
    loss = rot_loss + trans_loss + norm_loss
    print(f"{i}: Average rot error: {rot_loss/math.pi*180 / B}, Average trans error: {100*trans_loss.item()}, Norm loss: {norm_loss.item()}")
