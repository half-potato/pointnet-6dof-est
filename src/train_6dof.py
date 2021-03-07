import loader
import mesh_sampler
import open3d
from pathlib import Path
from models.simple_6dof import Simple6DOF, rotation_from_vectors
import loss as losses
import torch
from icecream import ic
from torch.autograd import Variable
import time

base_path = Path("/data/haosu")

device = torch.device("cuda:0")
pc_loader = loader.TrainLoader(base_path, "train", [20, 500, 1000])
#  batch_accum = loader.BatchAccumulator(pc_loader, 128)
dataset = torch.utils.data.DataLoader(pc_loader, num_workers=16, batch_size=32, collate_fn=loader.collate_fn, shuffle=True)
#  dataset = torch.utils.data.DataLoader(pc_loader, num_workers=16, batch_size=4, collate_fn=loader.collate_fn, shuffle=True)

meshes = mesh_sampler.MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
meshes.load_cache()
NUM_OBJECTS = len(meshes)

model = Simple6DOF(NUM_OBJECTS)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

ckpt = torch.load("/root/data/haosu/checkpoint.pth")
#  ckpt = torch.load("checkpoint.pth")
model.load_state_dict(ckpt["model"])

optimizer.load_state_dict(ckpt["optim"])

#  vecs = torch.rand(95, 79, 2, 3).to(device)
#  vecs.requires_grad = True
#  a1 = vecs.reshape(-1, 2, 3)[:, 0]
#  a2 = vecs.reshape(-1, 2, 3)[:, 1]
#  rots = rotation_from_vectors(a1, a2, NUM_OBJECTS)
#
#  ts = torch.rand(95, 79, 3).to(device)
#  ts.requires_grad = True
#
#  optimizer = torch.optim.Adam([ts, vecs], lr=3e-2)

#  loss_fn = losses.geodesic_distance
loss_fn = losses.shape_aware_loss

def save():
    print("Saving")
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
    }, "/root/data/haosu/checkpoint.pth")
    #  }, "checkpoint.pth")
    print("Done")

#  for i, (points, Rgt, tgt, idx, names, scales) in enumerate(dataset):
#      if i > 0:
#          break
for epoch in range(500):
    #  (points, Rgt, tgt, idx, names, scales) = loader.collate_fn([pc_loader[i] for i in range(32)])
    total_loss = 0
    #  for i, (points, Rgt, tgt, idx, names, scales) in enumerate(dataset):
    for i, batches in enumerate(dataset):
        #  batches = pc_loader[i]
        for j in range(0, len(batches), pc_loader.num_g):
            (points, Rgt, tgt, idx, names, scales, _) = batches[j:j+pc_loader.num_g]
            for k in range(5):
                # B, 3, N
                B = points.shape[0]
                points = points.to(device)
                Rgt = Rgt.to(device)
                tgt = tgt.to(device)
                if B == 1:
                    continue

                centers = points.mean(dim=2, keepdim=True)
                #  centers = torch.cat([meshes.get_center(name) for name in names], dim=0).reshape(-1, 3, 1).to(device)
                a1, a2, ts = model(points-centers)
                rots = rotation_from_vectors(a1, a2, NUM_OBJECTS)
                inds = torch.arange(B)
                Rpred = rots[inds, idx]
                tpred = ts[inds, idx]+centers.squeeze(2)

                # Loss
                # Model predicts: Rt from base reference frame to points reference frame
                # Therefore, we test the loss on the points in the base reference frame
                #  start_time = time.time()
                rot_loss = losses.symmetric_loss(loss_fn, Rpred, Rgt, [meshes.get_symmetries(name) for name in names], X=torch.bmm(Rgt.transpose(1, 2), points))
                #  rot_loss = loss_fn(Rpred, Rgt, X=torch.bmm(Rgt.transpose(1, 2), points)).sum()
                #  rot_loss = losses.symmetric_loss(loss_fn, Rpred, Rgt, [meshes.get_symmetries(name) for name in names])
                trans_loss = torch.norm(tpred-tgt, dim=1).sum()
                norm_loss = losses.margin_regularizer(a1) + losses.margin_regularizer(a2)
                loss = rot_loss + trans_loss# + 0.01*norm_loss
                #  print(f"Time to loss: {time.time()-start_time}")
                total_loss += float(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if i % 100 == 0:
        #      print(f"{i}: Rot loss: {float(rot_loss)}, Trans loss: {float(trans_loss)}, Norm loss: {float(norm_loss)}")
        #      #  save()
            print(f"{i}: Total loss: {total_loss}")
            total_loss = 0
            save()

save()
