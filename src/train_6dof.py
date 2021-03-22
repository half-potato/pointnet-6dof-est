import mesh_sampler
import open3d
from pathlib import Path
from models.simple_6dof import Simple6DOF, rotation_from_vectors
import loss as losses
import torch
from icecream import ic
from torch.autograd import Variable
import time
from object_loader import ObjectLoader
import h5py
import argparse
import random

device = torch.device("cuda:0")

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=Path, default=Path("/data/haosu"))
parser.add_argument('--save_dir', type=Path, default=Path("./"))
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--max_points', default=60000, type=int)
parser.add_argument('--cache_location', default=Path('/data/haosu/object_cache.hdf5'), type=Path)

args = parser.parse_args()

base_path = args.base_path

meshes = mesh_sampler.MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
meshes.load_cache()
NUM_OBJECTS = len(meshes)

model = Simple6DOF(NUM_OBJECTS)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#  ckpt = torch.load("/root/data/haosu/checkpoint.pth")
ckpt = torch.load("pose_ckpt.pth")
model.load_state_dict(ckpt["model"])

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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)

data = h5py.File(args.cache_location, 'r')
point_number_groups = data['point_nums']
#  counts = data['counts']
loaders = []
for i in range(len(point_number_groups)):
    group = data[f'point_group{i}-{point_number_groups[i]}']
    loaders.append(ObjectLoader(group))

dataloaders = [
        torch.utils.data.DataLoader(
            loader,
            batch_size=int(args.max_points//loader.n),
            shuffle=False,
            num_workers=1,
            drop_last=True)
        for loader in loaders if len(loader) > 1 and int(args.max_points//loader.n) > 0]

def save():
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
    }, args.save_dir / "pose_ckpt.pth")
    #  }, "checkpoint.pth")

print("Started")
for epoch in range(args.epochs):
    total_loss = 0
    accuracy = 0
    n = 0

    dsl = [iter(dataloader) for dataloader in dataloaders]
    dsi = list(range(len(dsl)))
    weights = [len(dataloader)/dataloader.batch_size for dataloader in dataloaders]

    while len(dsi) > 0:
        i = random.choices(list(range(len(dsi))), weights=weights, k=1)[0]
        di = dsi[i]
        try:
            points, Rgt, tgt, idxs = dsl[di].next()
            weights[i] -= 1
        except StopIteration:
            del dsi[i]
            del weights[i]
            continue

        B = points.shape[0]
        points = points.to(device)
        Rgt = Rgt.to(device)
        tgt = tgt.to(device).squeeze(2)
        idx = idxs.long().to(device).squeeze()
        if B == 1:
            continue

        centers = points[:, :3].mean(dim=2, keepdim=True)
        points[:, :3] -= centers
        #  centers = torch.cat([meshes.get_center(name) for name in names], dim=0).reshape(-1, 3, 1).to(device)
        for sub_iter in range(1):
            a1, a2, ts = model(points)
            rots = rotation_from_vectors(a1, a2, NUM_OBJECTS)
            inds = torch.arange(B)
            Rpred = rots[inds, idx]
            tpred = ts[inds, idx]+centers.squeeze(2)

            # Loss
            # Model predicts: Rt from base reference frame to points reference frame
            # Therefore, we test the loss on the points in the base reference frame
            #  start_time = time.time()
            unrot = torch.bmm(Rgt.transpose(1, 2), points[:, :3])
            rot_loss = losses.symmetric_loss(
                    loss_fn, Rpred, Rgt,
                    [meshes.get_symmetries(meshes.idx_to_name(idxi)) for idxi in idxs.long()],
                    X=unrot)

            #  rot_loss = loss_fn(Rpred, Rgt, X=torch.bmm(Rgt.transpose(1, 2), points)).sum()
            #  rot_loss = losses.symmetric_loss(loss_fn, Rpred, Rgt, [meshes.get_symmetries(name) for name in names])

            trans_loss = torch.norm(tpred-tgt, dim=1).sum()
            norm_loss = losses.margin_regularizer(a1) + losses.margin_regularizer(a2)
            loss = rot_loss + trans_loss# + 0.01*norm_loss
            #  print(f"Time to loss: {time.time()-start_time}")
            total_loss += float(loss)
            n += B
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #  if epoch % 100 == 0:
    #      print(f"{i}: Rot loss: {float(rot_loss)}, Trans loss: {float(trans_loss)}, Norm loss: {float(norm_loss)}")
    #      #  save()
    print(f"{epoch}: Total loss: {total_loss/n}, LR: {scheduler.get_lr()}")
    total_loss = 0
    save()
    scheduler.step()

save()
