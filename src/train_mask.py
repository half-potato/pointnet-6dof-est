from models.simple_mask import SimpleMask
from models.pointnet2_sem_seg import get_model, get_loss
import mesh_sampler
import open3d
from pathlib import Path
import torch
from icecream import ic
from torch.autograd import Variable
import time
import argparse
from frustrum_loader import create_loaders
import utils
import random
#  torch.autograd.set_detect_anomaly(True)
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=Path, default=Path("/data/haosu"))
parser.add_argument('--save_dir', type=Path, default=Path("./"))
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--max_points', default=30000, type=int)
parser.add_argument('--draw_output', action='store_true')
parser.add_argument('--cache_location', default=Path('/data/haosu/frustrum_cache.hdf5'), type=Path)

args = parser.parse_args()

base_path = args.base_path
device = torch.device("cuda:0")
meshes = mesh_sampler.MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
NUM_OBJECTS = len(meshes)+1 # for background
calib = [514.13902522, 514.13902522, 640, 360, 1280, 720]

#  loss_fn = torch.nn.CrossEntropyLoss()
#  model = SimpleMask(6, NUM_OBJECTS).to(device)
loss_fn = torch.nn.NLLLoss(reduction='sum')
#  loss_fn = get_loss()
model = get_model(6, NUM_OBJECTS).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#  ckpt = torch.load("/data/haosu/mask_chpt.pth")
ckpt = torch.load("mask_chpt.pth")
model.load_state_dict(ckpt["model"])
#  model.eval()

#  points = Variable(torch.rand(32,6,2500)).to(device)

print("Reading data")
loaders, point_number_groups, data = create_loaders(args.cache_location, calib)

print([len(loader) for loader in loaders])
dataloaders = [torch.utils.data.DataLoader(loader, batch_size=int(args.max_points//loader.n), shuffle=True, num_workers=1, drop_last=False) for loader in loaders if len(loader) > 1 and int(args.max_points//loader.n) > 0]
#  dataloaders = [torch.utils.data.DataLoader(loader, batch_size=1, shuffle=True, num_workers=1, drop_last=False) for loader in loaders if len(loader) > 1 and int(args.max_points//loader.n) > 0]
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.7)

def save():
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
    }, args.save_dir / "mask_chpt.pth")

#  vals = torch.rand(3, NUM_OBJECTS, 1000, requires_grad=True, device=device)
#  optimizer = torch.optim.Adam([vals], lr=1e-1)
#  labels = torch.zeros((32, 2500), dtype=torch.long).to(device)
#  labels[:, :] = 1

print("Started")
for epoch in range(args.epochs):
    total_loss = 0
    accuracy = 0
    n = np.zeros((1), np.uint64)

    dsl = [iter(dataloader) for dataloader in dataloaders]
    dsi = list(range(len(dsl)))
    weights = [len(dataloader)/dataloader.batch_size for dataloader in dataloaders]

    while len(dsi) > 0:
        i = random.choices(list(range(len(dsi))), weights=weights, k=1)[0]
        di = dsi[i]
        try:
            points, labels = dsl[di].next()
            weights[i] -= 1
        except:
            del dsi[i]
            del weights[i]
            continue

        # sample points to smooth out gaps
        #  origN = points.shape[2]
        #  Ni = list(point_number_groups).index(origN)
        #  prevN = point_number_groups[Ni-1] if Ni > 0 else 32
        #  N = random.randint(prevN, origN)
        #  samp_ind = np.random.choice(origN, size=N)
        #  points = points[:, :, samp_ind].to(device)
        #  labels = labels[:, samp_ind].to(device)

        points = points.to(device)
        labels = labels.to(device)

        if points.shape[0] < 1:
            continue

        #  utils.draw_augpoints([points.cpu().numpy()[0]], False, True)
        mask = model(points)
        #  mask = torch.nn.Softmax(1)(vals)
        #  out_mask = mask.transpose(1, 2).reshape(-1, NUM_OBJECTS)
        out_mask = mask.reshape(-1, NUM_OBJECTS)
        #  print(torch.exp(out_mask).sum(dim=-1))
        loss = loss_fn(out_mask, labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #  print(float(loss))

        total_loss += float(loss)
        pred = out_mask.max(dim=-1).indices
        acc = float((pred == labels.reshape(-1)).float().sum())
        accuracy += acc
        #  print(f'{float(loss)} {(float(acc))}')
        n += pred.shape[0]

        # Draw label
        if args.draw_output:
            for i in range(points.shape[0]):
                indpred = pred.reshape(points.shape[0], points.shape[2])[i]
                acc = float((indpred == labels[i]).float().mean())
                indpred = indpred.cpu().numpy()
                disp = points.cpu().numpy()[i]
                disp[3, :] = indpred/NUM_OBJECTS
                disp[4, :] = indpred/NUM_OBJECTS
                disp[5, :] = indpred/NUM_OBJECTS
                print(f'Accuracy: {acc}')
                utils.draw_augpoints([disp], False, True)

    print(f"{epoch}: Total loss: {total_loss/n}, Accuracy: {accuracy/n}, LR: {scheduler.get_lr()}, N: {n}")
    scheduler.step()
    if n > 0:
        save()

save()
data.close()
