from models.simple_mask import SimpleMask
#  from models.pointnet2_sem_seg import get_model, get_loss
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

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=Path, default=Path("/data/haosu"))
parser.add_argument('--save_dir', type=Path, default=Path("./"))
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--max_points', default=120000, type=int)
parser.add_argument('--cache_location', default=Path('caches/frustrum_cache.hdf5'), type=Path)

args = parser.parse_args()

base_path = args.base_path
device = torch.device("cuda:0")
meshes = mesh_sampler.MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
NUM_OBJECTS = len(meshes)+1 # for background
calib = [514.13902522, 514.13902522, 640, 360, 1280, 720]

model = SimpleMask(6, NUM_OBJECTS).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
#  loss_fn = get_loss()
#  loss_fn = torch.nn.NLLLoss(reduction='sum')
#  model = get_model(NUM_OBJECTS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
ckpt = torch.load("/data/haosu/mask_chpt.pth")
model.load_state_dict(ckpt["model"])

#  points = Variable(torch.rand(32,6,2500)).to(device)

print("Reading data")
loaders, point_number_groups, data = create_loaders(args.cache_location, calib)

dataloaders = [torch.utils.data.DataLoader(loader, batch_size=int(args.max_points//loader.n), shuffle=True, num_workers=1, drop_last=True) for loader in loaders if len(loader) > 1]
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.7)

def save():
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
    }, args.save_dir / "mask_chpt.pth")

weights = torch.ones((NUM_OBJECTS)).to(device)

#  vals = torch.rand(3, NUM_OBJECTS, 1000, requires_grad=True, device=device)
#  optimizer = torch.optim.Adam([vals], lr=1e-1)
#  labels = torch.zeros((32, 2500), dtype=torch.long).to(device)
#  labels[:, :] = 1

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
            points, labels = dsl[di].next()
            weights[i] -= 1
        except:
            del dsi[i]
            del weights[i]

        points = points.to(device)
        labels = labels.to(device)
        N = points.shape[2]
        if points.shape[0] <= 1:
            continue

        #  utils.draw_augpoints([points.cpu().numpy()[0]], False, True)
        mask = model(points)
        #  mask = torch.nn.Softmax(1)(vals)
        #  out_mask = mask.transpose(1, 2).reshape(-1, NUM_OBJECTS)
        out_mask = mask.reshape(-1, NUM_OBJECTS)
        loss = loss_fn(out_mask, labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #  print(float(loss))

        total_loss += float(loss)
        acc = float((out_mask.max(dim=-1).indices == labels.reshape(-1)).float().mean())
        accuracy += acc
        #  print(float(loss))
        #  print(float(acc))
        n += 1
    print(f"{epoch}: Total loss: {total_loss/n}, Accuracy: {accuracy/n}, LR: {scheduler.get_lr()}, N: {n}")
    scheduler.step()
    save()

save()
data.close()
