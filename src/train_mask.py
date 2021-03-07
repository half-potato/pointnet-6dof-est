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
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=Path, default=Path("/data/haosu"))
parser.add_argument('--save_dir', type=Path, default=Path("./"))
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--max_points', default=30000, type=int)
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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
#  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
ckpt = torch.load("mask_chpt.pth")
model.load_state_dict(ckpt["model"])

#  points = Variable(torch.rand(32,6,2500)).to(device)

print("Reading data")
loaders, point_number_groups = create_loaders('caches/frustrum_cache.hdf5', calib)
dataloaders = [torch.utils.data.DataLoader(loader, batch_size=int(args.max_points/loader.n+1), shuffle=True, num_workers=0, drop_last=False) for loader in loaders if len(loader) > 1]
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.7)

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

    for dataloader in dataloaders:
        for points, labels, label in dataloader:
            # Data preproc
            #  print(points[:, 0, 0].sort().values)
            points = points.to(device)
            labels = labels.to(device)
            N = points.shape[2]

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
            accuracy += float((out_mask.max(dim=-1).indices == labels.reshape(-1)).float().mean())
            n += 1
    scheduler.step()
    print(f"{epoch}: Total loss: {total_loss/n}, Accuracy: {accuracy/n}")
    save()

save()
