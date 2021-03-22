from models.unet import UNet
from pathlib import Path
import torch
from icecream import ic
import utils
import random
import loader
import argparse
import matplotlib.pyplot as plt
#  torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=Path, default=Path("/data/haosu"))
parser.add_argument('--save_dir', type=Path, default=Path("./"))
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--backward_freq', default=32, type=int)
parser.add_argument('--epochs', default=500, type=int)

args = parser.parse_args()

base_path = args.base_path
device = torch.device("cuda:0")
NUM_OBJECTS = 80
pc_loader = loader.MaskLoader(base_path, "train")
dataset = torch.utils.data.DataLoader(pc_loader, num_workers=16, batch_size=args.batch_size, shuffle=False)
weights = torch.ones((NUM_OBJECTS)).to(device)
weights[-1] = 0.1

model = UNet(NUM_OBJECTS).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weights)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

#  ckpt = torch.load("/data/haosu/mask_2D_chpt.pth")
ckpt = torch.load("mask_2D_chpt.pth")
model.load_state_dict(ckpt["model"])

def save():
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
    }, args.save_dir / "mask_2D_chpt.pth")

for epoch in range(args.epochs):
    total_loss = 0
    accuracy = 0

    n = 0
    accum_loss = 0
    accum_acc = 0
    for i, (data, labels) in enumerate(dataset):
        data = data.to(device)
        labels = labels.to(device)
        labels = torch.clip(labels, 0, NUM_OBJECTS-1)
        output = model(data)
        loss = loss_fn(output, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        #  plt.imshow(output.max(dim=1).indices.cpu().squeeze())
        #  plt.show()
        #  print(loss)

        if i % args.backward_freq == 0 and i > 0:
            print(f'{i}: Loss: {float(accum_loss/args.backward_freq)} Accuracy: {accum_acc/args.backward_freq}')
            accum_loss = 0
            accum_acc = 0
            save()
            break
        accum_loss += float(loss)
        acc = float((output.max(dim=1).indices == labels).float().mean())
        accum_acc += acc

        accuracy += acc
        total_loss += float(loss)
        n += 1
    print(f"{epoch}: Total loss: {total_loss/n}, Accuracy: {accuracy/n}, N: {n}")
    save()

