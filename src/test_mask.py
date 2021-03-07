import numpy as np
import loader
import frustrum_loader
from pathlib import Path
from tqdm import tqdm
import fps_cuda
import torch
from models.pointnet2_sem_seg import get_model, get_loss

import utils
import cv2
import open3d
import matplotlib.pyplot as plt
import mesh_sampler
from models.simple_mask import SimpleMask

base_path = Path('/data/haosu')
output_dir = Path('~/ScaledYOLOv4/inference/output/')
dataset = loader.PCDDataset(base_path, 'train', use_full=False)
meshes = mesh_sampler.MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
NUM_OBJECTS = len(meshes)+1 # for background
calib = [514.13902522, 514.13902522, 640, 360, 1280, 720]

device = torch.device("cuda:0")

model = SimpleMask(6, NUM_OBJECTS).to(device)
#  model = get_model(NUM_OBJECTS).to(device)
ckpt = torch.load("mask_chpt.pth")
model.load_state_dict(ckpt["model"])
model.eval()

for i in tqdm(range(len(dataset))):
    # Get frustrums
    raw = dataset.load_raw(i)
    rgb, depth, label, meta = raw
    H, W = rgb.shape[:2]
    #  bbxs = dataset.load_bbxs(i, output_dir)
    bbxs = dataset.get_bbxs(label, meta, margin=5)

    frustrums = frustrum_loader.extract_frustrums(raw, bbxs)
    for (pc, lab), bbx in zip(frustrums, bbxs):
        _, cx, cy, w, h = bbx
        points = torch.tensor(pc[:, 1:].T).float()
        points = frustrum_loader.center_frustrum(points, bbx, calib).unsqueeze(0).to(device)
        #  utils.draw_augpoints(points.cpu().numpy(), False, True)
        pred = model(points)
        pred = pred.transpose(1, 2)
        print(pred[0].shape)
        print(f'Full Accuracy: {(pred.max(dim=1).indices.cpu().numpy() == pc[:, 0:1].T).mean()}')
        t = int(max(cy-h/2, 0) * H)
        b = int(min(cy+h/2, 1) * H)
        l = int(max(cx-w/2, 0) * W)
        r = int(min(cx+w/2, 1) * W)
        #  mask = (pred.max(dim=1).indices == lab).reshape(b-t, r-l)
        accuracy = ((pred.max(dim=1).indices == lab).cpu().numpy() == (pc[:, 0:1].T == lab)).mean()
        print(f'Accuracy: {accuracy}')
        mask = (pred.max(dim=1).indices).reshape(b-t, r-l)
        plt.imshow(mask.cpu())
        plt.title(f'True label: {lab}')
        plt.show()

