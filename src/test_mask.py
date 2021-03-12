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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_gt_bbx', action='store_true')
parser.add_argument('--output_dir', type=Path, default=Path('/data/output/haosu/'))
parser.add_argument('--display_individual', action='store_true')
parser.add_argument('--display', action='store_true')
args = parser.parse_args()

base_path = Path('/data/haosu')
output_dir = Path('/data/haosu/testing_data_perception/v2.2/')

#  output_dir = Path('~/ScaledYOLOv4/inference/output/')
if args.use_gt_bbx:
    dataset = loader.PCDDataset(base_path, 'train', use_full=False)
else:
    dataset = loader.PCDDataset(base_path, 'test_perception', use_full=True)

meshes = mesh_sampler.MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
BG_ID = len(meshes)
NUM_OBJECTS = len(meshes)+1 # for background
calib = [514.13902522, 514.13902522, 640, 360, 1280, 720]
margin = 0.00

intrinsic = np.eye(3)
intrinsic[0, 0] = calib[0]
intrinsic[1, 1] = calib[1]
intrinsic[0, 2] = calib[2]
intrinsic[1, 2] = calib[3]

device = torch.device("cuda:0")

model = SimpleMask(6, NUM_OBJECTS).to(device)
#  model = get_model(NUM_OBJECTS).to(device)
#  ckpt = torch.load("mask_chpt.pth")
ckpt = torch.load("mask_chpt.pth")
model.load_state_dict(ckpt["model"])
model.eval()

print(f"Saving to {args.output_dir}")
for i in tqdm(range(len(dataset))):
    raw = dataset.load_raw(i)
    rgb, depth, label, meta = raw
    H, W = rgb.shape[:2]

    # Get box
    if args.use_gt_bbx:
        bbxs = dataset.get_bbxs(label, meta, margin=5)
    else:
        bbxs = dataset.load_bbxs(i, output_dir)

    frustrums = frustrum_loader.extract_frustrums(raw, bbxs, intrinsic, margin)

    # Get labels
    prefix = dataset.prefix[i]
    allowed = loader.get_training_set_items(base_path, prefix)
    allowed = np.concatenate((allowed, [79]), axis=0)

    masks = []
    bounds = []
    for (pc, lab), bbx in zip(frustrums, bbxs):
        # Load
        _, cx, cy, w, h = bbx
        t = int(max(cy-h/2-margin, 0) * H)
        b = int(min(cy+h/2+margin, 1) * H)
        l = int(max(cx-w/2-margin, 0) * W)
        r = int(min(cx+w/2+margin, 1) * W)
        points = torch.tensor(pc[:, 1:].T).float()
        points = frustrum_loader.center_frustrum(points, bbx, calib).unsqueeze(0).to(device)
        #  utils.draw_augpoints(points.cpu().numpy(), False, True)

        # Run and predict
        pred = model(points)
        pred = pred.transpose(1, 2)
        allowed = np.array([int(lab), 79])

        # reshape and append
        inds = pred[0][allowed].max(dim=0).indices.cpu().numpy()
        pred_lab = allowed[inds].reshape(b-t, r-l)
        #  pred_lab = pred.max(dim=1).indices.cpu().numpy().reshape(b-t, r-l)
        mask = np.where(pred_lab == lab, lab, BG_ID)
        masks.append(mask)
        bounds.append([b, t, r, l])

        # Debug Out
        if args.use_gt_bbx:
            gt_mask = np.where(pc[:, 0:1].T == lab, lab, BG_ID).reshape(b-t, r-l)
            accuracy = (mask == gt_mask).mean()
            print(f'Accuracy: {accuracy}')
        if args.display_individual:
            figs, axs = plt.subplots(1, 2, dpi=400)
            axs[0].imshow(pred_lab)
            axs[0].label = f'Pred: {lab}'
            if not args.use_gt_bbx:
                secondary = rgb[t:b, l:r]
            else:
                secondary = np.clip(label[t:b, l:r], 0, BG_ID)
            axs[1].imshow(secondary)
            axs[1].label = f'Secondary: {lab}'
            plt.show()


    # Composite new labels
    composite = np.ones((H, W), dtype=np.uint8)*BG_ID
    for (b, t, r, l), mask in zip(bounds, masks):
        inds = np.where(mask != BG_ID)
        composite[t:b, l:r][inds] = mask[inds]

    # Save
    path = args.output_dir / f'{prefix}_label_kinect.png'
    cv2.imwrite(str(path), composite)

    # Display
    if args.display:
        figs, axs = plt.subplots(1, 2, dpi=400)
        axs[0].imshow(composite)
        axs[0].label = 'Pred'
        if not args.use_gt_bbx:
            secondary = rgb
        else:
            secondary = np.clip(label, 0, BG_ID)
        axs[1].imshow(secondary)
        axs[1].label = 'Gt'
        plt.show()

