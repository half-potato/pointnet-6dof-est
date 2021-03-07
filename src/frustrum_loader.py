import os
import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.data.dataset import Dataset, IterableDataset
from PIL import Image
from pyquaternion import Quaternion
import math
import h5py

def create_loaders(cache_path, calib):
    data = h5py.File(cache_path, 'r')
    point_number_groups = data['point_nums']
    loaders = []
    for i in range(len(point_number_groups)):
        loaders.append(FrustrumLoader(data[f'pcs{i}'], data[f'labs{i}'], data[f'bbxs{i}'], point_number_groups[i], calib))
    return loaders, point_number_groups

class FrustrumLoader(Dataset):
    def __init__(self, pcs, labs, bbxs, n, calib):
        self.pcs = pcs
        self.labs = labs
        self.bbxs = bbxs
        self.n = n
        self.calib = calib

    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, i):
        points = torch.tensor(self.pcs[i])[:, 1:].transpose(0, 1).float()
        labels = torch.tensor(self.pcs[i])[:, 0].long()
        labels = torch.clip(labels, 0, 79)
        return center_frustrum(points, self.bbxs[i], self.calib), labels, self.labs[i].astype(np.long)


def center_frustrum(points, bbx, calib, labeled=False):
    centers = points.mean(dim=1, keepdim=True)
    D = points.shape[0]
    fx, fy, cx, cy, W, H = calib
    _, bcx, bcy, w, h = bbx
    # angular position of bbx
    ax = np.arctan2((bcy*H-cy)/fy, 1) + 3*math.pi/4
    ay = np.arctan2((bcx*W-cx)/fx, 1)

    q = Quaternion(axis=[1., 0., 0.], radians=ax) * Quaternion(axis=[0., 1., 0.], radians=ay)
    T = torch.eye(D)
    si = 1 if labeled else 0
    T[si:si+3, si:si+3] = torch.tensor(q.rotation_matrix.T)
    rotp = T @ points
    ncenters = rotp[si:].median(dim=1, keepdim=True).values
    rotp[si:] -= ncenters
    return rotp

def extract_frustrums(raw, bbxs):
    rgb, depth, label, meta = raw
    H, W = rgb.shape[:2]
    # Points
    v, u = np.indices(depth.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(depth)], axis=-1)
    points = uv1 @ np.linalg.inv(meta["intrinsic"]).T * depth[..., None] # [H, W, 3]

    if label is None:
        label = np.zeros((H, W, 1))
    augpoints = np.concatenate([label.reshape(H, W, 1), points, rgb], axis=2)
    D = augpoints.shape[-1]
    frustrums = []
    for bbx in bbxs:
        # First, make sure bbxs are not out of bounds
        lab, cx, cy, w, h = bbx
        t = int(max(cy-h/2, 0) * H)
        b = int(min(cy+h/2, 1) * H)
        l = int(max(cx-w/2, 0) * W)
        r = int(min(cx+w/2, 1) * W)
        frustrums.append((augpoints[t:b, l:r].reshape(-1, D), lab))
    return frustrums
