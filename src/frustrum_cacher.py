import numpy as np
import loader
import frustrum_loader
from pathlib import Path
from tqdm import tqdm
import fps_cuda
import torch

import utils
import cv2
import open3d
import matplotlib.pyplot as plt
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_full', action='store_true')
args = parser.parse_args()

base_path = Path('/data/haosu')
dataset = loader.PCDDataset(base_path, 'train', use_full=args.use_full)
#  point_number_groups = [50, 500, 1000, 5000, 7000, 9000, 11000, 15000, 20000, 50000, 100000]
point_number_groups = [1024, 3000, 5000, 7000, 9000, 11000, 15000, 20000, 50000, 100000]
calib = [514.13902522, 514.13902522, 640, 360, 1280, 720]
print("Caching frustrums")

f = h5py.File('caches/frustrum_cache.hdf5', 'w')
f.create_dataset('point_nums', data=point_number_groups)

pgs = [f.create_group(f'point_group{i}-{j}') for i, j in enumerate(point_number_groups)]
for pg, n in zip(pgs, point_number_groups):
    pg.create_dataset('bbxs', (0, 5), maxshape=(None, 5), chunks=True)
    pg.create_dataset('pcs', (0, n, 7), maxshape=(None, n, 7), chunks=True)

counts = [0 for _ in point_number_groups]
def add_to_group(g, bbx, pc):
    N = pgs[g]['bbxs'].shape[0]
    pgs[g]['bbxs'].resize(N+1, axis=0)
    pgs[g]['bbxs'][N] = bbx
    pgs[g]['pcs'].resize(N+1, axis=0)
    pgs[g]['pcs'][N] = pc

    counts[g] += 1

for i in tqdm(range(0, len(dataset), 5)):
    # Get frustrums
    raw = dataset.load_raw(i)
    rgb, depth, label, meta = raw
    fbbxs = dataset.get_bbxs(label, meta, margin=5)

    #  H, W = label.shape[:2]
    #  for bbx in bbxs:
    #      _, cx, cy, w, h = bbx
    #      t = int(max(cy-h/2, 0) * H)
    #      b = int(min(cy+h/2, 1) * H)
    #      l = int(max(cx-w/2, 0) * W)
    #      r = int(min(cx+w/2, 1) * W)
    #      cv2.rectangle(label, (l, t), (r, b), (0,255,0),3)
    #  plt.imshow(label)
    #  plt.show()

    frustrums = frustrum_loader.extract_frustrums(raw, fbbxs)
    # Sample frustrums
    for (pc, lab), bbx in zip(frustrums, fbbxs):
        # Get point number group
        M, D = pc.shape
        Ns = [p for p in point_number_groups if M >= p]
        if not Ns:
            # pad up to minimum
            N = point_number_groups[0]
            #  pad = np.zeros((N-M, D))
            #  pad[:, 0] = 79
            inds = np.random.choice(M, N, replace=True)
            pc = pc[inds]
            #  pc = np.concatenate([pc, pad], axis=0)
            #  g = pgs[0].create_group(f'{counts[0]}')
            #  g.create_dataset(f'lab', [lab])
            #  g.create_dataset(f'bbx', data=bbx)
            #  g.create_dataset(f'pc', data=pc)
            add_to_group(0, bbx, pc)
            continue
        N = Ns[-1]

        pc_t = torch.tensor(pc)[:, :3].unsqueeze(0).cuda()
        pc_i = fps_cuda.farthest_point_sample(pc_t, N).squeeze(0).cpu().numpy()
        pc = pc[pc_i]

        # Display cloud
        #  utils.draw_augpoints([pc], True)
        """
        print(pc.shape)
        npc = frustrum_loader.center_frustrum(torch.tensor(pc.T).float(), bbx, calib).numpy().T
        print(pc.shape)

        points = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                  [0,0,1],[1,0,1],[0,1,1],[1,1,1]])
        lines = [[0,1],[0,2],[1,3],[2,3],
                 [4,5],[4,6],[5,7],[6,7],
                 [0,4],[1,5],[2,6],[3,7]]
        colors = [[0, 0, 0] for i in range(len(lines))]
        colors[0] = [0, 1, 0]
        colors[1] = [0, 0, 1]
        colors[8] = [1, 0, 0]
        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(points)
        line_set.lines = open3d.utility.Vector2iVector(lines)
        line_set.colors = open3d.utility.Vector3dVector(colors)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pc[:, 1:4])
        pcd.colors = open3d.utility.Vector3dVector(pc[:, 4:7])

        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(npc[:, 1:4])
        pcd2.colors = open3d.utility.Vector3dVector(npc[:, 4:7])

        open3d.visualization.draw_geometries([pcd2, line_set])
        """

        # Add frustrums to each category
        G = len(Ns)-1
        add_to_group(G, bbx, pc)

print("Sizes of point groups")
for num_pt, n in zip(point_number_groups, counts):
    print(f'{num_pt}: {n}')
f.create_dataset('counts', data=counts)
f.close()
