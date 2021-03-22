import numpy as np
import loader
import frustrum_loader
from pathlib import Path
from tqdm import tqdm
#  import fps_cuda
import torch

import utils
import cv2
import open3d
import matplotlib.pyplot as plt
import h5py
import argparse
from cacher import Cacher
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--use_full', action='store_true')
parser.add_argument('--cache_location', default=Path('/data/haosu/object_cache.hdf5'), type=Path)
args = parser.parse_args()

base_path = Path('/data/haosu')
dataset = loader.PCDDataset(base_path, 'train', use_full=args.use_full)
#  point_number_groups = [50, 500, 1000, 5000, 7000, 9000, 11000, 15000, 20000, 50000, 100000]
point_number_groups = [1024, 3000, 5000, 7000, 9000, 11000, 15000, 20000, 50000, 100000]
calib = [514.13902522, 514.13902522, 640, 360, 1280, 720]
print("Caching objects")

cacher = Cacher(args.cache_location, point_number_groups, 6, [('rgts', (3, 3)), ('tgts', (3, 1)), ('idxs', [1])])

for i in tqdm(range(0, len(dataset))):
    # Get frustrums
    objects, prefix = dataset[i]

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

    # Sample frustrums
    for obj in objects:
        name = obj["object_name"]
        #  s = obj["scale"].reshape(1, 3)
        idx = obj["object_id"]
        Rgt = torch.tensor(obj["pose"][:3, :3])
        tgt = torch.tensor(obj["pose"][:3, 3:4])

        pc = obj['points']
        cols = obj['colors']
        pc = np.concatenate([pc, cols], axis=1)

        # Get point number group
        M, D = pc.shape
        G = [i for i, p in enumerate(point_number_groups) if M <= p]
        if len(G) == 0:
            continue
        G = G[0]
        N = point_number_groups[G]

        inds = np.random.choice(M, N-M, replace=True)
        pc = np.concatenate([pc, pc[inds]], axis=0)
        """
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
            cacher.add_to_group(0, {'pcs': pc, 'rgts': Rgt, 'tgts': tgt, 'idxs': idx})
            continue
        N = Ns[-1]

        pc_t = torch.tensor(pc)[:, :3].unsqueeze(0).cuda()
        pc_i = fps_cuda.farthest_point_sample(pc_t, N).squeeze(0).cpu().numpy()
        pc = pc[pc_i]
        """

        # Display cloud
        #  utils.draw_augpoints([pc], True)

        # Add frustrums to each category
        #  G = len(Ns)-1
        cacher.add_to_group(G, {'pcs': pc, 'rgts': Rgt, 'tgts': tgt, 'idxs': idx})

cacher.print_sizes()
cacher.close()

