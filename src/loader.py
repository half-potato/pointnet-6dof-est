import os
import open3d
import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.data.dataset import Dataset, IterableDataset
from PIL import Image
#  import fps_cuda

import matplotlib.pyplot as plt

NUM_OBJECTS = 79

def load_pickle(filename):
    # keys:  'extents', 'scales', 'object_ids', 'object_names', 'extrinsic', 'intrinsic'
    with open(filename, 'rb') as f:
        return pickle.load(f)

def to_geometry(obj):
    # obj: dict
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(obj["points"].reshape(-1, 3))
    pcd.colors = open3d.utility.Vector3dVector(obj["colors"].reshape(-1, 3))

    extent = obj["box"] #/obj["scale"]
    bbx = open3d.geometry.TriangleMesh.create_box(width=extent[0], height=extent[1], depth=extent[2])
    centering = np.eye(4)
    centering[:3, 3] = -extent/2
    bbx.transform(centering)
    bbx.transform(obj["pose"])

    return pcd, bbx

def get_training_set_items(base_path, prefix):
    path = base_path / 'training_data' / 'v2.2' / f'{prefix}_meta.pkl'
    meta = load_pickle(path)
    return meta['object_ids']


class PCDDataset(Dataset):
    def __init__(self, base_path, split_name, use_full=True):
        self.use_full = use_full
        self.base_path = base_path
        if split_name in ["val", "train"]:
            self.data_dir = self.base_path / "training_data" / "v2.2"
        elif split_name == 'test':
            self.data_dir = self.base_path / "testing_data" / "v2.2"
        elif split_name == 'test_perception':
            self.data_dir = self.base_path / "testing_data_perception" / "v2.2"
        self.rgb_files, self.depth_files, self.label_files, self.meta_files, self.prefix = self.get_split_files(split_name)

    def get_split_files(self, split_name):
        if split_name == "test" or split_name == 'test_perception':
            names = [fname.split("_")[0] for fname in os.listdir(self.data_dir) if "color" in fname]
        else:
            with open(self.base_path / "training_data" / "splits" / "v2" / f"{split_name}.txt", 'r') as f:
                names = [line.strip() for line in f if line.strip()]

        prefix = [os.path.join(self.data_dir, name) for name in names]
        rgb = [p + "_color_kinect.png" for p in prefix]
        depth = [p + "_depth_kinect.png" for p in prefix]
        label = [p + "_label_kinect.png" for p in prefix]
        meta = [p + "_meta.pkl" for p in prefix]
        return rgb, depth, label, meta, names

    def load_bbxs(self, i, output_dir):
        prefix = self.prefix[i]
        with (output_dir / f'{prefix}_color_kinect.txt').open() as f:
            return [[float(v) for v in l.strip().split(' ')] for l in f if l.strip()]

    def load_raw(self, i):
        rgb = np.array(Image.open(self.rgb_files[i])) / 255
        depth = np.array(Image.open(self.depth_files[i])) / 1000 # to meters
        if os.path.exists(self.label_files[i]):
            label = np.array(Image.open(self.label_files[i]))
            meta = load_pickle(self.meta_files[i])
        else:
            label = None
            meta = None
        return rgb, depth, label, meta

    def get_bbxs(self, label, meta, margin=5):
        H, W = label.shape[:2]
        bbxs = []
        for i, idx in enumerate(meta["object_ids"]):
            r, c = np.where(label == idx)
            if len(r) < 6:
                continue
            t = max((np.min(r)-margin) / H, 0)
            b = min((np.max(r)+margin) / H, 1)
            l = max((np.min(c)-margin) / W, 0)
            r = min((np.max(c)+margin) / W, 1)
            h = b-t
            w = r-l
            cy = (t+b)/2
            cx = (r+l)/2
            bbxs.append([idx, cx, cy, w, h])
        return bbxs

    def __len__(self):
        return len(self.rgb_files) if self.use_full else 10000

    def __getitem__(self, i):
        prefix = self.prefix[i]
        rgb, depth, label, meta = self.load_raw(i)
        v, u = np.indices(depth.shape)
        uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(depth)], axis=-1)
        #  points = uv1 @ np.linalg.inv(meta["intrinsic"]).T * depth[..., None] # [H, W, 3]
        points = uv1 @ np.linalg.inv(meta["intrinsic"]).T * depth[..., None] # [H, W, 3]
        # augment points and transform
        H, W, _ = points.shape
        points = (np.concatenate([points, np.ones((H, W, 1))], axis=2) @ np.linalg.inv(meta["extrinsic"]).T)[:, :, :3]
        # next, we split the points up according to their labels
        objects = []
        for i, idx in enumerate(meta["object_ids"]):
            s = meta['scales'][idx]
            mask = np.where(label == idx)
            if (label==idx).sum() == 0:
                continue
            name = meta['object_names'][i]
            obj_points = points[mask[0], mask[1], :]
            colors = rgb[mask[0], mask[1], :]
            data = {
                "object_id": idx,
                "object_name": name,
                "colors": colors,
                "points": obj_points,
                "scale": s,
                "box": meta['extents'][idx]*meta['scales'][idx]
            }
            if "poses_world" in meta:
                data["pose"] = meta["poses_world"][idx]
            objects.append(data)
        return objects, prefix

class MaskLoader(Dataset):
    def __init__(self, base_path, split_name):
        self.pcddata = PCDDataset(base_path, split_name)

    def __getitem__(self, i):
        rgb, depth, label, meta = self.pcddata.load_raw(i)
        return (np.rollaxis(rgb, 2, 0)/255.).astype(np.float32), label.astype(np.long)

    def __len__(self):
        return len(self.pcddata)

class TrainLoader(Dataset):
    def __init__(self, base_path, split_name, point_number_groups):
        self.pcddata = PCDDataset(base_path, split_name)
        self.point_number_groups = point_number_groups
        self.num_g = 7

    def resample(self, points):
        M = points.shape[0]
        Ns = [p for p in self.point_number_groups if M >= p]
        if not Ns:
            return -1, None
        N = Ns[-1]
        if points.shape[0] > N*2:
        #      # Done for speed
            inds = np.random.choice(np.arange(0, points.shape[0]), N*2)
            points = points[inds]

        #  inds = np.random.choice(np.arange(0, points.shape[0]), N)
        pc_t = torch.tensor(points)[:, :3].unsqueeze(0).cuda()
        inds = fps_cuda.farthest_point_sample(pc_t, N).squeeze(0).cpu().numpy()
        points = points[inds]
        return len(Ns)-1, points

    def __len__(self):
        return len(self.pcddata)

    def get_prefix(self, i):
        return self.pcddata.prefix[i]

    def __getitem__(self, i):
        grouped_points = [[[] for _ in range(self.num_g)] for _ in self.point_number_groups]
        objects, prefix = self.pcddata[i]
        geoms = []
        for obj in objects:
            name = obj["object_name"]
            idx = obj["object_id"]
            s = obj["scale"].reshape(1, 3)

            group_num, samp = self.resample(obj['points'])
            if group_num < 0:
                #  print(f"Skipped, only {obj['points'].shape[0]} points")
                continue
            #  obj['points'] = samp
            #  geoms.extend(to_geometry(obj))

            samp = torch.tensor(samp).float()
            #  samp = torch.tensor(obj['points'][samp] / s).float()
            if "pose" in obj:
                Rgt = torch.tensor(obj["pose"][:3, :3])
                tgt = torch.tensor(obj["pose"][:3, 3])
            else:
                Rgt = torch.zeros((3, 3))
                tgt = torch.zeros((3, 1))
            group = grouped_points[group_num]
            group[0].append(samp.T)
            group[1].append(Rgt)
            group[2].append(tgt)
            group[3].append(torch.tensor(idx))
            group[4].append(name)
            group[5].append(obj['scale'])
            group[6].append(self.get_prefix(i))
        #  open3d.visualization.draw_geometries(geoms)

        if len(self.point_number_groups) == 1:
            return [torch.stack(g, 0) if type(g[0]) == torch.Tensor else g for g in grouped_points[0]]
        else:
            out = []
            for gp in grouped_points:
                if len(gp[0]) == 0:
                    out.extend(gp)
                else:
                    out.extend([torch.stack(g, 0) if type(g[0]) == torch.Tensor else g for g in gp])
            return out

class BatchAccumulator(IterableDataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch = []
        self.i = 0

    def __len__(self):
        return sum([a[0].shape[0] for a in self.batch])

    def fill(self):
        while len(self) < self.batch_size:
            self.batch.append(self.dataset[self.i])
            self.i += 1
            if len(self.dataset) <= self.i:
                return False
        return True

    def unload(self):
        batch = []
        for i in range(len(self.batch[0])):
            elements = [b[i] for b in self.batch]
            if type(self.batch[0][i]) == torch.Tensor:
                element = torch.cat(elements, dim=0)
            else:
                element = sum(elements, [])
            batch.append(element)
        self.batch = []
        return batch

    def __next__(self):
        if not self.fill():
            self.i = 0
            self.batch = []
            raise StopIteration
        return self.unload()

    def __iter__(self):
        return self

def collate_fn(batch):
    out = []
    num_g = 6
    for i in range(len(batch[0])):
        elements = [b[i] for b in batch if b[i] != []]
        if len(elements) == 0:
            continue
        if type(elements[0]) == torch.Tensor:
            element = torch.cat(elements, dim=0)
        else:
            element = sum(elements, [])
        out.append(element)
    return out

