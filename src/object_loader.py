import torch
from torch.utils.data.dataset import Dataset

class ObjectLoader(Dataset):
    def __init__(self, group):
        self.group = group
        self.N = group['pcs'].shape[0] # number of elements in group
        self.n = group['pcs'].shape[1] # number of points per element
        self.k = 1

    def __len__(self):
        return self.N // self.k

    def __getitem__(self, i):
        if i >= len(self):
            raise StopIteration
        i = i // self.k
        cloud = self.group['pcs'][i]
        Rgt = self.group['rgts'][i]
        tgt = self.group['tgts'][i]
        idx = self.group['idxs'][i]

        points = torch.tensor(cloud).transpose(0, 1).float()
        std = 0.001
        noise = torch.rand(*points.shape)*std*2-std
        points += noise

        return points, Rgt, tgt, idx
