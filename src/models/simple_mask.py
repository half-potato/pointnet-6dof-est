from models.pointnet import PointNetfeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SimpleMask(nn.Module):
    def __init__(self, k, num_objects):
        super(SimpleMask, self).__init__()
        # M = num_objects
        self.num_objects = num_objects
        self.net = PointNetfeat(k, enable_transform=False, global_feat=False, feature_transform=True)
        self.seq = nn.Sequential(
            torch.nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            torch.nn.ReLU(True),
            torch.nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv1d(256, self.num_objects, 1),
            torch.nn.Softmax(1)
        )
        self.k = k

    def center(self, points):
        return points

    def forward(self, points):
        # points: (B, N, k)
        # feat: (B, 1024)
        feat, trans, _ = self.net(points)
        feat = self.seq(feat)
        feat = feat.transpose(1, 2)
        return feat

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    model = SimpleMask(79)
    mask = model(sim_data)
    print(mask.shape)
    print(mask)
