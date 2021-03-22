from models.pointnet import PointNetfeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def rotation_from_vectors(a1, a2, num_objects):
    b1 = a1 / (torch.norm(a1, dim=1, keepdim=True)+1e-8)
    uv = torch.bmm(a2.reshape(-1, 1, 3), a1.reshape(-1, 3, 1)).reshape(-1, 1)
    uu = torch.bmm(a1.reshape(-1, 1, 3), a1.reshape(-1, 3, 1)).reshape(-1, 1)
    b2 = a2 - uv/uu * a1

    b2 = b2 / (torch.norm(b2, dim=1, keepdim=True)+1e-8)
    # cross product
    a3 = torch.cross(b1, b2, dim=1)
    b3 = a3 / (torch.norm(a3, dim=1, keepdim=True)+1e-8)

    # assemble rot matrix
    rots = torch.stack([b1, b2, b3], dim=2).reshape(-1, num_objects, 3, 3)
    return rots

class Simple6DOF(nn.Module):
    def __init__(self, num_objects):
        super(Simple6DOF, self).__init__()
        # M = num_objects
        self.num_objects = num_objects
        self.seq = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_objects*9),
        )
        self.net = PointNetfeat(global_feat=True, enable_transform=True, feature_transform=False)
        #  self.net = PointNetfeat(k=6, global_feat=True, enable_transform=False, feature_transform=False)

    def forward(self, points):
        # points: (B, N, 3)
        # feat: (B, 1024)
        feat, trans, _ = self.net(points)
        # (B, M*6)
        raw_out = self.seq(feat)
        vecs = raw_out.reshape(-1, 3, 3)
        # gram-schmidt process
        # (B*M, 3)
        a1 = vecs[:, 0]
        a2 = vecs[:, 1]
        t = vecs[:, 2]
        return a1, a2, t.reshape(-1, self.num_objects, 3)

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    model = Simple6DOF(79)
    rots, ts = model(sim_data)
    print(rots.shape, ts.shape)
    print(rots[0, 0], ts[0, 0])
    print(torch.norm(rots[0, 0], dim=1))
    print(rots[0, 0] @ rots[0, 0].T)
