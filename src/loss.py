from math import pi
import torch
from pytorch3d.transforms import axis_angle_to_matrix, so3_relative_angle
import torch.nn as nn
import torch.nn.functional as F
import itertools
import functools
import time

def geodesic_distance(Rpred, Rgt, **kwargs):
    # Rpred, Rgt: (N, 3, 3)
    # out: N
    #  print(torch.bmm(Rpred, Rpred.transpose(1, 2)))
    #  print(torch.bmm(Rgt, Rgt.transpose(1, 2)))
    #  print(torch.arccos(tr - 1/2))
    #  print(tr - 1/2)
    tr = torch.einsum('bii->b', torch.bmm(Rgt, Rpred.transpose(1, 2))/2)
    return torch.arccos(torch.clamp(tr - 1/2, -1+1e-7, 1-1e-7))
    # doesn't work
    return so3_relative_angle(Rpred, Rgt)


def shape_aware_loss(Rpred, Rgt, X):
    # X: (B, 3, N)
    #  print("HI")
    #  for i in range(4):
    #      print(Rpred[i], Rgt[i])
    #      print(X[i])
    #      print(torch.bmm(Rpred, X)[i, :, :], torch.bmm(Rgt, X)[i, :, :])
    #  return torch.norm(torch.bmm(Rpred, X) - torch.bmm(Rgt, X))
    return torch.norm(torch.bmm(Rpred-Rgt, X), dim=1).mean(dim=1)


def gen_group(sym, inf_n=1):
    if "no" == sym:
        yield torch.eye(3)
        return
    # get axis
    axis = torch.zeros((3, 1))
    axis[[l in sym for l in "xyz"].index(True)] = 1
    # get angles
    if "inf" in sym:
        n = inf_n
    else:
        n = int(sym[1])
    angles = torch.linspace(0, 2*pi, n+1)[:-1]
    for ang in angles:
        yield axis_angle_to_matrix(ang*axis.T)[0]


def gen_group_products(syms):
    groups = [sym.strip() for sym in syms.split("|") if sym.strip()]
    groups = [gen_group(sym.strip()) for sym in syms.split("|") if sym.strip()]
    for group in groups:
        for item in group:
            yield item
    #  for mats in itertools.product(*groups):
    #      yield functools.reduce(torch.matmul, mats, torch.eye(3))


def symmetric_loss(loss_fn, Rpred, Rgt, symmetrices, **kwargs):
    # Rpred, Rgt: (N, 3, 3)
    # symmetrices: list of strings, N long

    sym_free = [i for i, syms in enumerate(symmetrices) if 'no' in syms]
    sym_bound = [(i, syms) for i, syms in enumerate(symmetrices) if 'no' not in syms]

    # First, handle sym free
    if "X" in kwargs:
        nkwargs = {
            "X": kwargs["X"][sym_free]
        }
    else:
        nkwargs = kwargs
    loss = loss_fn(Rpred[sym_free], Rgt[sym_free], **nkwargs).sum()

    # Second, handle sym bound shapes
    start_time = time.time()
    for j, (i, syms) in enumerate(sym_bound):
        sym_rots = torch.stack([rot for rot in gen_group_products(syms)], dim=0).to(Rgt.device)
        if "X" in kwargs:
            nkwargs = {
                "X": kwargs["X"][i][None].repeat(sym_rots.shape[0], 1, 1)
                #  "X": kwargs["X"][i][None]
            }
        else:
            nkwargs = kwargs
        # Model predicts: Rt from base reference frame to points reference frame
        # Therefore, we apply sim first at base frame, then apply rotation
        gt_rots = torch.einsum("ij,bjk->bik", Rgt[i], sym_rots)
        #  gt_rots = torch.einsum("bij,jk->bik", sym_rots, Rgt[i])
        pred_rots = Rpred[i].reshape(1, 3, 3).repeat(sym_rots.shape[0], 1, 1)
        new_loss = loss_fn(pred_rots, gt_rots, **nkwargs)
        loss += new_loss.min()
    #  print(f"Sym loss: {time.time()-start_time}")
    return loss


def margin_regularizer(a1, margin=0.3):
    return torch.max(torch.abs(torch.norm(a1, dim=1)-1)-margin, 0).values.sum()

if __name__ == "__main__":
    from pathlib import Path
    import open3d
    from mesh_sampler import MeshSampler
    base_path = Path("/data/haosu")
    meshes = MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
    meshes.load_cache()

    #  for name in meshes.iter():
    for name in ['jenga', 'master_chef_can', 'potted_meat_can', 'pudding_box', 'wood_block']:
        sample = meshes.cloud_pts[name]
        sample_pc = open3d.geometry.PointCloud()
        sample_pc.points = open3d.utility.Vector3dVector(sample)
        for rot in gen_group_products(meshes.get_symmetries(name)):
            trans = torch.eye(4)
            trans[:3, :3] = rot
            sample_pc.transform(trans)
            open3d.visualization.draw_geometries([sample_pc])
            sample_pc.transform(torch.inverse(trans))
