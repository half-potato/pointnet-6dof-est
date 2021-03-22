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
parser.add_argument('--checkpoint', type=Path, default=Path("mask_chpt.pth"))

args = parser.parse_args()

base_path = Path('/data/haosu')

#  output_dir = Path('~/ScaledYOLOv4/inference/output/')
if args.use_gt_bbx:
    dataset = loader.PCDDataset(base_path, 'train', use_full=False)
else:
    #  dataset = loader.PCDDataset(base_path, 'test_perception', use_full=True)
    dataset = loader.PCDDataset(base_path, 'test_final', use_full=True)
output_dir = dataset.data_dir

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

#  model = SimpleMask(6, NUM_OBJECTS).to(device)
#  model = get_model(NUM_OBJECTS).to(device)
#  ckpt = torch.load("mask_chpt.pth")

#  loss_fn = get_loss()
model = get_model(6, NUM_OBJECTS).to(device)

point_number_groups = [1024, 3000, 5000, 7000, 9000, 11000, 15000, 20000, 50000]
ckpt = torch.load(args.checkpoint)
model.load_state_dict(ckpt["model"])
#  model.eval()

print(f"Saving to {args.output_dir}")
with torch.no_grad():
    for i in tqdm(range(220, len(dataset))):
        raw = dataset.load_raw(i)
        rgb, depth, label, meta = raw
        H, W = rgb.shape[:2]

        # Get box
        if args.use_gt_bbx:
            bbxs = dataset.get_bbxs(label, meta, margin=margin)
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

            M, D = pc.shape
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

            G = [i for i, p in enumerate(point_number_groups) if M <= p]
            if not G:
                continue
            G = G[0]
            N = point_number_groups[G]

            inds = np.random.choice(M, N-M, replace=True)
            pc = np.concatenate([pc, pc[inds]], axis=0)

            points = torch.tensor(pc[:, 1:].T).float()
            if points.shape[1] < 32:
                continue
            points = frustrum_loader.center_frustrum(points, bbx, calib).unsqueeze(0).to(device)
            #  utils.draw_augpoints(points.cpu().numpy(), False, True)

            # Run and predict
            pred = model(points)
            pred = pred[:, :M]
            #  pred = pred.transpose(1, 2)
            allowed = np.array([int(lab), 79])

            # Draw label
            """
            pred_lab = pred.max(dim=-1).indices.cpu().numpy()
            pred2 = pred_lab.reshape(points.shape[0], M)
            disp = points.cpu().numpy()[0, :, :M]
            print(pred2.shape, disp.shape)
            disp[3, :] = pred2/NUM_OBJECTS
            disp[4, :] = pred2/NUM_OBJECTS
            disp[5, :] = pred2/NUM_OBJECTS
            utils.draw_augpoints([disp], False, True)
            """
            #  continue

            # reshape and append
            inds = pred[0][:, allowed].max(dim=-1).indices.cpu().numpy()
            pred_allowed = allowed[inds]
            pred_lab = pred_allowed.reshape(b-t, r-l)
            #  pred_lab = pred.max(dim=-1).indices.cpu().numpy().reshape(b-t, r-l)
            mask = np.where(pred_lab == lab, lab, BG_ID)
            #  masks.append(mask)
            masks.append(pred_lab)
            bounds.append([lab, b, t, r, l])

            # Debug Out
            if args.use_gt_bbx:
                gt_mask = np.where(pc[:M, 0:1].T == lab, lab, BG_ID).reshape(b-t, r-l)
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
        for (lab, b, t, r, l), mask in zip(bounds, masks):
            #  inds = np.where((mask != BG_ID) | (composite[t:b, l:r] == BG_ID))
            assign_ifnot = ((mask != lab) & (composite[t:b, l:r] == BG_ID))
            assign_if = ((mask == lab) & (composite[t:b, l:r] != lab))
            inds = np.where((mask != BG_ID & (assign_if | assign_ifnot)))
            #  inds = np.where((composite[t:b, l:r] == BG_ID))
            composite[t:b, l:r][inds] = mask[inds]

        # Save
        if not args.use_gt_bbx:
            path = args.output_dir / f'{prefix}_label_kinect.png'
            print(path)
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

