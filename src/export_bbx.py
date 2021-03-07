import cv2
import loader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_full', action='store_true')
args = parser.parse_args()
base_path = Path('/data/haosu')

def export(prefix):
    dataset = loader.PCDDataset(base_path, prefix, use_full=args.use_full)
    for i in tqdm(range(len(dataset))):
        rgb, depth, label, meta = dataset.load_raw(i)
        prefix = dataset.prefix[i]
        bbxs = dataset.get_bbxs(label, meta, margin=5)
        H, W = label.shape[:2]
        with open(dataset.data_dir / f'{prefix}_color_kinect.txt', 'w') as f:
            for bbx in bbxs:
                _, cx, cy, w, h = bbx
                #  t = int(max(cy-h/2, 0) * H)
                #  b = int(min(cy+h/2, 1) * H)
                #  l = int(max(cx-w/2, 0) * W)
                #  r = int(min(cx+w/2, 1) * W)
                #  cv2.rectangle(label, (l, t), (r, b), (0,255,0),3)
                f.write(' '.join([str(s) for s in bbx]) + '\n')
            #  plt.imshow(label)
            #  plt.show()

export('train')
export('val')
