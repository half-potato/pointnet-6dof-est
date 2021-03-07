import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("split")
parser.add_argument("output")
args = parser.parse_args()
with open(args.split, "r") as f:
    bases = [l.strip() for l in f.readlines()]

fnames = Path('/data/haosu/training_data/v2.2')
ext = '_color_kinect.png'

with open(args.output, 'w') as f:
    for base in bases:
        l = fnames / f'{base}{ext}'
        f.write(f'{l}\n')
