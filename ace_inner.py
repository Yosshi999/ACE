import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
import h5py

from ace.ace import ConceptDiscovery

parser = argparse.ArgumentParser()
parser.add_argument('cav_dir')
parser.add_argument('working_dir', help='input grads and output inners')
parser.add_argument('--category', default="person", help='category name for the collected concepts')
args = parser.parse_args()

input_dir = os.path.join(args.working_dir, 'data4_grad')
output_dir = os.path.join(args.working_dir, 'data5_inner')
# Make a destination directory. If it already exists, raise an error.
os.mkdir(output_dir)

print("load cavs")
cavs = []
for fn in tqdm(sorted(Path(args.cav_dir).iterdir())):
    c, r, bn, *_ = fn.name.split('-')
    if c.startswith('{}_concept'.format(args.category)):
        cav = ConceptDiscovery.load_cav_direction(args, c, r, bn)
        cav = cav.squeeze().astype(np.float32)
        cavs.append((fn.stem, cav))
print("cav shape:", cavs[0][1].shape)

print("calc grad cav")
f = h5py.File(os.path.join(output_dir, "inner.hdf5"), "w")
for fn in tqdm(sorted(Path(input_dir).iterdir())):
    grad_of_image = np.load(os.path.join(input_dir, fn.name)) # (dets, grad vector)
    grad_of_image = grad_of_image / np.linalg.norm(grad_of_image, axis=-1, keepdims=True)
    g = f.create_group('/' + fn.stem)
    for cav_name, cav in cavs:
        inner = np.dot(grad_of_image, cav)
        g.create_dataset(name=cav_name, data=inner)
f.close()
