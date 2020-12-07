import argparse
import json
import os
from pathlib import Path

import PIL.Image
import numpy as np
from tqdm import tqdm

import ace.config
from ace import ace_helpers

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bottleneck', help='override bottleneck')
parser.add_argument('config', help='Only model, target_class and bottlenecks are used. The number of bottlenecks must be 1. model.fix_ratio is forced to False.')
parser.add_argument('det', help='~/git/detectron2_v/experiments/002/inference/bdd_val/coco_instances_results_bdd.json')
parser.add_argument('image_dir', help='~/git/CenterNet/data/bdd/bdd100k/images/100k/val')
parser.add_argument('working_dir', help='output acts')
args = parser.parse_args()

# model
config = ace.config.load(args.config)
config.model.fix_ratio = False
if args.bottleneck is not None:
    config.bottlenecks[:] = [args.bottleneck]
assert len(config.bottlenecks) == 1
model = ace_helpers.make_model(config.model)

with open(args.det) as f:
    dets = json.load(f)

assert ('name' in dets[0]) and ('category' in dets[0]) and "det results must have name&category. See git/detectron2/tools/append_mapped_dets.py"

output_dir = os.path.join(args.working_dir, 'data4_grad')
# Make a destination directory. If it already exists, raise an error.
os.mkdir(output_dir)

names = []
boxeses = []
name = dets[0]['name']
boxes = []
n_dropped = 0
for det in dets:
    assert name <= det['name']
    if name < det['name']:
        if boxes:
            names.append(name)
            boxeses.append(boxes)
        else:
            n_dropped += 1
        name = det['name']
        boxes = []
    if det['category'] == config.target_class:
        boxes.append(det['bbox'])
names.append(name)
boxeses.append(boxes)
assert len(names) == len(boxeses)
print('n_image: {:5}'.format(len(names) + n_dropped))
print('  det>0: {:5}'.format(len(names)))
print('  det=0: {:5}'.format(n_dropped))
del dets

n_cpu = os.cpu_count()
print('n_cpu:', n_cpu, flush=True)

def load_img(i):
    return i, np.array(PIL.Image.open(os.path.join(args.image_dir, names[i])))

class_id = model.label_to_id(config.target_class.replace('_', ' '))
for i in tqdm(range(len(names))):
    i, img = load_img(i)
    act = model.run_imgs([img], config.bottlenecks[0], boxeses[i])
    n_det = len(boxeses[i])
    grad = model.get_gradient(act, np.repeat(class_id, n_det), config.bottlenecks[0]).reshape(n_det, -1)
    np.save(os.path.join(output_dir, Path(names[i]).stem + ".npy"), grad, allow_pickle=False)
