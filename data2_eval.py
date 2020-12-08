import argparse
import json
import os

import torch
from tqdm import tqdm

# Detectron2
from detectron2.modeling.matcher import Matcher
from detectron2.structures.boxes import Boxes, pairwise_iou

parser = argparse.ArgumentParser()
parser.add_argument('target_class', help='person')
parser.add_argument('gt', help='~/git/CenterNet/data/bdd/detection_val.json')
parser.add_argument('det', help='~/git/detectron2_v/experiments/002/inference/bdd_val/coco_instances_results_bdd.json')
parser.add_argument('working_dir', help='output evals')
parser.add_argument('--iou', type=float, default=0.75, help='iou threshold')
args = parser.parse_args()

output_dir = os.path.join(args.working_dir, 'data2')
# Make a destination directory. If it already exists, raise an error.
os.mkdir(output_dir)

def load_names_and_boxeses(target_class, json_filename, keep_empty):
    with open(json_filename) as f:
        dets = json.load(f)
    names = []
    boxeses = []
    name = dets[0]['name']
    boxes = []
    n_dropped = 0
    for det in dets:
        assert name <= det['name']
        if name < det['name']:
            if keep_empty or boxes:
                names.append(name)
                boxeses.append(boxes)
            else:
                n_dropped += 1
            name = det['name']
            boxes = []
        if det['category'] == target_class:
            boxes.append(det['bbox'])
    names.append(name)
    boxeses.append(boxes)
    assert len(names) == len(boxeses)
    print('n_image: {:5}'.format(len(names) + n_dropped))
    print('  det>0: {:5}'.format(len(names)))
    print('  det=0: {:5}'.format(n_dropped))
    return names, boxeses

def iou(gt_boxes, det_boxes):
    gt_boxes = Boxes(torch.tensor(gt_boxes, dtype=torch.float32))
    det_boxes = Boxes(torch.tensor(det_boxes, dtype=torch.float32))
    return pairwise_iou(gt_boxes, det_boxes)

gt_names, gt_boxeses = load_names_and_boxeses(args.target_class, args.gt, True)
det_names, det_boxeses = load_names_and_boxeses(args.target_class, args.det, False)
gt = {name: boxes for name, boxes in zip(gt_names, gt_boxeses)}
matcher = Matcher([args.iou], [0, 1])
results = []
for i, name in enumerate(tqdm(det_names)):
    gt_boxes = gt[name]
    det_boxes = det_boxeses[i]
    if gt_boxes:
        matrix = iou(gt_boxes, det_boxes)
        _, evals_of_image = matcher(matrix)
        results.append({'name': name, 'dets': det_boxes, 'gts': gt_boxes, 'dets_eval': evals_of_image})
with open(os.path.join(output_dir, 'evaluate.json'), 'w') as f:
    json.dump({'category': args.target_class, 'results': results}, f)
