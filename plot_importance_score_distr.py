"""plots tp/fp importance score's mean and std"""

import sys
sys.path.insert(0, "/home/u00215/git/detectron2/tools/")

import custom

import argparse
import json
import os
from pathlib import Path
import pickle
from typing import List, Optional
import re
from tqdm import tqdm

import numpy as np
from pycocotools.coco import COCO
from detectron2.data import MetadataCatalog
import h5py
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.linear_model import SGDClassifier
from ace import ace_helpers

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help="dataset id registered to detectron2")
parser.add_argument('category', type=str, help="target category")
parser.add_argument('data2_eval', type=str, help="path to data2_eval json")
parser.add_argument('data5_inner', type=str, help="path to inner hdf5")
parser.add_argument('--working_dir', type=str, default=".")
args = parser.parse_args()

with open(args.data2_eval) as f:
    eval_data = json.load(f)
meta = MetadataCatalog.get(args.dataset)
coco = COCO(meta.json_file)
imgname2id = {v['file_name']: k for k, v in coco.imgs.items()}
inner = h5py.File(args.data5_inner, "r")
innerFP = []
innerTP = []

concept_n = len(inner['/' + Path(coco.imgs[1]['file_name']).stem].keys()) // 20
for sample in tqdm(eval_data['results']):
    name = sample['name']
    idx = imgname2id[name]
    is_tps = np.array(sample['dets_eval']) == 1
    score = np.zeros((concept_n, 20, len(is_tps)))
    pattern = re.compile(r"{}_concept(\d+)-random500_(\d+)-".format(args.category))
    for k, v in inner['/' + Path(name).stem].items():
        obj = pattern.match(k)
        score[int(obj.group(1))-1, int(obj.group(2)), :] = -v[:]
    for i, tp in enumerate(is_tps):
        if tp:
            innerTP.append(score[:,:,i].mean(axis=-1))
        else:
            innerFP.append(score[:,:,i].mean(axis=-1))

innerFP = np.array(innerFP)
innerTP = np.array(innerTP)
print(f"TP inner rate: {(innerTP>0).sum() / innerTP.size}")
print(f"FP inner rate: {(innerFP>0).sum() / innerFP.size}")
df = pd.DataFrame({
    'TP_mean': innerTP.mean(0),
    'TP_std': innerTP.std(0),
    'FP_mean': innerFP.mean(0),
    'FP_std': innerFP.std(0)})
df.to_csv(os.path.join(args.working_dir, 'concept_stat.csv'))

X = np.arange(1, len(innerTP.mean(0))+1)
plt.errorbar(X, innerTP.mean(0), fmt='o', yerr=innerTP.std(0))
plt.errorbar(X + 0.1, innerFP.mean(0), fmt='o', yerr=innerFP.std(0))
plt.legend(['tp', 'fp'])
plt.xticks(range(1, 21))
plt.savefig(os.path.join(args.working_dir, "concept_stat.png"))

