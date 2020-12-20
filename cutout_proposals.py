import sys
sys.path.insert(0, "/home/u00215/git/detectron2/tools/")
import custom

import argparse
from pathlib import Path
import pickle
from typing import List

from detectron2.data import MetadataCatalog
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from PIL import Image
from skimage.segmentation import slic
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help="dataset id registered to detectron2")
parser.add_argument('category', help="target category name")
parser.add_argument('lm', help="path to the folder of linear-model concept classifiers")
parser.add_argument('concepts', help=",-delimiter integer sequence of target concepts for cutout")
parser.add_argument('working_dir')
parser.add_argument('--n_segments', default="15,50,80", help=",-delimiter integer sequence of the number of superpixel segments")
parser.add_argument('--compactness', default="20,20,20", help=",-delimiter float sequence of the compactness of slic superpixels")
parser.add_argument('--sigma', default="1,1,1", help=",-delimiter float sequence of the compactness of slic superpixels")
args = parser.parse_args()
concepts = list(map(int, args.concepts.split(',')))
n_segments = list(map(int, args.n_segments.split(',')))
compactness = list(map(float, args.compactness.split(',')))
sigma = list(map(float, args.sigma.split(',')))

def filter_segments(segments, unique_masks):
    for s in range(segments.max()+1):
        mask = (segments == s).astype(np.float32)
        if np.mean(mask) > 0.001:
            unique = True
            for seen_mask in unique_masks:
                jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                if jaccard > 0.5:
                    unique = False
                    break
            if unique:
                param_masks.append(mask)
        unique_masks.extend(param_masks)

def extract_patch(image, mask):
    """Parameters:
        image: float(h, w, ch), mask: float (0.0 or 1.0) (h, w)
    Returns:
        patch_cropped (PIL.Image): """
    mask_expanded = np.expand_dims(mask, -1)
    patch = (mask_expanded * image +
        (1 - mask_expanded) * float(self.average_image_value) / 255)  # outside mask will be filled by average
    # calculate bbox of mask
    ones = np.where(mask == 1)
    h1, h2, w1, w2 = ones[0].min(), ones[0].max() + 1, ones[1].min(), ones[1].max() + 1
    # crop
    patch_cropped = patch[h1:h2, w1:w2]
    if self.resize_patches:
        patch_cropped = Image.fromarray((patch_cropped * 255).astype(np.uint8))
        patch_cropped = patch_cropped.resize(self.image_shape, Image.BICUBIC)
        patch_cropped = np.array(patch_cropped).astype(np.float32) / 255
    return patch_cropped, patch

def return_superpixels(image: np.ndarray):
    """image: [0.0, 1.0] float32 image"""
    unique_masks = []
    for n_seg, comp, sig in zip(n_segments, compactness, sigma):
        segments = slic(image, n_segments=n_seg, compactness=comp, sigma=sig)
        filter_segments(segments, unique_masks)

# TODO: parallelize
def create_df(anns: List[dict], image: Image) -> pd.DataFrame:
    """Parameters:
        anns: annotation list from coco api
        image: PIL image
    Returns:
        df (pd.DataFrame): dataframe to save for given image"""
    ret = {
        'annot_id': [],
        'bbox': [],
        'rlemask': []
    }
    for ann in anns:
        x,y,w,h = ann['bbox']
        crop_image = image.crop(x, y, x+w, y+h)
        crop_image = np.asarray(crop_image).astype(np.float32) / 255.0
        return_superpixels(crop_image)
    return pd.DataFrame(ret)

if __name__ == '__main__':
    WORKDIR = Path(args.working_dir)
    meta = MetadataCatalog(args.dataset)
    api = COCO(meta.json_file)
    IMGROOT = Path(meta.image_root)
    store = pd.HDFStore(str(WORKDIR / 'cutout_proposals.h5'))
    for image_idx, image_info in tqdm(api.imgs.items(), total=len(list(api.imgs.keys()))):
        ann_indice = api.getAnnIds([image_idx])
        anns = api.loadAnns(ann_indice)
        df = create_df(anns, Image.open(IMGROOT / image_info['file_name']))
        store.put('%d' % image_idx, df)

