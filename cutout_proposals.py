import sys
sys.path.insert(0, "/home/u00215/git/detectron2/tools/")
import custom

import argparse
from collections import defaultdict
import logging
from multiprocessing import Pool
import os
from pathlib import Path
import pickle
from typing import List, Tuple

import ace.config
from ace import ace_helpers
from detectron2.data import MetadataCatalog
import logzero
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
from skimage.segmentation import slic
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('config', help='model config. model.fix_ratio is forced to False.')
parser.add_argument('--bottleneck', help='override bottleneck')
parser.add_argument('dataset', help="dataset id registered to detectron2")
parser.add_argument('category', help="target category name")
parser.add_argument('lm', help="path to the folder of linear-model concept classifiers")
parser.add_argument('concepts', help=",-delimiter integer sequence of target concepts for cutout")
parser.add_argument('working_dir')
parser.add_argument('--n_segments', default="15,50,80", help=",-delimiter integer sequence of the number of superpixel segments")
parser.add_argument('--compactness', default="20,20,20", help=",-delimiter float sequence of the compactness of slic superpixels")
parser.add_argument('--sigma', default="1,1,1", help=",-delimiter float sequence of the compactness of slic superpixels")
parser.add_argument('--worker', type=int, default=0, help="multiprocessing worker. if 0, it is processed sequentially. if -1, use cpu count")
args = parser.parse_args()
concepts = list(map(int, args.concepts.split(',')))
n_segments = list(map(int, args.n_segments.split(',')))
compactness = list(map(float, args.compactness.split(',')))
sigma = list(map(float, args.sigma.split(',')))

_average_image_value = 117
_resize_patches = False
_image_shape = None

def setup_logger(working_dir):
    color_formatter = logzero.LogFormatter()
    monochrome_formatter = logzero.LogFormatter(color=False)
  
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(color_formatter)
    color_file_handler = logging.FileHandler(os.path.join(working_dir, 'log.color'))
    color_file_handler.setFormatter(color_formatter)
    txt_file_handler = logging.FileHandler(os.path.join(working_dir, 'log.txt'))
    txt_file_handler.setFormatter(monochrome_formatter)
  
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(color_file_handler)
    logger.addHandler(txt_file_handler)

def filter_segments(segments: np.ndarray, unique_masks: List[np.ndarray]):
    """filter segments and append them to unique_masks"""
    for s in range(segments.max()+1):
        mask = (segments == s).astype(np.float32)
        param_masks = []
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

def extract_patch(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int], np.ndarray]:
    """Parameters:
        image: float(h, w, ch), mask: float (0.0 or 1.0) (h, w)
    Returns:
        patch_cropped float (h, w, ch): cropped image"""
    mask_expanded = np.expand_dims(mask, -1)
    patch = (mask_expanded * image +
        (1 - mask_expanded) * float(_average_image_value) / 255)  # outside mask will be filled by average
    # calculate bbox of mask
    ones = np.where(mask == 1)
    h1, h2, w1, w2 = ones[0].min(), ones[0].max() + 1, ones[1].min(), ones[1].max() + 1
    # crop
    patch_cropped = patch[h1:h2, w1:w2]
    if _resize_patches:
        patch_cropped = Image.fromarray((patch_cropped * 255).astype(np.uint8))
        patch_cropped = patch_cropped.resize(_image_shape, Image.BICUBIC)
        patch_cropped = np.array(patch_cropped).astype(np.float32) / 255
    return patch_cropped, (w1, h1, w2, h2), mask

def return_superpixel_patches(image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int]], List[np.ndarray]]:
    """image: [0.0, 1.0] float32 image"""
    unique_masks = []
    for n_seg, comp, sig in zip(n_segments, compactness, sigma):
        segments = slic(image, n_segments=n_seg, compactness=comp, sigma=sig, start_label=0)
        filter_segments(segments, unique_masks)
    patches = []
    bboxes = []
    masks = []
    for p, b, m in map(lambda _mask: extract_patch(image, _mask), unique_masks):
        patches.append(p)
        bboxes.append(b)
        masks.append(m)
    return patches, bboxes, masks

class LMTest:
    """load trained linear models and test acts"""
    def __init__(self, lm_path: str, category: str, concepts: List[int]):
        lm_dict = defaultdict(dict)
        target_cnames = [f'{category}_concept{c}' for c in concepts]
        dim = None # dimension of linear model's decision function
        for fn in Path(lm_path).iterdir():
            cname, randc, *_ = fn.stem.split("-")
            if not cname in target_cnames:
                continue
            with fn.open("rb") as f:
                data = pickle.load(f)
            lm: SGDClassifier = data['lm']
            lm_dict[cname][randc] = lm
            if dim is None:
                dim = lm.coef_.shape[1]
        self.n_concept = len(lm_dict.keys())
        self.id2concept = list(lm_dict.keys())
        self.concept2id = {c: i for i, c in enumerate(self.id2concept)}
        self.n_random = len(next(iter(lm_dict.values())).keys())
        for lm_cname in lm_dict.values():
            assert len(lm_cname.keys()) == self.n_random
        assert len(lm_dict.keys()) > 0
        self.stat = {'dets': 0, **{k: 0 for k in lm_dict.keys()}}
        self.lm_coef = np.zeros((self.n_concept, self.n_random, dim), np.float32)
        self.lm_bias = np.zeros((self.n_concept, self.n_random, 1), np.float32)
        for i in range(self.n_concept):
            for j, lm in enumerate(lm_dict[self.id2concept[i]].values()):
                self.lm_coef[i, j:j+1] = lm.coef_
                self.lm_bias[i, j] = lm.intercept_
    def test(self, acts: np.ndarray) -> List[str]:
        """tests if given acts is classified to one of the target concepts
        Parameters:
            acts: list of activations. (n_samples, n_features)
        Returns:
            inner: concept name which each act is classified to. None if no concept.
        """
        scores = np.zeros((self.n_concept, self.n_random, len(acts)), np.float32)
        scores[:] = np.matmul(self.lm_coef, acts.T) + self.lm_bias
        predictions = (scores > 0).astype(int)
        # label 0 for concept, 1 for random; if sum is 0 along to n_random, the act is inner.
        ret = []
        inner = predictions.sum(axis=1) == 0
        for i in range(len(acts)):
            y = np.where(inner[:,i])[0]
            self.stat['dets'] += 1
            if len(y) == 0:
                ret.append(None)
            else:
                cname = self.id2concept[y[0]]
                self.stat[cname] += 1
                ret.append(cname)
        return ret

# TODO: parallelize
def create_df(anns: List[dict], image: Image, model, bottleneck: str, lmtest: LMTest) -> dict:
    """Parameters:
        anns: annotation list from coco api
        image: PIL image
        model: loaded model
        bottleneck: name of bottleneck to extract
    Returns:
        df (pd.DataFrame): dataframe to save for given image"""
    ret = {
        'annot_id': [],
        'bbox': [],
        'concept': [],
        'rlemask': []
    }
    for ann in anns:
        x,y,w,h = ann['bbox']
        crop_image = image.crop((x, y, x+w, y+h))
        crop_image = np.asarray(crop_image)
        assert isinstance(crop_image, np.ndarray), str(crop_image)
        crop_image = crop_image.astype(np.float32) / 255.0
        superpixels, bboxes, masks = return_superpixel_patches(crop_image)
        acts = model.run_imgs(superpixels, bottleneck)

        ## channel mean
        # acts = np.mean(acts, (1, 2))
        # just flatten
        acts = np.reshape(acts, [acts.shape[0], -1])

        inners = lmtest.test(acts)
        for inner, bbox, mask in zip(inners, bboxes, masks):
            if inner is None:
                continue 
            ret['annot_id'].append(ann['id'])
            ret['bbox'].append(bbox)
            ret['concept'].append(inner)
            ret['rlemask'].append(maskUtils.encode(np.asfortranarray(mask.astype(np.uint8))))
    return ret
 
def create_df_worker(args):
    """parallelizable jobs"""
    ann, image, image_idx = args
    x,y,w,h = ann['bbox']
    crop_image = image.crop((x, y, x+w, y+h))
    crop_image = np.asarray(crop_image).astype(np.float32) / 255.0
    ret = return_superpixel_patches(crop_image)
    return ret, image_idx, ann

class CreateDFSingle:
    def __init__(self, model, bottleneck, lmtest):
        self.model = model
        self.bottleneck = bottleneck
        self.lmtest = lmtest
        self.reset()
    def get_df(self):
        return self.ret
    def reset(self):
        self.ret = {
            'annot_id': [],
            'bbox': [],
            'concept': [],
            'rlemask': []
        }
    def __call__(self, interm, ann):
        superpixels, bboxes, masks = interm
        acts = self.model.run_imgs(superpixels, self.bottleneck)

        ## channel mean
        # acts = np.mean(acts, (1, 2))
        # just flatten
        acts = np.reshape(acts, [acts.shape[0], -1])
        inners = self.lmtest.test(acts)
        for inner, bbox, mask in zip(inners, bboxes, masks):
            if inner is None:
                continue 
            self.ret['annot_id'].append(ann['id'])
            self.ret['bbox'].append(bbox)
            self.ret['concept'].append(inner)
            self.ret['rlemask'].append(maskUtils.encode(np.asfortranarray(mask.astype(np.uint8))))

if __name__ == '__main__':
    #setup_logger(args.working_dir)
    logger = logging.getLogger(__name__)
    logger.debug(str(args))
    # setup mode
    config = ace.config.load(args.config)
    config.model.fix_ratio = False
    if args.bottleneck is not None:
        config.bottlenecks[:] = [args.bottleneck]
    assert len(config.bottlenecks) == 1
    model = ace_helpers.make_model(config.model)
    _image_shape = model.get_image_shape()[:2]
    lmtest = LMTest(args.lm, args.category, concepts)

    # dataset
    WORKDIR = Path(args.working_dir)
    meta = MetadataCatalog.get(args.dataset)
    api = COCO(meta.json_file)
    IMGROOT = Path(meta.image_root)
    global_step = 0
    if args.worker == 0:
        for image_idx, image_info in tqdm(api.imgs.items(), total=len(list(api.imgs.keys()))):
            global_step += 1
            if global_step % 100:
                logger.debug(lmtest.stat)
            ann_indice = api.getAnnIds([image_idx])
            anns = api.loadAnns(ann_indice)
            df = create_df(anns, Image.open(IMGROOT / image_info['file_name']), model, config.bottlenecks[0], lmtest)
            with (WORKDIR / ('image%d.pickle' % image_idx)).open('wb') as f:
                pickle.dump(df, f)
    else:
        if args.worker == -1:
            args.worker = os.cpu_count()
        def preprocess_single_gen(args_list):
            for args in args_list:
                image_idx, image_info = args
                ann_indice = api.getAnnIds([image_idx])
                anns = api.loadAnns(ann_indice)
                im = Image.open(IMGROOT / image_info['file_name'])
                for an in anns:
                    yield an, im, image_idx
        with Pool(args.worker) as p:
            mapper = preprocess_single_gen(api.imgs.items())
            create_df_single = CreateDFSingle(model, config.bottlenecks[0], lmtest)
            prev_idx = None
            for interm, image_idx, ann in tqdm(p.imap(create_df_worker, mapper), total=len(list(api.anns.keys()))):
                global_step += 1
                if global_step % 100:
                    logger.debug(lmtest.stat)
                if prev_idx is None:
                    prev_idx = image_idx
                elif prev_idx != image_idx:
                    with (WORKDIR / ('image%d.pickle' % prev_idx)).open('wb') as f:
                        pickle.dump(create_df_single.get_df(), f)
                    prev_idx = image_idx
                    create_df_single.reset()
                create_df_single(interm, ann)
        #mapper = map(preprocess_single, api.imgs.items())
        #for interm, image_idx in tqdm(map(create_df_worker, mapper), total=len(list(api.imgs.keys()))):
        #    df = create_df_single(interm, model, config.bottlenecks[0], lmtest)
        #    with (WORKDIR / ('image%d.pickle' % image_idx)).open('wb') as f:
        #        pickle.dump(df, f)
    logger.debug(lmtest.stat)
