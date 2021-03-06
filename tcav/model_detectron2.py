import logging
import re

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from tcav.model import PublicModelWrapper
# Detectron2
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures import Boxes

logger = logging.getLogger(__name__)


class FasterRCNNR50C4Wrapper(PublicModelWrapper):
  do_not_use_tf_session = True

  def __init__(self, sess, model_saved_path, labels_path, fix_ratio, config_path):
    with open(labels_path) as f:
      self.labels = f.read().splitlines()
    self.image_shape = [600, 600, 3]  # TODO

    self.cfg = cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.freeze()
    self.model = build_model(cfg)
    self.model.eval()
    DetectionCheckpointer(self.model).load(model_saved_path)
    self.fix_ratio = fix_ratio
    if not fix_ratio:
      self.transform_gen = T.ResizeShortestEdge(
          [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

  def get_gradient(self, acts, y, bottleneck_name):
    match = re.fullmatch('res5_([0-3])', bottleneck_name)
    assert match
    res5_i = int(match.group(1))

    model = self.model
    y = torch.tensor(y, device=model.device)
    x = torch.as_tensor(acts.transpose(0, 3, 1, 2), device=model.device).requires_grad_()
    box_features = model.roi_heads.res5[res5_i:](x)
    feature_pooled = box_features.mean(dim=[2, 3])
    pred_class_logits, _ = model.roi_heads.box_predictor(feature_pooled)
    loss_cls = F.cross_entropy(pred_class_logits, y, reduction="mean")
    loss_cls.backward()
    return x.grad.cpu().numpy().transpose(0, 2, 3, 1)

  def run_imgs(self, imgs, bottleneck_name, _boxes=None):
    match = re.fullmatch('res5_([0-3])', bottleneck_name)
    assert match
    res5_i = int(match.group(1))

    def run_img(img):
      logger.debug('img.shape: {}'.format(img.shape))
      img = img[:, :, ::-1]
      if self.fix_ratio:
        transform = T.ResizeTransform(*img.shape[:2], *[self.cfg.INPUT.MIN_SIZE_TEST]*2, Image.BILINEAR)
      else:
        transform = self.transform_gen.get_transform(img)
      img = apply_image(transform, img)
      input_y, input_x = img.shape[:2]
      if _boxes is None:
        boxes = [[0, 0, input_x, input_y]]
      else:
        boxes = np.asarray(_boxes).reshape(-1, 2)
        boxes = transform.apply_coords(boxes)
        boxes = boxes.reshape(-1, 4)
      with torch.no_grad():
        img = torch.as_tensor(img.astype('float32').transpose(2, 0, 1))
        model = self.model
        images = model.preprocess_image([{'image': img}])
        features = model.backbone(images.tensor)
        features = [features[f] for f in model.roi_heads.in_features]
        x = model.roi_heads.pooler(features, [Boxes(torch.tensor(boxes, dtype=torch.float32, device=model.device))])
        x = model.roi_heads.res5[:res5_i](x)
        return x.cpu().numpy().transpose(0, 2, 3, 1)

    return np.concatenate([run_img(img) for img in imgs])


def apply_image(self: T.ResizeTransform, img):
  assert img.shape[:2] == (self.h, self.w)
  assert self.interp == Image.BILINEAR
  trans_input = cv2.getAffineTransform(
      np.array([[0, 0], [self.w, 0], [0, self.h]], dtype=np.float32),
      np.array([[0, 0], [self.new_w, 0], [0, self.new_h]], dtype=np.float32))
  img = cv2.warpAffine(img, trans_input,
                       (self.new_w, self.new_h),
                       flags=cv2.INTER_LINEAR)
  return img
