import logging

import cv2
import numpy as np
import torch

from tcav.model import PublicModelWrapper
# CenterNet
from datasets.dataset_factory import get_dataset
from models.model import create_model, load_model
from opts import opts
from utils.image import get_affine_transform

logger = logging.getLogger(__name__)


class CenterNetWrapper(PublicModelWrapper):
  do_not_use_tf_session = True

  def __init__(self, sess, model_saved_path, labels_path, fix_ratio):
    with open(labels_path) as f:
      self.labels = f.read().splitlines()
    self.image_shape = [600, 600, 3]  # TODO

    self.opt = opt = opts().parse('ctdet --dataset bdd --keep_res'.split())  # TODO: others than bdd
    Dataset = get_dataset(opt.dataset, opt.task)
    opts.update_dataset_info_and_set_heads(None, opt, Dataset)
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, model_saved_path)
    # TODO: loss
    # TODO: model_with_loss
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = model.to(self.device)  # TODO: model_with_loss
    self.model.eval()  # TODO: model_with_loss
    self.fix_ratio = fix_ratio

  def get_gradient(self, acts, y, bottleneck_name):
    return np.zeros_like(acts)  # TODO

  def run_imgs(self, imgs, bottleneck_name):
    def run_img(img):
      logger.debug('img.shape: {}'.format(img.shape))
      input_h, input_w = self.opt.input_h, self.opt.input_w
      if self.fix_ratio:
        trans_input = cv2.getAffineTransform(
            np.array([[0, 0], [img.shape[1], 0], [0, img.shape[0]]], dtype=np.float32),
            np.array([[0, 0], [input_w, 0], [0, input_h]], dtype=np.float32))
      else:
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
      img = cv2.warpAffine(img, trans_input,
                           (input_w, input_h),
                           flags=cv2.INTER_LINEAR)
      img = (img - self.opt.mean) / self.opt.std
      imgs = img.transpose(2, 0, 1)[None]
      with torch.no_grad():
        x = torch.from_numpy(imgs)
        x = x.to(self.device)
        model = self.model
        x = model.base(x)
        if bottleneck_name == 'base_last':
          return x[-1].cpu().numpy().transpose(0, 2, 3, 1)
        x = model.dla_up(x)
        if bottleneck_name == 'dla_up_0':
          return x[0].cpu().numpy().transpose(0, 2, 3, 1)
        y = []
        for i in range(model.last_level - model.first_level):
          y.append(x[i].clone())
        model.ida_up(y, 0, len(y))
        if bottleneck_name == 'y_last':
          return y[-1].cpu().numpy().transpose(0, 2, 3, 1)
        raise NotImplementedError

    return np.concatenate([run_img(img) for img in imgs])
