import logging

import numpy as np
import torch

from tcav.model import PublicModelWrapper
# CenterNet
from datasets.dataset_factory import get_dataset
from models.model import create_model, load_model
from opts import opts

logger = logging.getLogger(__name__)


class CenterNetWrapper(PublicModelWrapper):
  do_not_use_tf_session = True

  def __init__(self, sess, model_saved_path, labels_path):
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

  def get_gradient(self, acts, y, bottleneck_name):
    if bottleneck_name != 'y_last':
      raise NotImplementedError

    return np.zeros_like(acts)  # TODO

  def run_imgs(self, imgs, bottleneck_name):
    if bottleneck_name != 'y_last':
      raise NotImplementedError

    def do_run_imgs(imgs):
      logger.debug('imgs.shape: {}'.format(imgs.shape))
      imgs = (imgs - self.opt.mean) / self.opt.std
      imgs = imgs.transpose(0, 3, 1, 2)
      with torch.no_grad():
        x = torch.from_numpy(imgs)
        x = x.to(self.device)
        model = self.model
        x = model.base(x)
        x = model.dla_up(x)
        y = []
        for i in range(model.last_level - model.first_level):
          y.append(x[i].clone())
        model.ida_up(y, 0, len(y))
        return y[-1].cpu().numpy()

    if len(imgs.shape) == 4:
      return do_run_imgs(imgs)
    if len(imgs.shape) == 1:
      return np.concatenate([do_run_imgs(img[None]) for img in imgs])
    raise ValueError('len(imgs.shape) must be 4 or 1, got {}'.format(len(imgs.shape)))
