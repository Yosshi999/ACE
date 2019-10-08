"""This script runs the whole ACE method."""
import argparse
import logging
import os
import shutil
import subprocess
import sys

import logzero
import tensorflow as tf

import ace.config
from ace import ace_helpers
from ace.ace import ConceptDiscovery
from ace.timer import Timer
from tcav import utils


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


def main(args):
  config = ace.config.load(args.config)
  source_dir = config.source_dir
  test_dir = config.test_dir
  working_dir = args.working_dir
  target_class = config.target_class
  target_class_mask = config.target_class_mask or None
  bottlenecks = list(config.bottlenecks)
  num_test = args.num_test
  num_random_exp = args.num_random_exp
  max_imgs = config.max_imgs
  min_imgs = args.min_imgs
  ###### related DIRs on CNS to store results #######
  discovered_concepts_dir = os.path.join(working_dir, 'concepts/')
  results_dir = os.path.join(working_dir, 'results/')
  cavs_dir = os.path.join(working_dir, 'cavs/')
  activations_dir = os.path.join(working_dir, 'acts/')
  results_summaries_dir = os.path.join(working_dir, 'results_summaries/')
  if tf.gfile.Exists(working_dir):
    tf.gfile.DeleteRecursively(working_dir)
  tf.gfile.MakeDirs(working_dir)
  tf.gfile.MakeDirs(discovered_concepts_dir)
  tf.gfile.MakeDirs(results_dir)
  tf.gfile.MakeDirs(cavs_dir)
  tf.gfile.MakeDirs(activations_dir)
  tf.gfile.MakeDirs(results_summaries_dir)
  setup_logger(working_dir)
  shutil.copyfile(args.config, os.path.join(working_dir, 'config.pbtxt'))
  with open(os.path.join(working_dir, 'commit'), 'w') as f:
    subprocess.run(['git', 'log', '-1', '--format=%H'], stdout=f, cwd=os.path.dirname(__file__) or '.')
  timer = Timer(os.path.join(working_dir, 'timer.txt'), 'create_patches')
  random_concept = 'random500_{}'.format(num_random_exp)  # Random concept for statistical testing
  sess = utils.create_session()
  mymodel = ace_helpers.make_model(config.model, sess)
  # Creating the ConceptDiscovery class instance
  cd = ConceptDiscovery(
      mymodel,
      target_class,
      target_class_mask,
      random_concept,
      bottlenecks,
      sess,
      source_dir,
      activations_dir,
      cavs_dir,
      num_random_exp=num_random_exp,
      channel_mean=True,
      max_imgs=max_imgs,
      min_imgs=min_imgs,
      num_discovery_imgs=max_imgs,
      num_workers=config.num_workers,
      resize_images=config.resize_images,
      resize_patches=config.resize_patches)
  # Creating the dataset of image patches
  cd.create_patches(param_dict={'n_segments': list(config.slic.n_segments)})
  # Discovering Concepts
  timer('discover_concepts')
  cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
  del cd.dataset  # Free memory
  del cd.image_numbers
  del cd.patches
  # Calculating CAVs and TCAV scores
  timer('cav')
  cav_accuraciess = cd.cavs(min_acc=0.0)
  timer('tcav')
  scores = cd.tcavs(test=False)
  timer('save_ace_report')
  ace_helpers.save_ace_report(cd, cav_accuraciess, scores,
                                 results_summaries_dir + 'ace_results.txt')
  # Plot examples of discovered concepts
  timer('plot_concepts')
  for bn in cd.bottlenecks:
    ace_helpers.plot_concepts(cd, bn, 10, address=results_dir)
  timer.close()

def parse_arguments(argv):
  """Parses the arguments passed to the run.py script."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', required=True)
  parser.add_argument('--working_dir', type=str,
      help='Directory to save the results.', default='./ACE')
  parser.add_argument('--num_test', type=int,
      help="Number of test images used for binary profile classifier",
                      default=20)
  parser.add_argument('--num_random_exp', type=int,
      help="Number of random experiments used for statistical testing, etc",
                      default=20)
  parser.add_argument('--min_imgs', type=int,
      help="Minimum number of images in a discovered concept",
                      default=40)
  return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
