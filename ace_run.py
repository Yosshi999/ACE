"""This script runs the whole ACE method."""
import argparse
import logging
import os
import sys

import logzero
import tensorflow as tf

import ace.config
from ace import ace_helpers
from ace.ace import ConceptDiscovery
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

  source_dir = args.config.source_dir
  test_dir = args.config.test_dir
  working_dir = args.working_dir
  target_class = args.config.target_class
  target_class_mask = args.config.target_class_mask or None
  bottlenecks = list(args.config.bottlenecks)
  num_test = args.num_test
  num_random_exp = args.num_random_exp
  max_imgs = args.config.max_imgs
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
  random_concept = 'random500_{}'.format(num_random_exp)  # Random concept for statistical testing
  sess = utils.create_session()
  mymodel = ace_helpers.make_model(args.config.model, sess)
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
      num_workers=args.config.num_workers,
      resize_images=args.config.resize_images,
      resize_patches=args.config.resize_patches)
  # Creating the dataset of image patches
  cd.create_patches(param_dict={'n_segments': list(args.config.slic.n_segments)})
  # Discovering Concepts
  cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
  del cd.dataset  # Free memory
  del cd.image_numbers
  del cd.patches
  # Calculating CAVs and TCAV scores
  cav_accuraciess = cd.cavs(min_acc=0.0)
  scores = cd.tcavs(test=False)
  ace_helpers.save_ace_report(cd, cav_accuraciess, scores,
                                 results_summaries_dir + 'ace_results.txt')
  # Plot examples of discovered concepts
  for bn in cd.bottlenecks:
    ace_helpers.plot_concepts(cd, bn, 10, address=results_dir)

def parse_arguments(argv):
  """Parses the arguments passed to the run.py script."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=ace.config.load, required=True)
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
