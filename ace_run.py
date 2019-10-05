"""This script runs the whole ACE method."""


import sys
import os
import numpy as np
import sklearn.metrics as metrics
from tcav import utils
import tensorflow as tf

from ace import ace_helpers
from ace.ace import ConceptDiscovery
import ace.config
import argparse


def main(args):

  source_dir = args.config.source_dir
  test_dir = args.config.test_dir
  working_dir = args.working_dir
  target_class = args.config.target_class
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
  random_concept = 'random500_{}'.format(num_random_exp)  # Random concept for statistical testing
  sess = utils.create_session()
  mymodel = ace_helpers.make_model(args.config.model, sess)
  # Creating the ConceptDiscovery class instance
  cd = ConceptDiscovery(
      mymodel,
      target_class,
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
  # Saving the concept discovery target class images
  image_dir = os.path.join(discovered_concepts_dir, 'images')
  tf.gfile.MakeDirs(image_dir)
  ace_helpers.save_images(image_dir,
                            (cd.discovery_images * 256).astype(np.uint8))
  # Discovering Concepts
  cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
  del cd.dataset  # Free memory
  del cd.image_numbers
  del cd.patches
  # Save discovered concept images (resized and original sized)
  ace_helpers.save_concepts(cd, discovered_concepts_dir)
  # Calculating CAVs and TCAV scores
  cav_accuraciess = cd.cavs(min_acc=0.0)
  scores = cd.tcavs(test=False)
  ace_helpers.save_ace_report(cd, cav_accuraciess, scores,
                                 results_summaries_dir + 'ace_results.txt')
  # Plot examples of discovered concepts
  for bn in cd.bottlenecks:
    ace_helpers.plot_concepts(cd, bn, 10, address=results_dir)
  # Delete concepts that don't pass statistical testing
  cd.test_and_remove_concepts(scores)
  # Train a binary classifier on concept profiles
  report = '\n\n\t\t\t ---Concept space---'
  report += '\n\t ---Classifier Weights---\n\n'
  pos_imgs = cd.load_concept_imgs(cd.target_class,
                                  2 * cd.max_imgs + num_test)[-num_test:]
  neg_imgs = cd.load_concept_imgs('random500_{}'.format(num_random_exp+1), num_test)
  a = ace_helpers.flat_profile(cd, pos_imgs)
  b = ace_helpers.flat_profile(cd, neg_imgs)
  lm, _ = ace_helpers.cross_val(a, b, methods=['logistic'])
  for bn in cd.bottlenecks:
    report += bn + ':\n'
    for i, concept in enumerate(cd.dic[bn]['concepts']):
      report += concept + ':' + str(lm.coef_[-1][i]) + '\n'
  # Test profile classifier on test images
  cd.source_dir = test_dir
  pos_imgs = cd.load_concept_imgs(cd.target_class, num_test)
  neg_imgs = cd.load_concept_imgs('random500_{}'.format(num_random_exp+1), num_test)
  a = ace_helpers.flat_profile(cd, pos_imgs)
  b = ace_helpers.flat_profile(cd, neg_imgs)
  x, y = ace_helpers.binary_dataset(a, b, balanced=True)
  probs = lm.predict_proba(x)[:, 1]
  report += '\nProfile Classifier accuracy= {}'.format(
      np.mean((probs > 0.5) == y))
  report += '\nProfile Classifier AUC= {}'.format(
      metrics.roc_auc_score(y, probs))
  report += '\nProfile Classifier PR Area= {}'.format(
      metrics.average_precision_score(y, probs))
  # Compare original network to profile classifier
  target_id = cd.model.label_to_id(cd.target_class.replace('_', ' '))
  predictions = []
  for img in pos_imgs:
    predictions.append(mymodel.get_predictions([img]))
  predictions = np.concatenate(predictions, 0)
  true_predictions = (np.argmax(predictions, -1) == target_id).astype(int)
  truly_predicted = np.where(true_predictions)[0]
  report += '\nNetwork Recall = ' + str(np.mean(true_predictions))
  report += ', ' + str(np.mean(np.max(predictions, -1)[truly_predicted]))
  agreeableness = np.sum(lm.predict(a) * true_predictions)*1./\
      np.sum(true_predictions + 1e-10)
  report += '\nProfile classifier agrees with network in {}%'.format(
      100 * agreeableness)
  with tf.gfile.Open(results_summaries_dir + 'profile_classifier.txt', 'w') as f:
    f.write(report)

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
