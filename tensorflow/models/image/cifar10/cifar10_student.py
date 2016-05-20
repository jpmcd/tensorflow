# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10
import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/scratch/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

IMAGE_HEIGHT = 24
IMAGE_WIDTH = 24
IMAGE_DEPTH = 3
NUM_CLASSES = 10


class Dataset(object):

  def __init__(self, images, labels):
    assert images.shape[0] == labels.shape[0]
    self.images = images
    self.labels = labels
    self.num_examples = self.images.shape[0]
    self.index_in_epoch = 0
    self.epochs_completed = 0

  def next_batch(self, batch_size):
    start = self.index_in_epoch
    self.index_in_epoch += batch_size

    if self.index_in_epoch > self.num_examples:
      # Finished epoch
      self.epochs_completed += 1

      # Shuffle the data
      perm = np.arange(self.num_examples)
      np.random.shuffle(perm)
      self.images = self.images[perm]
      self.labels = self.labels[perm]

      # Start next epoch
      start = 0
      self.index_in_epoch = batch_size
      assert batch_size <= self.num_examples

    end = self.index_in_epoch

    return self.images[start:end], self.labels[start:end]
  

def fill_feed_dict(data_set, images_pl, labels_pl):
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
  feed_dict = {images_pl: images_feed, labels_pl: labels_feed}

  return feed_dict


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
#    with tf.variable_scope('student') as s_scope:
    st_global_step = tf.Variable(0, trainable=False)

    images = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT,
                            IMAGE_WIDTH, IMAGE_DEPTH))
    logits = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    st_logits_unnorm = cifar10.inference(images)
    st_logits = tf.sub(st_logits_unnorm, tf.reduce_mean(st_logits_unnorm,
                       1, keep_dims=True))

    # Calculate loss.
    st_loss = cifar10.target_loss(st_logits, logits)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    st_train_op = cifar10.train(st_loss, st_global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    #tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)

    images_path = os.path.join(FLAGS.data_dir, 'img.npz')
    logits_path = os.path.join(FLAGS.train_dir, 'log.npz')

    if not tf.gfile.Exists(images_path):
      raise ValueError('Failed to find file: ' + images_path)
    if not tf.gfile.Exists(logits_path):
      raise ValueError('Failed to find file: ' + logits_path)

    with np.load(images_path) as data:
      images_set = data['images_set']
      print ('images_set shape type ', images_set.shape, images_set.dtype)
    with np.load(logits_path) as data:
      logits_set = data['logits_set']
      # Normalize logits by mean
      logits_set = logits_set - np.mean(logits_set, axis=1, keepdims=True)
      print ('logits_set shape type ', logits_set.shape, logits_set.dtype)

    data_set = Dataset(images_set, logits_set)

    for step in xrange(FLAGS.max_steps):
      print (step)
      start_time = time.time()
      feed_dict = fill_feed_dict(data_set, images, logits)
      _, st_loss_value = sess.run([st_train_op, st_loss], feed_dict=feed_dict)
      duration = time.time() - start_time

      assert not np.isnan(st_loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, st_loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, st_loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model_student.ckpt')
        saver.save(sess, checkpoint_path, global_step=step,
                   latest_filename='checkpoint_student')


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  train()


if __name__ == '__main__':
  tf.app.run()
