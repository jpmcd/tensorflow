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
#  images_feed, logits = data_set.next_batch(FLAGS.batch_size)

  # Sample teacher prediction using multinomial on
  # softmax of logits by Gumbel trick
#  labels_feed = np.argmax(logits -
#    np.log(-np.log(np.random.uniform(size=logits.shape))), axis=1)

  feed_dict = {images_pl: images_feed, labels_pl: labels_feed}

  return feed_dict


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    st_global_step = tf.Variable(0, trainable=False)

    images = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT,
                            IMAGE_WIDTH, IMAGE_DEPTH))
    logits = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES)) 
    targets = cifar10.multinomial(logits)

    with tf.variable_scope('student') as s_scope:
      # Build a Graph that computes the logits predictions from the
      # inference model.
      st_logits = cifar10.inference(images)

      # Calculate loss.
      st_loss = cifar10.loss(st_logits, targets)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    st_train_op = cifar10.train(st_loss, st_global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

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

    data_set = Dataset(images_set, logits_set)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
#      feed_dict = fill_feed_dict(data_set, images, targets)
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

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model_student.ckpt')
        saver.save(sess, checkpoint_path, global_step=step,
                   latest_filename='checkpoint_student')


def train_simult():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    lg_global_step = tf.Variable(0, trainable=False)
    md_global_step = tf.Variable(0, trainable=False)
    sm_global_step = tf.Variable(0, trainable=False)
    mds_global_step = tf.Variable(0, trainable=False)
    sms_global_step = tf.Variable(0, trainable=False)
    md_l_global_step = tf.Variable(0, trainable=False)
    sm_l_global_step = tf.Variable(0, trainable=False)
    mds_l_global_step = tf.Variable(0, trainable=False)
    sms_l_global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    with tf.device('/cpu:0'):
      with tf.variable_scope('train') as scope:
        images, labels = cifar10.distorted_inputs()
      with tf.variable_scope('eval') as scope:
        images_ev, labels_ev = cifar10.inputs(eval_data='test')

    with tf.variable_scope('model') as scope:
      # Build a Graph that computes the logits predictions from the
      # inference model.
      logits = cifar10.inference(images)
      targets = cifar10.multinomial(logits) 

      # Calculate loss.
      loss = cifar10.loss(logits, labels)

      # Compute logits and calculate predictions for validation error
      scope.reuse_variables()
      logits_ev = cifar10.inference(images_ev)
      top_k_op = tf.nn.in_top_k(logits_ev, labels_ev, 1)

      # Build a graph that trains the model with one batch of examples
      # and updates the model parameters.
      train_op = cifar10.train(loss, global_step)

    with tf.variable_scope('lrg') as scope:
      # Student graph that computes the logits predictions from the
      # inference model.
      lg_logits = cifar10.inference(images)
      lg_targets = cifar10.multinomial(lg_logits)

      # Calculate loss according to multinomial sampled target predictions,
      # equally weighted against original loss with labels.
      lg_loss = tf.add(loss, cifar10.loss(lg_logits, targets))

      scope.reuse_variables()
      lg_logits_ev = cifar10.inference(images_ev)
      lg_top_k_op = tf.nn.in_top_k(lg_logits_ev, labels_ev, 1)
      lg_train_op = cifar10.train(lg_loss, lg_global_step)

    with tf.variable_scope('med') as scope:
      md_logits = cifar10.inference_vars(images, 48, 48, 192, 96)
      md_targets = cifar10.multinomial(md_logits)
      md_loss = tf.add(loss, cifar10.loss(md_logits, lg_targets))
      scope.reuse_variables()
      md_logits_ev = cifar10.inference_vars(images_ev, 48, 48, 192, 96)
      md_top_k_op = tf.nn.in_top_k(md_logits_ev, labels_ev, 1)
      md_train_op = cifar10.train(md_loss, md_global_step)

    with tf.variable_scope('sml') as scope:
      sm_logits = cifar10.inference_vars(images, 32, 32, 96, 48)
      sm_loss = tf.add(loss, cifar10.loss(sm_logits, md_targets))
      scope.reuse_variables()
      sm_logits_ev = cifar10.inference_vars(images_ev, 32, 32, 96, 48)
      sm_top_k_op = tf.nn.in_top_k(sm_logits_ev, labels_ev, 1)
      sm_train_op = cifar10.train(sm_loss, sm_global_step)

    # Medium sized model trained on labels
    with tf.variable_scope('med_lab') as scope:
      mds_logits = cifar10.inference_vars(images, 48, 48, 192, 96)
      mds_loss = cifar10.loss(mds_logits, labels)
      scope.reuse_variables()
      mds_logits_ev = cifar10.inference_vars(images_ev, 48, 48, 192, 96)
      mds_top_k_op = tf.nn.in_top_k(mds_logits_ev, labels_ev, 1)
      mds_train_op = cifar10.train(mds_loss, mds_global_step)

    # Small sized model trained on labels
    with tf.variable_scope('sml_lab') as scope:
      sms_logits = cifar10.inference_vars(images, 32, 32, 96, 48)
      sms_loss = cifar10.loss(sms_logits, labels)
      scope.reuse_variables()
      sms_logits_ev = cifar10.inference_vars(images_ev, 32, 32, 96, 48)
      sms_top_k_op = tf.nn.in_top_k(sms_logits_ev, labels_ev, 1)
      sms_train_op = cifar10.train(sms_loss, sms_global_step)

    # Medium sized model trained on large model, delayed start
    with tf.variable_scope('med_late') as scope:
      md_l_logits = cifar10.inference_vars(images, 48, 48, 192, 96)
      md_l_loss = tf.add(loss, cifar10.loss(md_l_logits, lg_targets))
      scope.reuse_variables()
      md_l_logits_ev = cifar10.inference_vars(images_ev, 48, 48, 192, 96)
      md_l_top_k_op = tf.nn.in_top_k(md_l_logits_ev, labels_ev, 1)
      md_l_train_op = cifar10.train(md_l_loss, md_l_global_step)

    # Small sized model trained on medium model, delayed start
    with tf.variable_scope('sml_late') as scope:
      sm_l_logits = cifar10.inference_vars(images, 32, 32, 96, 48)
      sm_l_loss = tf.add(loss, cifar10.loss(sm_l_logits, md_targets))
      scope.reuse_variables()
      sm_l_logits_ev = cifar10.inference_vars(images_ev, 32, 32, 96, 48)
      sm_l_top_k_op = tf.nn.in_top_k(sm_l_logits_ev, labels_ev, 1)
      sm_l_train_op = cifar10.train(sm_l_loss, sm_l_global_step)

    # Medium sized model trained on labels, delayed start
    with tf.variable_scope('med_lab_late') as scope:
      mds_l_logits = cifar10.inference_vars(images, 48, 48, 192, 96)
      mds_l_loss = cifar10.loss(mds_l_logits, labels)
      scope.reuse_variables()
      mds_l_logits_ev = cifar10.inference_vars(images_ev, 48, 48, 192, 96)
      mds_l_top_k_op = tf.nn.in_top_k(mds_l_logits_ev, labels_ev, 1)
      mds_l_train_op = cifar10.train(mds_l_loss, mds_l_global_step)

    # Small sized model trained on labels, delayed start
    with tf.variable_scope('sml_lab_late') as scope:
      sms_l_logits = cifar10.inference_vars(images, 32, 32, 96, 48)
      sms_l_loss = cifar10.loss(sms_l_logits, labels)
      scope.reuse_variables()
      sms_l_logits_ev = cifar10.inference_vars(images_ev, 32, 32, 96, 48)
      sms_l_top_k_op = tf.nn.in_top_k(sms_l_logits_ev, labels_ev, 1)
      sms_l_train_op = cifar10.train(sms_l_loss, sms_l_global_step)


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
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    num_examples = 10000
    num_iter = int(np.ceil(num_examples / FLAGS.batch_size))
    total_sample_count = (num_iter-1) * FLAGS.batch_size
    
    accuracy = []
    losses = []

#    if True:
#      for step in range(10000):
#        V0, V1, V2 = sess.run([images_ev[0,0,0,0], images_ev[1,0,0,0], images_ev[2,0,0,0]])
#        print(step, V0, V1, V2)

    sm_val = 0.0
    md_val = 0.0
    sms_val = 0.0
    mds_val = 0.0
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      if step < 5000:
        _, loss_value, __, lg_loss_value = sess.run([train_op,
          loss, lg_train_op, lg_loss])
      elif step < 10000:
        (_, loss_value, __, lg_loss_value, ___, md_val,
          ___s) = sess.run([train_op,
          loss, lg_train_op, lg_loss, md_train_op, md_loss,
          mds_train_op])
      elif step < 1E5:
        (_, loss_value, __, lg_loss_value, ___, md_val, ____, sm_val,
          ___s,  ____s) = sess.run([train_op,
          loss, lg_train_op, lg_loss, md_train_op, md_loss, sm_train_op, sm_loss,
          mds_train_op, sms_train_op])
      else:
        (_, loss_value, __, lg_loss_value, ___, md_val, ____, sm_val,
          ___s,  ____s, __mlto, __slto,
          __mslto, __sslto) = sess.run([train_op,
          loss, lg_train_op, lg_loss, md_train_op, md_loss, sm_train_op, sm_loss,
          mds_train_op, sms_train_op, md_l_train_op, sm_l_train_op,
          mds_l_train_op, sms_l_train_op])
      duration = time.time() - start_time

      assert not np.isnan(lg_loss_value), 'Model diverged with loss = NaN'

      losses.append(np.array([loss_value, lg_loss_value, md_val, sm_val]))

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f, lg_loss = %.2f, md_loss = %.2f, '
                      'sm_loss = %.2f, (%.1f examples/sec; %.3f sec/batch)')
        print (format_str % (datetime.now(), step, loss_value, lg_loss_value, md_val,
                             sm_val, examples_per_sec, sec_per_batch))

      if step % 500 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
        true_count = np.zeros(10)
        #eval_step = 0
        #while eval_step < num_iter-1:
        for eval_step in xrange(num_iter-1):
          predictions = sess.run([top_k_op,
            lg_top_k_op, md_top_k_op, sm_top_k_op,
            mds_top_k_op, sms_top_k_op, md_l_top_k_op, sm_l_top_k_op,
            mds_l_top_k_op, sms_l_top_k_op])
          predictions = np.array(predictions)
          true_count += np.sum(predictions, axis=1)
          #eval_step += 1

        precision = true_count / total_sample_count 
        print (precision)
        accuracy.append(precision)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model_student.ckpt')
        saver.save(sess, checkpoint_path, global_step=step,
                   latest_filename='checkpoint_student')

        eval_history = np.array(accuracy).T
        loss_history = np.array(losses).T
        np.save(os.path.join(FLAGS.train_dir, 'eval_history'), eval_history)
        np.save(os.path.join(FLAGS.train_dir, 'loss_history'), loss_history)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
#  train()
  train_simult()


if __name__ == '__main__':
  tf.app.run()
