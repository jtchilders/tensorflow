# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Distributed MNIST training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on two parameter servers (ps), while the
ops are defined on a worker node. The TF sessions also run on the worker
node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dist_test/python/mnist_replica.py
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math,os
import sys
import tempfile
import time
import glob, json

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials import mnist
import numpy as np


flags = tf.app.flags
flags.DEFINE_string("data_dir", "data",
                    "Directory for storing mnist data")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 1,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
flags.DEFINE_string("ps_hosts","localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None,"job name: worker or ps")
flags.DEFINE_string("log_dir",None,"log file location and training checkpoints")

flags.DEFINE_string("checkpoint_dir", None,"location of checkpoints")
flags.DEFINE_string("checkpoint_prefix", 'model.ckpt-', "checkpoint prefix")
flags.DEFINE_string("output_json",'output.json',' output json file with all the precision measurements')

FLAGS = flags.FLAGS

IMAGE_PIXELS = 28

def eval_dir(path):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        filenames = glob.glob(path + os.sep + FLAGS.checkpoint_prefix + '*')
        op = {}
        for fname in filenames:
            if 'meta' in fname:
                continue
            steps = int(fname.split(os.sep)[-1].split('-')[-1])
            saver.restore(sess, fname)
            val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
            val_mine = sess.run(accuracy, feed_dict=val_feed)
            op[steps] = float(val_mine)

        f = open(os.path.join(path, 'acc_measures.json'), 'w')
        f.write(json.dumps(op))


def main(unused_argv):
    print('starting main')
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    if FLAGS.download_only:
       sys.exit(0)
    
    global_step = tf.Variable(0, name="global_step", trainable=False)

    ## Variables of the hidden layer
    hid_w = tf.Variable(
        tf.truncated_normal(
            [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
            stddev=1.0 / IMAGE_PIXELS),
        name="hid_w")
    
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

    ## Variables of the softmax layer
    sm_w = tf.Variable(
        tf.truncated_normal(
            [FLAGS.hidden_units, 10],
            stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
        name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    # Ops: located on the worker specified with FLAGS.task_index
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, 10])

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    print ('Processing folder %s' % flags.checkpoint_dir) 

    saver = tf.train.Saver()
    with tf.Session() as sess:
      #ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      filenames = glob.glob(flags.checkpoint_dir + os.sep + FLAGS.checkpoint_prefix + '*')
      op = {}
      for fname in filenames:
          if 'meta' in fname:
              continue
          steps = int(fname.split(os.sep)[-1].split('-')[-1])
          saver.restore(sess, fname)
          val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
          #val_xent = sess.run(cross_entropy, feed_dict=val_feed)
          val_mine = sess.run(accuracy, feed_dict=val_feed)
          op[steps] = float(val_mine)

      f = open(flags.output_json, 'w')
      #print(op)
      f.write(json.dumps(op))
    return 0

if __name__ == "__main__":
  tf.app.run()

