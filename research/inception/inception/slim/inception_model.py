# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inception-v3 expressed in TensorFlow-Slim.

  Usage:

  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      # Force all Variables to reside on the CPU.
      with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
        logits, endpoints = slim.inception.inception_v3(
            images,
            dropout_keep_prob=0.8,
            num_classes=num_classes,
            is_training=for_training,
            restore_logits=restore_logits,
            scope=scope)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes

import debug

def inception_v3(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 is_training=True,
                 restore_logits=True,
                 scope=''):
  """Latest Inception from http://arxiv.org/abs/1512.00567.

    "Rethinking the Inception Architecture for Computer Vision"

    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
    Zbigniew Wojna

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    dropout_keep_prob: dropout keep_prob.
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: Optional scope for name_scope.

  Returns:
    a list containing 'logits', 'aux_logits' Tensors.
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.name_scope(scope, 'inception_v3', [inputs]):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='VALID'):

        inputs = debug.add_prob(inputs, name='input_image')
        print('input_image shape: {}'.format(inputs.get_shape()))

        # 299 x 299 x 3
        inputs = ops.conv2d(inputs, 32, [3, 3], stride=2,
                                         scope='conv0')
        end_points['conv0'] = inputs
        inputs = debug.add_prob(inputs, name='conv0')
        print('conv0 shape: {}'.format(inputs.get_shape()))

        # 149 x 149 x 32
        inputs = ops.conv2d(inputs, 32, [3, 3],
                                         scope='conv1')
        end_points['conv1'] = inputs
        inputs = debug.add_prob(inputs, name='conv1')
        print('conv1 shape: {}'.format(inputs.get_shape()))

        # 147 x 147 x 32
        inputs = ops.conv2d(inputs, 64, [3, 3],
                                         padding='SAME', scope='conv2')
        end_points['conv2'] = inputs
        inputs = debug.add_prob(inputs, name='conv2')
        print('conv2 shape: {}'.format(inputs.get_shape()))

        # 147 x 147 x 64
        inputs = ops.max_pool(inputs, [3, 3],
                                           stride=2, scope='pool1')
        end_points['pool1'] = inputs
        inputs = debug.add_prob(inputs, name='pool1')
        print('pool1 shape: {}'.format(inputs.get_shape()))

        # 73 x 73 x 64
        inputs = ops.conv2d(inputs, 80, [1, 1],
                                         scope='conv3')
        end_points['conv3'] = inputs
        inputs = debug.add_prob(inputs, name='conv3')
        print('conv3 shape: {}'.format(inputs.get_shape()))

        # 73 x 73 x 80.
        inputs = ops.conv2d(inputs, 192, [3, 3],
                                         scope='conv4')
        end_points['conv4'] = inputs
        inputs = debug.add_prob(inputs, name='conv4')
        print('conv4 shape: {}'.format(inputs.get_shape()))

        # 71 x 71 x 192.
        inputs = ops.max_pool(inputs, [3, 3],
                                           stride=2, scope='pool2')
        end_points['pool2'] = inputs
        inputs = debug.add_prob(inputs, name='pool2')
        print('pool2 shape: {}'.format(inputs.get_shape()))

        # 35 x 35 x 192.
        net = inputs

      # Inception blocks
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='SAME'):
        # mixed: 35 x 35 x 256.
        with tf.variable_scope('mixed_0'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1])

          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1])
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 32, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_0'] = net

        net = debug.add_prob(net, name='mixed_0')
        print('mixed_0 shape: {}'.format(net.get_shape()))

        # mixed_1: 35 x 35 x 288.
        with tf.variable_scope('mixed_1'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1])
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_1'] = net

        net = debug.add_prob(net, name='mixed_1')
        print('mixed_1 shape: {}'.format(net.get_shape()))

        # mixed_2: 35 x 35 x 288.
        with tf.variable_scope('mixed_2'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1])
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_2'] = net

        net = debug.add_prob(net, name='mixed_2')
        print('mixed_2 shape: {}'.format(net.get_shape()))

        # mixed_3: 17 x 17 x 768.
        with tf.variable_scope('mixed_3'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],
                                      stride=2, padding='VALID')
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
          net = tf.concat(axis=3, values=[branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_3'] = net

        net = debug.add_prob(net, name='mixed_3')
        print('mixed_3 shape: {}'.format(net.get_shape()))

        # mixed4: 17 x 17 x 768.
        with tf.variable_scope('mixed_4'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 128, [1, 1])
            branch7x7 = ops.conv2d(branch7x7, 128, [1, 7])
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 128, [1, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [1, 7])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_4'] = net

        net = debug.add_prob(net, name='mixed_4')
        print('mixed_4 shape: {}'.format(net.get_shape()))

        # mixed_5: 17 x 17 x 768.
        with tf.variable_scope('mixed_5'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 160, [1, 1])
            branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 160, [1, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_5'] = net

        net = debug.add_prob(net, name='mixed_5')
        print('mixed_5 shape: {}'.format(net.get_shape()))

        # mixed_6: 17 x 17 x 768.
        with tf.variable_scope('mixed_6'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 160, [1, 1])
            branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 160, [1, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_6'] = net

        net = debug.add_prob(net, name='mixed_6')
        print('mixed_6 shape: {}'.format(net.get_shape()))

        # mixed_7: 17 x 17 x 768.
        with tf.variable_scope('mixed_7'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 192, [1, 1])
            branch7x7 = ops.conv2d(branch7x7, 192, [1, 7])
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 192, [1, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_7'] = net

        net = debug.add_prob(net, name='mixed_7')
        print('mixed_7 shape: {}'.format(net.get_shape()))

        # Auxiliary Head logits
        # aux_logits = tf.identity(end_points['mixed_7'])
        # with tf.variable_scope('aux_logits'):
        #   aux_logits = ops.avg_pool(aux_logits, [5, 5], stride=3,
        #                             padding='VALID')
        #   aux_logits = ops.conv2d(aux_logits, 128, [1, 1], scope='proj')
        #   # Shape of feature map before the final layer.
        #   shape = aux_logits.get_shape()
        #   aux_logits = ops.conv2d(aux_logits, 768, shape[1:3], stddev=0.01,
        #                           padding='VALID')
        #   aux_logits = ops.flatten(aux_logits)
        #   aux_logits = ops.fc(aux_logits, num_classes, activation=None,
        #                       stddev=0.001, restore=restore_logits)
        #   end_points['aux_logits'] = aux_logits

        # mixed_8: 8 x 8 x 1280.
        # Note that the scope below is not changed to not void previous
        # checkpoints.
        # (TODO) Fix the scope when appropriate.
        with tf.variable_scope('mixed_8'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 192, [1, 1])
            branch3x3 = ops.conv2d(branch3x3, 320, [3, 3], stride=2,
                                   padding='VALID')
          with tf.variable_scope('branch7x7x3'):
            branch7x7x3 = ops.conv2d(net, 192, [1, 1])
            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [1, 7])
            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [7, 1])
            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [3, 3],
                                     stride=2, padding='VALID')
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
          net = tf.concat(axis=3, values=[branch3x3, branch7x7x3, branch_pool])
          end_points['mixed_8'] = net

        net = debug.add_prob(net, name='mixed_8')
        print('mixed_8 shape: {}'.format(net.get_shape()))

        # mixed_9: 8 x 8 x 2048.
        with tf.variable_scope('mixed_9'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 320, [1, 1])
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [1, 1])
            branch3x3 = tf.concat(axis=3, values=[ops.conv2d(branch3x3, 384, [1, 3]),
                                                  ops.conv2d(branch3x3, 384, [3, 1])])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 448, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
            branch3x3dbl = tf.concat(axis=3, values=[ops.conv2d(branch3x3dbl, 384, [1, 3]),
                                                     ops.conv2d(branch3x3dbl, 384, [3, 1])])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_9'] = net

        net = debug.add_prob(net, name='mixed_9')
        print('mixed_9 shape: {}'.format(net.get_shape()))

        # mixed_10: 8 x 8 x 2048.
        with tf.variable_scope('mixed_10'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 320, [1, 1])
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [1, 1])
            branch3x3 = tf.concat(axis=3, values=[ops.conv2d(branch3x3, 384, [1, 3]),
                                                  ops.conv2d(branch3x3, 384, [3, 1])])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 448, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
            branch3x3dbl = tf.concat(axis=3, values=[ops.conv2d(branch3x3dbl, 384, [1, 3]),
                                                     ops.conv2d(branch3x3dbl, 384, [3, 1])])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_10'] = net

        net = debug.add_prob(net, name='mixed_10')
        print('mixed_10 shape: {}'.format(net.get_shape()))

        # Final pooling and prediction
        with tf.variable_scope('logits'):
          shape = net.get_shape()
          net = ops.avg_pool(net, shape[1:3], padding='VALID', scope='pool')
          # 1 x 1 x 2048
          net = ops.dropout(net, dropout_keep_prob, scope='dropout')
          net = ops.flatten(net, scope='fc1')

          net = debug.add_prob(net, name='fc1')
          print('fc1 shape: {}'.format(net.get_shape()))

          # 2048
          logits = ops.fc(net, num_classes, activation=None, scope='fc2',
                          restore=restore_logits)

          net = debug.add_prob(net, name='fc2')
          print('fc2 shape: {}'.format(net.get_shape()))

          # 1000
          end_points['logits'] = logits
          end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
      return logits, end_points


def inception_v3_parameters(weight_decay=0.00004, stddev=0.1,
                            batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
  """Yields the scope with the default parameters for inception_v3.

  Args:
    weight_decay: the weight decay for weights variables.
    stddev: standard deviation of the truncated guassian weight distribution.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Yields:
    a arg_scope with the parameters needed for inception_v3.
  """
  # Set weight_decay for weights in Conv and FC layers.
  with scopes.arg_scope([ops.conv2d, ops.fc],
                        weight_decay=weight_decay):
    # Set stddev, activation and parameters for batch_norm.
    with scopes.arg_scope([ops.conv2d],
                          stddev=stddev,
                          activation=tf.nn.relu,
                          batch_norm_params={
                              'decay': batch_norm_decay,
                              'epsilon': batch_norm_epsilon}) as arg_scope:
      yield arg_scope
