import os
import numpy as np
import tensorflow as tf

@tf.RegisterGradient('OverrideGradientWithIdentity')
def ApplyIdentity(op, dy):
  with tf.name_scope("probe_diff"):
    dx = tf.identity(dy)
    return [dx]


def add_prob(tensor, name):
  with tf.name_scope("probe"):
    with tf.get_default_graph().gradient_override_map(
                                {'Identity': 'OverrideGradientWithIdentity'}):
        return tf.identity(tensor, name=name)


def tensor_hook(tensors, output_dir):
  for (tensor, array) in tensors:
    filename = tensor.name
    filepath = os.path.join(output_dir, os.path.dirname(filename))
    if not os.path.exists(filepath):
      os.makedirs(filepath)
    filename = os.path.join(filepath, os.path.basename(filename))
    np.save(filename, array)