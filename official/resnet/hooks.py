import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.training import basic_session_run_hooks

import csv

class DumpingTensorHook(basic_session_run_hooks.LoggingTensorHook):
  def __init__(self, cycle_index, prefixes, output_dir="./probe_output", exclude_keywords=[]):
    self.cycle_index = cycle_index
    self.prefixes = prefixes
    self.output_dir = output_dir
    self.exclude_keywords = exclude_keywords
    super(DumpingTensorHook, self).__init__(tensors=[], every_n_iter=1)

  def has_prefix(self, tensor_name):
    for prefix in self.prefixes:
      #  if tensor_name.startswith(prefix):
      if prefix in tensor_name:
        return True
    return False

  def has_exclude_keywords(self, tensor_name):
    for keyword in self.exclude_keywords:
      if keyword in tensor_name:
        return True
    return False

  def begin(self):

    def get_outputs():
      for op in tf.get_default_graph().get_operations():
        # tensor types to exclude
        for tensor in op.outputs:
          if tensor.dtype in {tf.variant, tf.string}:
            continue
          if self.has_prefix(tensor.name) and not self.has_exclude_keywords(tensor.name):
            yield tensor.name

    self._tensors = {item: item for item in get_outputs()}
    super(DumpingTensorHook, self).begin()
    
  def _log_tensors(self, tensor_values):
    # tensor_values is a map
    if self.cycle_index > 0:
      return
    if os.path.isdir(self.output_dir):
      shutil.rmtree(self.output_dir)
    os.makedirs(os.path.join(self.output_dir))

    if self.cycle_index == 0:
      with open(os.path.join(self.output_dir, 'tf_resnet_tensors.csv'), 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for k, v in tensor_values.items():
          filepath = os.path.join(self.output_dir, 'iter_{}/{}'.format(self.cycle_index, k))
          if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
          np.save(filepath, v)
          writer.writerow(['{}.npy'.format(filepath), v.shape])