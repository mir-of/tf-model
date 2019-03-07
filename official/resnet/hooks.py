import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.training import basic_session_run_hooks

class DumpingTensorHook(basic_session_run_hooks.LoggingTensorHook):
  def __init__(self, prefixes, output_dir="./probe_output", exclude_keywords=[]):
    self.prefixes = prefixes
    self.output_dir = output_dir
    self.exclude_keywords = exclude_keywords
    super(DumpingTensorHook, self).__init__(tensors=[], every_n_iter=1)

  def has_prefix(self, tensor_name):
    for prefix in self.prefixes:
      if tensor_name.startswith(prefix):
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
    if os.path.isdir(self.output_dir):
      shutil.rmtree(self.output_dir)
    os.makedirs(os.path.join(self.output_dir))
    for k, v in tensor_values.items():
      filepath = os.path.join(self.output_dir, 'iter_{}/{}'.format(self._iter_count, k))
      if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
      np.save(filepath, v)