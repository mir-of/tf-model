import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.training import basic_session_run_hooks

class DumpingTensorHook(basic_session_run_hooks.LoggingTensorHook):
  def __init__(self, include_keywords, output_dir='probe_output'):
    self._include_keywords = include_keywords
    self._output_dir = output_dir
    super(self, DumpingTensorHook).__init__(tensors=[], every_n_iter=1)

  def has_keyword(self, tensor_name):
    for k in self._include_keywords:
      if k in tensor_name: return True
    return False

  def begin(self):
    def get_outputs():
      for op in tf.get_default_graph().get_operations():
        for tensor in op.outputs:
          if self.has_keyword(tensor.name):
            yield tensor.name

    self._tensors = {item: item for item in get_outputs()}
    super(self, DumpingTensorHook).begin()

  def _log_tensors(self, tensor_values):
    # tensor_values is a map
    if os.path.isdir(self._output_dir):
      shutil.rmtree(self._output_dir)
    os.makedirs(os.path.join(self._output_dir))
    for k, v in tensor_values.items():
      filepath = os.path.join(self._output_dir, 'iter_{}/{}'.format(self._iter_count, k))
      if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
      np.save(filepath, v)