import tensorflow as tf
import numpy as np
import os


ckpt_dir = "/home/qiaojing/tmp/resnet_model/png_290_model/tf_model"
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
  new_saver = tf.train.import_meta_graph(os.path.join(ckpt_dir,'model.ckpt-1.meta'))
  new_saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

init_vars = tf.train.list_variables(ckpt_dir)

names=[]
arrays=[]
for name, shape in init_vars:
  print("Loading {} with shape {}".format(name, shape))
  array = tf.train.load_variable(ckpt_dir, name)
  print("Numpy array shape {}".format(array.shape))
  save_path = os.path.join('./resnet_npy_out', name)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  np.save(save_path, array)
  names.append(name)
  arrays.append(array)

