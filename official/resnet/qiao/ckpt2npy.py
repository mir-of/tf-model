import tensorflow as tf
import numpy as np
import os

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
  new_saver = tf.train.import_meta_graph('/home/qiaojing/tmp/resnet_model/tf_ckpt/model.ckpt-0.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('/home/qiaojing/tmp/resnet_model/tf_ckpt'))

init_vars = tf.train.list_variables('/home/qiaojing/tmp/resnet_model/tf_ckpt')

names=[]
arrays=[]
for name, shape in init_vars:
  print("Loading {} with shape {}".format(name, shape))
  array = tf.train.load_variable('/home/qiaojing/tmp/resnet_model/tf_ckpt', name)
  print("Numpy array shape {}".format(array.shape))
  save_path = os.path.join('./resnet_npy_out', name)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  np.save(save_path, array)
  names.append(name)
  arrays.append(array)

