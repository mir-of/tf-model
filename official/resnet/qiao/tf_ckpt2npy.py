#############################################################################################
# Read tf checkpoint and saved as numpy array.
# usage: 
#   python 
#############################################################################################
import os
import sys
import argparse
import csv
import numpy as np
import tensorflow as tf

def parse_arguement(argv):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--ckpt_dir', type=str, 
    help='', default='/dataset/models/tf-facenet/20180402-114759')
  parser.add_argument('--ckpt_file', type=str, 
    help='', default='model-20180402-114759.ckpt-275')
  parser.add_argument('--meta_file', type=str, 
    help='', default='model-20180402-114759.meta')
  parser.add_argument('--output_file', type=str, 
    help='', default='./output')
  
  return parser.parse_args(argv)

def main(args):
  checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_file)
  reader = tf.train.NewCheckpointReader(checkpoint_path)
  var_map = reader.get_variable_to_shape_map()
  # print to file
  var_list = [(key, var_map[key]) for key in var_map]
  with open('model_names.csv', 'wb') as f:
    wr = csv.writer(f, dialect='excel')
    wr.writerows(var_list)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  saver = tf.train.import_meta_graph(os.path.join(args.ckpt_dir, args.meta_file))
  saver.restore(sess, checkpoint_path)
  
  for var_name in var_map:
    tensor_name = var_name + ':0' 
    tensor = sess.run(tf.get_default_graph().get_tensor_by_name(tensor_name))
    path = os.path.join(args.output_file, os.path.dirname(var_name))
    if not os.path.exists(path):
      os.makedirs(path)
    np.save(os.path.join(path, os.path.basename(var_name)), tensor) 

if __name__ == '__main__':
  main(parse_arguement(sys.argv[1:]))
