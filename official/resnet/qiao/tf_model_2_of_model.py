import os
import sys
import argparse
import numpy as np

def parse_arguements(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('--tf_model_dir', type=str, help='',
                      default='/dataset/models/tf-facenet/20180402-114759-npy')
  parser.add_argument('--of_model_list', type=str, help='',
                      default='/home/qiaojing/dev/facenet_config/facenet_train/namelist.txt')
  parser.add_argument('--output_dir', type=str, help='', default='./facenet_of_model')

  return parser.parse_args(argv)

def save2file(numpy_array, of_model_path, name):
  tf_model = np.load(numpy_array)
  if name == 'weight' and ('Logits' in numpy_array):
    tf_model = tf_model.transpose(1, 0)
  elif name == 'weight' and ('Bottleneck' in numpy_array):
    tf_model = tf_model.transpose(1, 0)
  elif name == 'weight' and ('Bottleneck' not in numpy_array):
    tf_model = tf_model.transpose(3,2,0,1)
  if not os.path.exists(of_model_path): os.makedirs(of_model_path)
  with open(os.path.join(of_model_path, name), 'wb') as f:
    f.write(tf_model.tobytes())

def convert(of_model_path, tf_model_path):
  weight_path = os.path.join(tf_model_path, 'weights.npy')
  bias_path = os.path.join(tf_model_path, 'biases.npy')
  beta_path = os.path.join(tf_model_path, 'beta.npy')
  mv_path = os.path.join(tf_model_path, 'moving_variance.npy')
  mm_path = os.path.join(tf_model_path, 'moving_mean.npy')

  if os.path.exists(weight_path):
    save2file(weight_path, of_model_path, 'weight')
  if os.path.exists(bias_path):
    save2file(bias_path, of_model_path, 'bias')
  if os.path.exists(beta_path):
    save2file(beta_path, of_model_path, 'beta')
  if os.path.exists(mv_path):
    save2file(mv_path, of_model_path, 'moving_variance')
  if os.path.exists(mm_path):
    save2file(mm_path, of_model_path, 'moving_mean')


def main(args):
  if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
  with open(args.of_model_list, 'rb')  as f:
    for l in f.readlines():
      of_model_path = os.path.join(args.output_dir, l.strip())
      tf_model_path = os.path.join(args.tf_model_dir, l.strip().replace('-', '/'))

      convert(of_model_path, tf_model_path)


if __name__ == '__main__':
  main(parse_arguements(sys.argv[1:]))
