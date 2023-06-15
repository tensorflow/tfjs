import argparse
import tensorflow as tf
import tempfile
import os
from tensorflow import keras

curr_dir = os.path.dirname(os.path.realpath(__file__))
_tmp_dir = os.path.join(curr_dir, 'test_data_dir')

def _export_mlp_model(export_path, model_name):
  model = keras.Sequential()
  model.add(keras.layers.Dense(100,activation='relu'))
  model.add(keras.layers.Dense(50, activation='elu'))
  model.add(keras.layers.Dense(24))
  model.add(keras.layers.Activation(activation='elu'))
  model.add(keras.layers.Dense(8, activation='softmax'))
  model.build([1, 10])
  # input_layers = model.input
  # input_tensors = [input_layer.output for input_layer in input_layers]
  # for input_tensor in input_tensors:
  #   input_shape = input_tensor.shape
  #   input_shape[0] = 1
  #   if input_shape.

  save_model_and_random_inputs(model, export_path, model_name)

def save_model_and_random_inputs(model, export_path, model_name):
  os.mkdir(export_path)
  model.save(export_path + '/' + model_name + '.keras')
  print(export_path + '/' + model_name + '.keras')

  xs = []
  xsData = []
  xsShapes = []
  tensors = model.variables
  # print("inputs: ", tensors)
  t = [i.value() for i in tensors]
  # print(t)
  # return
  # input_tensors = [input_layer.output for input_layer in input_layers]
  for input_tensor in t:
    input_shape = input_tensor.shape
    print('input shape: ', input_shape)
    xTensor = tf.random.normal(input_shape)
    # xTensor
    print('xTensor: ', xTensor)
    print('xTensor data: ', xTensor.numpy())
    # input_shape[0] = 1
    # xTensor = tf.random_normal_initializer(input_shape)
    # print(xTensor)
    # print(xTensor.arraySync())

    # xs.append(xTensor)
    # xsData.append(xTensor.arraySync())






def main():
  print('tensorflow version: ', tf.__version__)

  parser = argparse.ArgumentParser(description='Create a keras model in python.')
  parser.add_argument('--test_data_dir', help='Input a directory name', default='test_data_dir')
  args = parser.parse_args()
  temp_dir = args.test_data_dir
  # if os.path.exists(_tmp_dir):
  #   os.rmdir(_tmp_dir)
  #   os.mkdir(_tmp_dir)
  # else:
  os.mkdir(_tmp_dir)
  _export_mlp_model(os.path.join(_tmp_dir, 'mlp'), 'mlp')

if __name__ == '__main__':
  main()
