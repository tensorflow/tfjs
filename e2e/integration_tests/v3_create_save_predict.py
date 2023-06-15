# Copyright 2020 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# This file is 1/3 of the test suites for keras v3: create->save->predict.
# E2E from Keras model -> TFJS model -> Comparsion between two models.
#
# This file does below things:
# - Load Keras models equivalent with models generated by Layers.
# - Load inputs.
# - Make inference and store in local files.
import argparse
import json
import tensorflow as tf
import tempfile
import os
from tensorflow import keras
from tensorflowjs.converters.converter import dispatch_keras_v3_to_tfjs_layers_model_conversion

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

  print(model.get_config())


  save_model_and_random_inputs(model, export_path, model_name)

def save_model_and_random_inputs(model, export_path, model_name):
  os.mkdir(export_path)
  keras_file_path = export_path + '/' + model_name + '.keras'
  model.save(keras_file_path)
  print('export path: ', export_path)
  print('keras path: ', keras_file_path)
  dispatch_keras_v3_to_tfjs_layers_model_conversion(keras_file_path, output_dir=export_path)

  xs = []
  xsData = []
  xsShapes = []
  tensors = model.variables
  t = [i.value() for i in tensors]

  for input_tensor in t:
    input_shape = input_tensor.shape
    xTensor = tf.random.normal(input_shape)
    # xTensor
    xs.append(xTensor)
    xsData.append(xTensor.numpy())
    xsShapes.append(xTensor.shape)

  xs_data = [x.tolist() for x in xsData]
  xs_shape = [list(x) for x in xsShapes]

  xs_shape_path = os.path.join(_tmp_dir, model_name + '.xs-shapes.json')
  xs_data_path = os.path.join(_tmp_dir, model_name + '.xs-data.json')
  with open(xs_data_path, 'w') as f:
    f.write(json.dumps(xs_data))

  with open(xs_shape_path, 'w') as f:
    f.write(json.dumps(xs_shape))



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
