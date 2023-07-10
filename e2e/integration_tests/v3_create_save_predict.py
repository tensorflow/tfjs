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
# - Build Keras v3 model
# - Save the model to local files.
# - Convert and save the model to TFJS layers format.
# - Extract data and shape from model and store in local files.
import argparse
import json
import tensorflow as tf
import tempfile
import os
from tensorflow import keras
from tensorflowjs.converters.converter import dispatch_keras_keras_to_tfjs_layers_model_conversion

print('tf-version: ', tf.version.VERSION)
curr_dir = os.path.dirname(os.path.realpath(__file__))
_tmp_dir = os.path.join(curr_dir, 'v3_create_save_predict_data')

def _export_mlp_model(export_path, model_name):
  model = keras.Sequential()
  d = keras.layers.Dense(100, activation='relu')
  model.add(d)
  model.add(keras.layers.Dense(50, activation='elu'))
  model.add(keras.layers.Dense(24))
  model.add(keras.layers.Activation(activation='elu'))
  model.add(keras.layers.Dense(8, activation='softmax'))
  model.build(input_shape=[1, 200])



  save_model_and_random_inputs(model, export_path, model_name)

def save_model_and_random_inputs(model, export_path, model_name):
  os.mkdir(export_path)
  keras_file_path = export_path + '/' + model_name + '.keras'
  model.save(keras_file_path)
  print('export path: ', export_path)
  print('keras path: ', keras_file_path)
  dispatch_keras_keras_to_tfjs_layers_model_conversion(keras_file_path, output_dir=export_path)

  xs = []
  xsData = []
  xsShapes = []
  tensors = model.variables
  mInput = model.inputs
  print(mInput)

  for input_tensor in mInput:
    input_shape = input_tensor.shape
    print(input_shape)
    xTensor = tf.random.normal(input_shape)
    # xTensor
    xs.append(xTensor)
    xsData.append(xTensor.numpy()[0])
    xsShapes.append(xTensor.shape)
    print('xTensor numpy: ',xTensor.numpy()[0])
    print('xTensor shape: ', xTensor.shape)
  print('xsData ===> ', xsData)
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
  _export_mlp_model(os.path.join(_tmp_dir, 'mlp'), 'mlp')

if __name__ == '__main__':
  main()
