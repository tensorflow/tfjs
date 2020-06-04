# Copyright 2020 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# This file is 2/3 of the test suites for CUJ: create->save->predict.
#
# This file does below things:
# - Load Keras models equivalent with models generated by Layers.
# - Load inputs.
# - Make inference and store in local files.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

if os.environ['TFJS2KERAS_TEST_USING_TF_KERAS'] == '1':
  print('Using tensorflow.keras.')
  from tensorflow import keras
else:
  print('Using keras-team/keras.')
  import keras

curr_dir = os.path.dirname(os.path.realpath(__file__))
_tmp_dir = os.path.join(curr_dir, 'create_save_predict_data')

def _load_predict_save(model_path):
  """Load a Keras Model from artifacts generated by tensorflow.js and inputs.
     Make inference with the model and inputs.
     Write outputs to file.

  Args:
    model_path: Path to the model JSON file.
  """
  xs_shape_path = os.path.join(
      _tmp_dir, model_path + '.xs-shapes.json')
  xs_data_path = os.path.join(
      _tmp_dir, model_path + '.xs-data.json')
  with open(xs_shape_path, 'rt') as f:
    xs_shapes = json.load(f)
  with open(xs_data_path, 'rt') as f:
    xs_values = json.load(f)
  xs = [np.array(value, dtype=np.float32).reshape(shape)
        for value, shape in zip(xs_values, xs_shapes)]
  if len(xs) == 1:
    xs = xs[0]

  session = tf.Session() if hasattr(tf, 'Session') else tf.compat.v1.Session()
  with tf.Graph().as_default(), session:
    model_json_path = os.path.join(_tmp_dir, model_path, 'model.json')
    print('Loading model from path %s' % model_json_path)
    model = tfjs.converters.load_keras_model(model_json_path)
    ys = model.predict(xs)

    ys_data = None
    ys_shape = None

    if isinstance(ys, list):
      ys_data = [y.tolist() for y in ys]
      ys_shape = [list(y.shape) for y in ys]
    else:
      ys_data = ys.tolist()
      ys_shape = [list(ys.shape)]

    ys_data_path = os.path.join(
      _tmp_dir, model_path + '.ys-data.json')
    ys_shape_path = os.path.join(
      _tmp_dir, model_path + '.ys-shapes.json')
    with open(ys_data_path, 'w') as f:
      f.write(json.dumps(ys_data))
    with open(ys_shape_path, 'w') as f:
      f.write(json.dumps(ys_shape))

def main():
  _load_predict_save('mlp')
  _load_predict_save('cnn')
  _load_predict_save('depthwise_cnn')
  _load_predict_save('simple_rnn')
  _load_predict_save('gru')
  _load_predict_save('bidirectional_lstm')
  _load_predict_save('time_distributed_lstm')
  _load_predict_save('one_dimensional')
  _load_predict_save('functional_merge')

if __name__ == '__main__':
  main()
