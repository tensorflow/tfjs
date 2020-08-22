# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary that generates saved model artifacts for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow.compat.v2 as tf

def parse_args():
  parser = argparse.ArgumentParser(
      'Generates saved model artifacts for testing.')
  parser.add_argument(
      'output_path',
      type=str,
      help='Model output path.')
  parser.add_argument(
      '--model_type',
      type=str,
      required=True,
      choices=set(['tf_keras_h5', 'tf_saved_model']),
      help='Model format to generate.')
  return parser.parse_known_args()


def main(_):

  if args.model_type == 'tf_keras_h5':
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(5, activation='relu', input_shape=(8,)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.save(os.path.join(args.output_path))
  elif args.model_type == 'tf_saved_model':
    class TimesThreePlusOne(tf.Module):

      @tf.function(input_signature=[
          tf.TensorSpec(shape=None, dtype=tf.float32)])
      def compute(self, x):
        return x * 3.0 + 1.0

    tf.saved_model.save(TimesThreePlusOne(), args.output_path)
  else:
    raise ValueError('Unrecognized model type: %s' % args.model_type)


if __name__ == '__main__':
  args, unparsed = parse_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
