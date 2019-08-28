# Copyright 2018 Google LLC
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
"""Unit tests for batch norm folding."""

import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflowjs.converters import fuse_prelu

class FusePreluTest(tf.test.TestCase):
  def setUp(self):
    super(FusePreluTest, self).setUp()
    self._tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(FusePreluTest, self).tearDown()

  def testFusePrelu(self):
    layers = [
        tf.keras.layers.PReLU(
            alpha_initializer=tf.initializers.constant(0.25)),
        tf.keras.layers.PReLU(
            alpha_initializer=tf.initializers.constant(0.25))
    ]
    model = tf.keras.Sequential(layers)
    tf.keras.backend.set_learning_phase(0)
    input_tensor = tf.constant([1.0, 1.0])

    @tf.function
    def execute_model(tensor):
      return model(tensor)

    graph_def = execute_model.get_concrete_function(input_tensor).graph.as_graph_def()
    optimized_graph_def = fuse_prelu.fuse_ops_for_prelu(graph_def)
    
    prelu_op_count = 0
    for node in optimized_graph_def.node:
      self.assertNotEqual("Relu", node.op)
      if node.op == 'Prelu':
          prelu_op_count += 1
    self.assertEqual(prelu_op_count, 2)

if __name__ == '__main__':
  tf.test.main()
