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
"""Unit tests for prelu op fusing."""

import os
import shutil
import tempfile

import tensorflow as tf
from tensorflow.core.protobuf import config_pb2

from tensorflowjs.converters import fuse_prelu
from tensorflowjs.converters import tf_saved_model_conversion_v2

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

    graph_def = execute_model.get_concrete_function(
        input_tensor).graph.as_graph_def()
    optimized_graph_def = fuse_prelu.fuse_ops_for_prelu(graph_def)

    prelu_op_count = 0
    for node in optimized_graph_def.node:
      self.assertNotEqual("Relu", node.op)
      if node.op == 'Prelu':
        prelu_op_count += 1
    self.assertEqual(prelu_op_count, 2)

  def testFusePreluWithConv2d(self):
    layers = [
        tf.keras.layers.Conv2D(
            16, [3, 3], padding='same', use_bias=True),
        tf.keras.layers.PReLU()
    ]
    model = tf.keras.Sequential(layers)
    tf.keras.backend.set_learning_phase(0)
    input_tensor = tf.constant([1.0, 1.0], shape=[1, 2, 1, 1])

    @tf.function
    def execute_model(tensor):
      return model(tensor)

    graph = execute_model.get_concrete_function(
        input_tensor).graph
    graph_def = graph.as_graph_def()

    for node in graph_def.node:
      if node.op == 'Conv2D':
        node.device = "/CPU:0"

    config = config_pb2.ConfigProto()
    rewriter_config = config.graph_options.rewrite_options
    rewriter_config.optimizers[:] = [
        'pruning', 'constfold', 'arithmetic', 'dependency', 'pruning', 'remap',
        'constfold', 'arithmetic', 'dependency'
    ]

    for output in ['Identity']:
      graph.add_to_collection('train_op', graph.get_operation_by_name(output))

    graph_def = tf_saved_model_conversion_v2._run_grappler(
        config, graph_def, graph)

    graph_def = fuse_prelu.fuse_ops_for_prelu(graph_def)

    optimized_graph_def = fuse_prelu.fuse_prelu_with_fused_conv2d(graph_def)

    conv2d_op = None
    for node in optimized_graph_def.node:
      self.assertNotEqual("Prelu", node.op)
      if node.op == '_FusedConv2D':
        conv2d_op = node
    self.assertNotEqual(conv2d_op, None)
    self.assertEqual(conv2d_op.attr['fused_ops'].list.s, [b'BiasAdd', b'Prelu'])
    self.assertEqual(conv2d_op.attr['num_args'].i, 2)

if __name__ == '__main__':
  tf.test.main()
