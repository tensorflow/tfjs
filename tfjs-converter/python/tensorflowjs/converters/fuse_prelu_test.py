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

import tensorflow.compat.v2 as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import tracking

from tensorflowjs.converters import fuse_depthwise_conv2d
from tensorflowjs.converters import fuse_prelu
from tensorflowjs.converters import tf_saved_model_conversion_v2
from tensorflowjs.converters import graph_rewrite_util

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

    graph = tf_saved_model_conversion_v2._freeze_saved_model_v2(
        execute_model.get_concrete_function(input_tensor))
    graph_def = graph.as_graph_def()
    for node in graph_def.node:
      if node.op == 'Conv2D':
        node.device = "/CPU:0"

    config = config_pb2.ConfigProto()
    rewriter_config = config.graph_options.rewrite_options
    rewriter_config.optimizers[:] = [
        'pruning', 'constfold', 'arithmetic', 'dependency', 'pruning',
        'remap', 'constfold', 'arithmetic', 'dependency'
    ]

    for output in ['Identity']:
      graph.add_to_collection('train_op', graph.get_operation_by_name(output))

    signature = meta_graph_pb2.SignatureDef()
    graph_def = tf_saved_model_conversion_v2._run_grappler(
        config, graph_def, graph, signature)

    optimized_graph_def = fuse_prelu.fuse_ops_for_prelu(graph_def)

    prelu_op_count = 0
    value = None
    for node in optimized_graph_def.node:
      self.assertNotEqual("Relu", node.op)
      if node.op == 'Prelu':
        prelu_op_count += 1
      if node.op == 'Const':
        value = graph_rewrite_util.values_from_const(node)
    self.assertEqual(prelu_op_count, 2)
    self.assertEqual(value, [0.25])

  def testFusePreluWithConv2d(self):
    layers = [
        tf.keras.layers.Conv2D(
            16, [3, 3], padding='same', use_bias=True,
            bias_initializer=tf.initializers.constant(0.25)),
        tf.keras.layers.PReLU()
    ]
    model = tf.keras.Sequential(layers)
    tf.keras.backend.set_learning_phase(0)
    input_tensor = tf.constant([1.0, 1.0], shape=[1, 2, 1, 1])

    @tf.function
    def execute_model(tensor):
      return model(tensor)

    graph = tf_saved_model_conversion_v2._freeze_saved_model_v2(
        execute_model.get_concrete_function(input_tensor))
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

    signature = meta_graph_pb2.SignatureDef()
    graph_def = tf_saved_model_conversion_v2._run_grappler(
        config, graph_def, graph, signature)
    graph_def = fuse_prelu.fuse_ops_for_prelu(graph_def)

    optimized_graph_def = fuse_prelu.fuse_prelu_with_fused_conv2d_or_matmul(
        graph_def)

    conv2d_op = None
    for node in optimized_graph_def.node:
      self.assertNotEqual("Prelu", node.op)
      if node.op == '_FusedConv2D':
        conv2d_op = node
    self.assertNotEqual(conv2d_op, None)
    self.assertEqual(conv2d_op.attr['fused_ops'].list.s, [b'BiasAdd', b'Prelu'])
    self.assertEqual(conv2d_op.attr['num_args'].i, 2)

  def testFusePreluWithMatMul(self):
    layers = [
        tf.keras.layers.Dense(
            2, use_bias=True,
            kernel_initializer=tf.initializers.constant(0.25),
            bias_initializer=tf.initializers.constant(0.25)),
        tf.keras.layers.PReLU()
    ]
    model = tf.keras.Sequential(layers)
    tf.keras.backend.set_learning_phase(0)
    input_tensor = tf.constant([1.0, 1.0], shape=[1, 2])

    @tf.function
    def execute_model(tensor):
      return model(tensor)

    graph = tf_saved_model_conversion_v2._freeze_saved_model_v2(
        execute_model.get_concrete_function(input_tensor))
    graph_def = graph.as_graph_def()
    for node in graph_def.node:
      if node.op == 'MatMul':
        node.device = "/CPU:0"

    config = config_pb2.ConfigProto()
    rewriter_config = config.graph_options.rewrite_options
    rewriter_config.optimizers[:] = [
        'pruning', 'constfold', 'arithmetic', 'dependency', 'pruning', 'remap',
        'constfold', 'arithmetic', 'dependency'
    ]

    for output in ['Identity']:
      graph.add_to_collection('train_op', graph.get_operation_by_name(output))

    signature = meta_graph_pb2.SignatureDef()
    graph_def = tf_saved_model_conversion_v2._run_grappler(
        config, graph_def, graph, signature)
    graph_def = fuse_prelu.fuse_ops_for_prelu(graph_def)
    optimized_graph_def = fuse_prelu.fuse_prelu_with_fused_conv2d_or_matmul(
        graph_def)

    matmul_op = None
    for node in optimized_graph_def.node:
      self.assertNotEqual("Prelu", node.op)
      if node.op == '_FusedMatMul':
        matmul_op = node
    self.assertNotEqual(matmul_op, None)
    self.assertEqual(matmul_op.attr['fused_ops'].list.s, [b'BiasAdd', b'Prelu'])
    self.assertEqual(matmul_op.attr['num_args'].i, 2)

  def testFusePreluWithDepthwiseConv2d(self):
    layers = [
        tf.keras.layers.DepthwiseConv2D(
            1, bias_initializer=tf.initializers.constant(0.25)),
        tf.keras.layers.PReLU()
    ]
    model = tf.keras.Sequential(layers)
    tf.keras.backend.set_learning_phase(0)
    input_tensor = tf.constant([1.0, 1.0], shape=[1, 2, 1, 1])

    @tf.function
    def execute_model(tensor):
      return model(tensor)

    graph = tf_saved_model_conversion_v2._freeze_saved_model_v2(
        execute_model.get_concrete_function(input_tensor))
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

    signature = meta_graph_pb2.SignatureDef()
    graph_def = tf_saved_model_conversion_v2._run_grappler(
        config, graph_def, graph, signature)
    graph_def = fuse_prelu.fuse_ops_for_prelu(graph_def)
    graph_def = fuse_depthwise_conv2d.fuse_depthwise_conv2d(graph_def)

    optimized_graph_def = fuse_prelu.fuse_prelu_with_fused_conv2d_or_matmul(
        graph_def)

    conv2d_op = None
    for node in optimized_graph_def.node:
      self.assertNotEqual("Prelu", node.op)
      if node.op == 'FusedDepthwiseConv2dNative':
        conv2d_op = node
    self.assertNotEqual(conv2d_op, None)
    self.assertEqual(conv2d_op.attr['fused_ops'].list.s, [b'BiasAdd', b'Prelu'])
    self.assertEqual(conv2d_op.attr['num_args'].i, 2)

  def testNonPreluPattern(self):
    """Test a basic model with functions to make sure functions are inlined."""
    input_data = constant_op.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)

    root.f = def_function.function(lambda x: tf.nn.relu(root.v1) + root.v2 * 2.0)
    to_save = root.f.get_concrete_function(input_data)
    graph = tf_saved_model_conversion_v2._freeze_saved_model_v2(
        root.f.get_concrete_function(input_data))
    graph_def = graph.as_graph_def()
    graph_def = fuse_prelu.fuse_ops_for_prelu(graph_def)
    const_op = None
    for node in graph_def.node:
      self.assertNotEqual("Prelu", node.op)
      if node.op == 'Const':
        const_op = node
    self.assertNotEqual(const_op, None)
    self.assertEqual(const_op.attr['value'].tensor.float_val, [2.0])

if __name__ == '__main__':
  tf.test.main()
