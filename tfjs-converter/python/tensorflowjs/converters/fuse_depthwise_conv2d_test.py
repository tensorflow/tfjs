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
"""Unit tests for depthwise conv2d op fusing."""

import os
import shutil
import tempfile

import tensorflow.compat.v2 as tf

from tensorflowjs.converters import fuse_depthwise_conv2d
from tensorflowjs.converters import graph_rewrite_util
from tensorflowjs.converters import tf_saved_model_conversion_v2


class FuseDepthwiseConv2dTest(tf.test.TestCase):
  def setUp(self):
    super(FuseDepthwiseConv2dTest, self).setUp()
    self._tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(FuseDepthwiseConv2dTest, self).tearDown()

  def testFuseDepthwiseConv2dNativeWithBias(self):
    layers = [
        tf.keras.layers.DepthwiseConv2D(
            1, bias_initializer=tf.initializers.constant(0.25))
    ]
    model = tf.keras.Sequential(layers)
    tf.keras.backend.set_learning_phase(0)
    input_tensor = tf.constant([1.0, 1.0], shape=[1, 1, 1, 2])

    @tf.function
    def execute_model(tensor):
      return model(tensor)

    graph = tf_saved_model_conversion_v2._freeze_saved_model_v2(
        execute_model.get_concrete_function(input_tensor))
    graph_def = graph.as_graph_def()

    optimized_graph_def = fuse_depthwise_conv2d.fuse_depthwise_conv2d(graph_def)

    depthwise_conv2d_count = 0
    depthwise_conv2d = None
    for node in optimized_graph_def.node:
      self.assertNotEqual("BiasAdd", node.op)
      self.assertNotEqual("DepthwiseConv2dNative", node.op)
      if node.op == graph_rewrite_util.FUSED_DEPTHWISE_CONV2D:
        depthwise_conv2d_count += 1
        depthwise_conv2d = node
    self.assertEqual(depthwise_conv2d_count, 1)
    self.assertEqual(depthwise_conv2d.attr['fused_ops'].list.s, [b'BiasAdd'])
    self.assertEqual(depthwise_conv2d.attr['num_args'].i, 1)

  def testFuseDepthwiseConv2dNativeWithBiasAndActivation(self):
    layers = [
        tf.keras.layers.DepthwiseConv2D(
            1, bias_initializer=tf.initializers.constant(0.25)),
        tf.keras.layers.ReLU()
    ]
    model = tf.keras.Sequential(layers)
    tf.keras.backend.set_learning_phase(0)
    input_tensor = tf.constant([1.0, 1.0], shape=[1, 1, 1, 2])

    @tf.function
    def execute_model(tensor):
      return model(tensor)

    graph = tf_saved_model_conversion_v2._freeze_saved_model_v2(
        execute_model.get_concrete_function(input_tensor))
    graph_def = graph.as_graph_def()

    optimized_graph_def = fuse_depthwise_conv2d.fuse_depthwise_conv2d(graph_def)
    depthwise_conv2d_count = 0
    depthwise_conv2d = None
    for node in optimized_graph_def.node:
      self.assertNotEqual("BiasAdd", node.op)
      self.assertNotEqual("DepthwiseConv2dNative", node.op)
      self.assertNotEqual("Relu", node.op)
      if node.op == graph_rewrite_util.FUSED_DEPTHWISE_CONV2D:
        depthwise_conv2d_count += 1
        depthwise_conv2d = node
    self.assertEqual(depthwise_conv2d_count, 1)
    self.assertEqual(
        depthwise_conv2d.attr['fused_ops'].list.s, [b'BiasAdd', b'Relu'])
    self.assertEqual(depthwise_conv2d.attr['num_args'].i, 1)

  def testFuseDepthwiseConv2dNativeWithActivation(self):
    layers = [
        tf.keras.layers.DepthwiseConv2D(1, use_bias=False),
        tf.keras.layers.ReLU()
    ]
    model = tf.keras.Sequential(layers)
    tf.keras.backend.set_learning_phase(0)
    input_tensor = tf.constant([1.0, 1.0], shape=[1, 1, 1, 2])

    @tf.function
    def execute_model(tensor):
      return model(tensor)

    graph = tf_saved_model_conversion_v2._freeze_saved_model_v2(
        execute_model.get_concrete_function(input_tensor))
    graph_def = graph.as_graph_def()

    optimized_graph_def = fuse_depthwise_conv2d.fuse_depthwise_conv2d(graph_def)
    depthwise_conv2d_count = 0
    depthwise_conv2d = None
    for node in optimized_graph_def.node:
      self.assertNotEqual("BiasAdd", node.op)
      self.assertNotEqual("DepthwiseConv2dNative", node.op)
      self.assertNotEqual("Relu", node.op)
      if node.op == graph_rewrite_util.FUSED_DEPTHWISE_CONV2D:
        depthwise_conv2d_count += 1
        depthwise_conv2d = node
    self.assertEqual(depthwise_conv2d_count, 1)
    self.assertEqual(
        depthwise_conv2d.attr['fused_ops'].list.s, [b'NoOp', b'Relu'])
    self.assertEqual(depthwise_conv2d.attr['num_args'].i, 0)
if __name__ == '__main__':
  tf.test.main()
