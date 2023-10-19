# Copyright 2023 Google LLC
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
from tensorflow.python.framework import dtypes

from tensorflowjs.converters import normalize_bias_add
from tensorflowjs.converters import graph_rewrite_util
from tensorflowjs.converters import tf_saved_model_conversion_v2


class NormalizeBiasAddTest(tf.test.TestCase):
  def setUp(self):
    super(NormalizeBiasAddTest, self).setUp()
    self._tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(NormalizeBiasAddTest, self).tearDown()

  def testFuseConv2DWithAddV2(self):
    @tf.function
    def conv2d_addV2(x):
      filter = tf.ones([1, 1, 1, 1])
      bias = tf.constant([100], dtype=dtypes.float32)
      res = tf.raw_ops.Conv2D(
        input=x, filter=filter, strides=[1, 1, 1, 1], padding="VALID")
      res = tf.raw_ops.AddV2(x=res, y=bias)
      return res

    input_tensor = tf.constant([1.0], shape=[1, 1, 1, 1])
    graph = tf_saved_model_conversion_v2._freeze_saved_model_v2(
        conv2d_addV2.get_concrete_function(input_tensor))
    graph_def = graph.as_graph_def()

    optimized_graph_def = normalize_bias_add.normalize_bias_add_op(graph_def)

    bias_add_count = 0
    bias_add = None
    for node in optimized_graph_def.node:
      self.assertNotEqual("AddV2", node.op)
      if node.op == "BiasAdd":
        bias_add_count += 1
        bias_add = node
    self.assertEqual(bias_add_count, 1)
    self.assertEqual(bias_add.attr['data_format'].s, b'NHWC')

  def testFuseDepthwiseConv2dNativeWithAddV2(self):
    @tf.function
    def depthwise_addV2(x):
      filter = tf.ones([1, 1, 1, 1])
      bias = tf.constant([100], dtype=dtypes.float32)
      res = tf.raw_ops.DepthwiseConv2dNative(
        input=x, filter=filter, strides=[1, 1, 1, 1], padding="VALID")
      res = tf.raw_ops.AddV2(x=res, y=bias)
      return res

    input_tensor = tf.constant([1.0], shape=[1, 1, 1, 1])
    graph = tf_saved_model_conversion_v2._freeze_saved_model_v2(
        depthwise_addV2.get_concrete_function(input_tensor))
    graph_def = graph.as_graph_def()

    optimized_graph_def = normalize_bias_add.normalize_bias_add_op(graph_def)

    bias_add_count = 0
    bias_add = None
    for node in optimized_graph_def.node:
      self.assertNotEqual("AddV2", node.op)
      if node.op == "BiasAdd":
        bias_add_count += 1
        bias_add = node
    self.assertEqual(bias_add_count, 1)
    self.assertEqual(bias_add.attr['data_format'].s, b'NHWC')

  def testMatmulWithAddV2(self):
    @tf.function
    def matmul_addV2(x):
      y = tf.ones([1, 1])
      bias = tf.constant([100], dtype=dtypes.float32)
      res = tf.raw_ops.MatMul(a=x, b=y)
      res = tf.raw_ops.AddV2(x=res, y=bias)
      return res

    input_tensor = tf.constant([1.0], shape=[1, 1])
    graph = tf_saved_model_conversion_v2._freeze_saved_model_v2(
        matmul_addV2.get_concrete_function(input_tensor))
    graph_def = graph.as_graph_def()

    optimized_graph_def = normalize_bias_add.normalize_bias_add_op(graph_def)

    bias_add_count = 0
    bias_add = None
    for node in optimized_graph_def.node:
      self.assertNotEqual("AddV2", node.op)
      if node.op == "BiasAdd":
        bias_add_count += 1
        bias_add = node
    self.assertEqual(bias_add_count, 1)
    self.assertEqual(bias_add.attr['data_format'].s, b'NHWC')
if __name__ == '__main__':
  tf.test.main()
