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
from tensorflowjs.converters import fold_batch_norms

class FoldBatchNormsTest(tf.test.TestCase):
  def setUp(self):
    super(FoldBatchNormsTest, self).setUp()
    self._tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(FoldBatchNormsTest, self).tearDown()

  def testFoldBatchNorms(self):
    with tf.compat.v1.Session() as sess:
      inputs = [1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6]
      input_op = constant_op.constant(
          np.array(inputs), shape=[1, 1, 6, 2], dtype=dtypes.float32)
      weights = [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]
      weights_op = constant_op.constant(
          np.array(weights), shape=[1, 2, 2, 2], dtype=dtypes.float32)
      conv_op = nn_ops.conv2d(
          input_op, weights_op, [1, 1, 1, 1], padding="SAME", name="conv_op")
      mean_op = constant_op.constant(
          np.array([10, 20]), shape=[2], dtype=dtypes.float32)
      variance_op = constant_op.constant(
          np.array([0.25, 0.5]), shape=[2], dtype=dtypes.float32)
      beta_op = constant_op.constant(
          np.array([0.1, 0.6]), shape=[2], dtype=dtypes.float32)
      gamma_op = constant_op.constant(
          np.array([1.0, 2.0]), shape=[2], dtype=dtypes.float32)
      test_util.set_producer_version(ops.get_default_graph(), 8)
      gen_nn_ops._batch_norm_with_global_normalization(
          conv_op,
          mean_op,
          variance_op,
          beta_op,
          gamma_op,
          0.00001,
          False,
          name="output")
      original_graph_def = sess.graph_def
      original_result = sess.run(["output:0"])
    optimized_graph_def = fold_batch_norms.fold_batch_norms(
        original_graph_def)
    with tf.compat.v1.Session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized")
      optimized_result = sess.run(["optimized/output:0"])

    self.assertAllClose(original_result, optimized_result)

    for node in optimized_graph_def.node:
      self.assertNotEqual("BatchNormWithGlobalNormalization", node.op)

  def testFoldFusedBatchNorms(self):
    for data_format, conv2d_func in [
        ("NHWC", nn_ops.conv2d), ("NCHW", nn_ops.conv2d),
        ("NHWC", nn_ops.depthwise_conv2d_native),
        ("NCHW", nn_ops.depthwise_conv2d_native)
    ]:
      with tf.compat.v1.Session() as sess:
        inputs = [1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6]
        input_op = constant_op.constant(
            np.array(inputs),
            shape=[1, 1, 6, 2] if data_format == "NHWC" else [1, 2, 1, 6],
            dtype=dtypes.float32)
        if conv2d_func == nn_ops.conv2d:
          weights = [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]
          weights_op = constant_op.constant(
              np.array(weights), shape=[1, 2, 2, 2], dtype=dtypes.float32)
        else:
          weights = [1, 2, 0.3, 0.4]
          weights_op = constant_op.constant(
              np.array(weights), shape=[1, 2, 2, 1], dtype=dtypes.float32)
        conv_op = conv2d_func(
            input_op,
            weights_op, [1, 1, 1, 1],
            padding="SAME",
            data_format=data_format,
            name="conv_op")
        mean_op = constant_op.constant(
            np.array([10, 20]), shape=[2], dtype=dtypes.float32)
        variance_op = constant_op.constant(
            np.array([0.25, 0.5]), shape=[2], dtype=dtypes.float32)
        beta_op = constant_op.constant(
            np.array([0.1, 0.6]), shape=[2], dtype=dtypes.float32)
        gamma_op = constant_op.constant(
            np.array([1.0, 2.0]), shape=[2], dtype=dtypes.float32)
        ops.get_default_graph().graph_def_versions.producer = 9
        gen_nn_ops._fused_batch_norm(
            conv_op,
            gamma_op,
            beta_op,
            mean_op,
            variance_op,
            0.00001,
            is_training=False,
            data_format=data_format,
            name="output")
        original_graph_def = sess.graph_def
        original_result = sess.run(["output:0"])
      optimized_graph_def = fold_batch_norms.fold_batch_norms(
          original_graph_def)
    with tf.compat.v1.Session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized")
      optimized_result = sess.run(["optimized/output:0"])

      self.assertAllClose(
          original_result, optimized_result, rtol=1e-04, atol=1e-06)

      for node in optimized_graph_def.node:
        self.assertNotEqual("FusedBatchNorm", node.op)

  def testFoldFusedBatchNormsV3(self):
    for data_format, conv2d_func in [
        ("NHWC", nn_ops.conv2d), ("NCHW", nn_ops.conv2d),
        ("NHWC", nn_ops.depthwise_conv2d_native),
        ("NCHW", nn_ops.depthwise_conv2d_native)
    ]:
      with tf.compat.v1.Session() as sess:
        _generate_fused_batchnorm(data_format, conv2d_func)
        original_graph_def = sess.graph_def
        original_result = sess.run(["output:0"])
      optimized_graph_def = fold_batch_norms.fold_batch_norms(
          original_graph_def)
    with tf.compat.v1.Session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized")
      optimized_result = sess.run(["optimized/output:0"])

      self.assertAllClose(
          original_result, optimized_result, rtol=1e-04, atol=1e-06)

      for node in optimized_graph_def.node:
        self.assertNotEqual("FusedBatchNormV3", node.op)


  def testFoldFusedBatchNormWithBias(self):
    for data_format, conv2d_func in [
        ("NHWC", nn_ops.conv2d),
        ("NHWC", nn_ops.depthwise_conv2d_native),
    ]:
      graph = tf.Graph()
      with tf.compat.v1.Session(graph=graph) as sess:
        count = 1
        add_bias = True
        _generate_fused_batchnorm(data_format, conv2d_func, count, add_bias)
        original_graph_def = sess.graph_def
        original_result = sess.run(["output:0"])
      optimized_graph_def = fold_batch_norms.fold_batch_norms(
          original_graph_def)
    with tf.compat.v1.Session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized")
      optimized_result = sess.run(["optimized/output:0"])

      self.assertAllClose(
          original_result, optimized_result, rtol=1e-04, atol=1e-06)

      bias_nodes = [
          node for node in optimized_graph_def.node if node.op == 'BiasAdd'
      ]
      self.assertEqual(len(bias_nodes), 1)
      for node in optimized_graph_def.node:
        self.assertNotEqual("FusedBatchNormV3", node.op)

  def testFoldFusedBatchNormsWithSharedWeights(self):
    for data_format, conv2d_func in [
        ("NHWC", nn_ops.conv2d), ("NCHW", nn_ops.conv2d),
        ("NHWC", nn_ops.depthwise_conv2d_native),
        ("NCHW", nn_ops.depthwise_conv2d_native)
    ]:
      with tf.compat.v1.Session() as sess:
        _generate_fused_batchnorm(data_format, conv2d_func, 2)
        original_graph_def = sess.graph_def
        original_result = sess.run(["output:0"])
      optimized_graph_def = fold_batch_norms.fold_batch_norms(
          original_graph_def)
    with tf.compat.v1.Session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized")
      optimized_result = sess.run(["optimized/output:0"])

      self.assertAllClose(
          original_result, optimized_result, rtol=1e-04, atol=1e-06)

      for node in optimized_graph_def.node:
        self.assertNotEqual("FusedBatchNormV3", node.op)

def _generate_fused_batchnorm(data_format, conv2d_func, count=1,
                              add_bias=False):
  inputs = [1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6]
  input_op = constant_op.constant(
      np.array(inputs),
      shape=[1, 1, 6, 2] if data_format == "NHWC" else [1, 2, 1, 6],
      dtype=dtypes.float32)
  if conv2d_func == nn_ops.conv2d:
    weights = [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]
    weights_op = constant_op.constant(
        np.array(weights), shape=[1, 2, 2, 2], dtype=dtypes.float32)
  else:
    weights = [1, 2, 0.3, 0.4]
    weights_op = constant_op.constant(
        np.array(weights), shape=[1, 2, 2, 1], dtype=dtypes.float32)
  mean_op = constant_op.constant(
      np.array([10, 20]), shape=[2], dtype=dtypes.float32)
  variance_op = constant_op.constant(
      np.array([0.25, 0.5]), shape=[2], dtype=dtypes.float32)
  beta_op = constant_op.constant(
      np.array([0.1, 0.6]), shape=[2], dtype=dtypes.float32)
  gamma_op = constant_op.constant(
      np.array([1.0, 2.0]), shape=[2], dtype=dtypes.float32)
  ops.get_default_graph().graph_def_versions.producer = 9
  for _ in range(count):
    conv_op = conv2d_func(
        input_op,
        weights_op, [1, 1, 1, 1],
        padding="SAME",
        data_format=data_format,
        name="conv_op")
    if add_bias:
      out_channels = conv_op.shape[3]
      bias = constant_op.constant(
          np.array([1.0]*out_channels),
          shape=[out_channels], dtype=dtypes.float32)
      conv_op = nn_ops.bias_add(conv_op, bias)
    gen_nn_ops.fused_batch_norm_v3(
        conv_op,
        gamma_op,
        beta_op,
        mean_op,
        variance_op,
        0.00001,
        is_training=False,
        data_format=data_format,
        name="output")
if __name__ == '__main__':
  tf.test.main()
