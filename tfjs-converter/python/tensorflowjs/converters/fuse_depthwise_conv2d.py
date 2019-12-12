# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
 This transformation rule tries to identify following transformation
  DepthwiseConv2dNative + BiasAdd + Activation => FusedDepthwiseConv2dNative
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflowjs.converters import graph_rewrite_util

def _is_supported_activation(node):
  return node.op == 'Relu' or node.op == 'Relu6' or node.op == 'Elu'


def _find_contraction_with_bias(node, node_map):
  if node.op != 'BiasAdd':
    return False

  # Input to the BiasAdd must be a DepthwiseConv2dNative.
  if not node.input:
    return False

  conv2d_node = graph_rewrite_util.node_from_map(node_map, node.input[0])
  if conv2d_node.op != 'DepthwiseConv2dNative':
    return False

  return {'contraction': conv2d_node, 'bias': node, 'activation': None}

def _find_contraction_with_bias_and_activation(node, node_map):
  if not _is_supported_activation(node):
    return False

  # And input to the activation node must match ContractionWithBiasAdd pattern.
  if len(node.input) != 1:
    return False

  bias_add = graph_rewrite_util.node_from_map(node_map, node.input[0])

  match = _find_contraction_with_bias(bias_add, node_map)
  if not match:
    return False

  match['activation'] = node
  return match

def _add_fused_contraction_node(contraction, bias_add, activation,
                                inputs_to_remove, nodes_to_skip):
  fused_op = contraction
  fused_op.input.extend([bias_add.input[1]])

  fused_op.op = graph_rewrite_util.FUSED_DEPTHWISE_CONV2D
  fused_op.attr['fused_ops'].list.s.extend([b'BiasAdd'])
  fused_op.attr['num_args'].i = fused_op.attr['num_args'].i + 1
  bias_add.input[:] = [contraction.name]

  if activation:
    fused_op.attr['fused_ops'].list.s.extend([activation.op.encode('ascii')])
    nodes_to_skip[activation.name] = True
    activation.input[:] = [contraction.name]
    inputs_to_remove.append(activation)

  inputs_to_remove.append(bias_add)
  nodes_to_skip[bias_add.name] = True

def fuse_depthwise_conv2d(input_graph_def):
  """Modifies the provided graph by fusing a set of ops into a single
  _FusedDepthwiseConv2d op.

  DepthwiseConv2dNative + BiasAdd + Activation => _FusedDepthwiseConv2dNative

  Args:
    input_graph_def: A GraphDef containing a model.

  Returns:
    Modified graph with FusedDepthwiseConv2dNative ops generated, and modified
    weights.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """
  # Two passes approach, first find pattern of
  #   DepthwiseConv2dNative + BiasAdd + Activation
  # Then find pattern of
  #   DepthwiseConv2dNative + BiasAdd
  graph_def = _fuse_depthwise_conv2d_with_match_function(
      input_graph_def, _find_contraction_with_bias_and_activation)
  graph_def = _fuse_depthwise_conv2d_with_match_function(
      graph_def, _find_contraction_with_bias)
  return graph_def

def _fuse_depthwise_conv2d_with_match_function(input_graph_def, match_function):
  """Modifies the provided graph by fusing a set of ops into a single
  _FusedDepthwiseConv2d op.

  DepthwiseConv2dNative + BiasAdd + Activation => _FusedDepthwiseConv2dNative

  Args:
    input_graph_def: A GraphDef containing a model.
    match_function: A Function that matches the pattern and return a dict of
      contraction, bias_add and activation nodes.

  Returns:
    Modified graph with FusedDepthwiseConv2dNative ops generated, and modified
    weights.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """
  input_node_map = {}
  for node in input_graph_def.node:
    if node.name not in input_node_map:
      input_node_map[node.name] = node
    else:
      raise ValueError('Duplicate node names detected for ', node.name)

  nodes_to_skip = {}
  inputs_to_remove = []
  for node in input_graph_def.node:
    nodes = match_function(node, input_node_map)
    if nodes:
      _add_fused_contraction_node(nodes['contraction'], nodes['bias'],
                                  nodes['activation'], inputs_to_remove,
                                  nodes_to_skip)

  if nodes_to_skip or inputs_to_remove:
    return graph_rewrite_util.cleanup_graph_def(
        input_graph_def, nodes_to_skip, inputs_to_remove)

  # No pattern detected
  return input_graph_def
