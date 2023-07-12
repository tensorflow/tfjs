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
"""Normalize BiasAdd op to be fused."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflowjs.converters import graph_rewrite_util

def normalize_bias_add_op(input_graph_def):
  """Convert AddV2 ops and Add ops to BiasAdd if they could be fused with the
  ancestor node.

  Grappler and the TFJS's fusing pass for DepthwiseConv2D can only fuse the
  BiasAdd op, but some AddV2 ops in the graph have the same functionality and
  can be fused with MatMul, Conv2D and DepthwiseConv2D ops. This function
  finds which AddV2 and Add ops in the graph can be fused and converts them
  to BiasAdd, which will be fused in the following passes. The AddV2 and Add ops
  must satisfy the following conditions to be fused:
    * The parent node has to be MatMul, Conv2D or DepthwiseConv.
    * The current node is the only child of the parent node (MatMul, Conv2D or
    DepthwiseConv).

  Args:
    input_graph_def: A GraphDef containing a model.

  Returns:
    Modified graph with fusable AddV2 and Add converted to BiasAdd.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """
  input_node_map = {}
  for node in input_graph_def.node:
    if node.name not in input_node_map:
      input_node_map[node.name] = node
    else:
      raise ValueError('Duplicate node names detected for ', node.name)

  for node in input_graph_def.node:
    if node.op == 'AddV2' or node.op == 'Add':
      ancestor_node_name = node.input[0]
      ancestor_node = graph_rewrite_util.node_from_map(input_node_map,
                                                       ancestor_node_name)
      if (ancestor_node.op == 'Conv2D' \
            or ancestor_node.op == 'DepthwiseConv2dNative'
            or ancestor_node.op == 'MatMul') \
          and len(graph_rewrite_util.get_output_node_names(input_node_map, ancestor_node_name)) == 1:
            node.op = 'BiasAdd'
            node.attr['data_format'].s = bytes('NHWC', 'utf-8')
  return input_graph_def
