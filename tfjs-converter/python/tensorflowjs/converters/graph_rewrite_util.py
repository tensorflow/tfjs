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
import re

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util

# Custom op name for fused depthwise conv2d
FUSED_DEPTHWISE_CONV2D = 'FusedDepthwiseConv2dNative'
# The grappler op name for fused MatMul which starts with '_'
FUSED_MATMUL = '_FusedMatMul'
FUSED_CONV2D = '_FusedConv2D'

def node_from_map(node_map, name):
  """Pulls a node def from a dictionary for a given name.

  Args:
    node_map: Dictionary containing an entry indexed by name for every node.
    name: Identifies the node we want to find.

  Returns:
    NodeDef of the node with the given name.

  Raises:
    ValueError: If the node isn't present in the dictionary.
  """
  stripped_name = node_name_from_input(name)
  if stripped_name not in node_map:
    raise ValueError("No node named '%s' found in map." % name)
  return node_map[stripped_name]


def values_from_const(node_def):
  """Extracts the values from a const NodeDef as a numpy ndarray.

  Args:
    node_def: Const NodeDef that has the values we want to access.

  Returns:
    Numpy ndarray containing the values.

  Raises:
    ValueError: If the node isn't a Const.
  """
  if node_def.op != "Const":
    raise ValueError(
        "Node named '%s' should be a Const op for values_from_const." %
        node_def.name)
  input_tensor = node_def.attr["value"].tensor
  tensor_value = tensor_util.MakeNdarray(input_tensor)
  return tensor_value

# Whether to scale by gamma after normalization.
def scale_after_normalization(node):
  if node.op == "BatchNormWithGlobalNormalization":
    return node.attr["scale_after_normalization"].b
  return True

def node_name_from_input(node_name):
  """Strips off ports and other decorations to get the underlying node name."""
  if node_name.startswith("^"):
    node_name = node_name[1:]
  m = re.search(r"(.*):\d+$", node_name)
  if m:
    node_name = m.group(1)
  return node_name

def cleanup_graph_def(input_graph_def, nodes_to_skip, inputs_to_remove):
  """Clean up the graph def by removing the skipped nodes and clean up the nodes
    with inputs that have been removed.

  Args:
    input_graph_def: GraphDef object to be cleaned.
    node_to_skip: Dict with node names to be skipped.
    inputs_to_remove: List of nodes to be removed from inputs of all nodes.
  Returns:
    GraphDef that has been cleaned.

  """
  result_graph_def = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if node.name in nodes_to_skip:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    for value in inputs_to_remove:
      for i, input_node in enumerate(new_node.input):
        if input_node == value.name:
          new_node.input[i] = value.input[0]
    result_graph_def.node.extend([new_node])
  result_graph_def.library.CopyFrom(input_graph_def.library)
  result_graph_def.versions.CopyFrom(input_graph_def.versions)
  return result_graph_def

def rename_constants(node_list, prefix):
  """Update all constants name by adding a prefix.

  Args:
    node_list: NodeDef list to update.
    prefix: string to add to the constant name.
  Returns:
    NodeDef list that has been updated.

  """
  nodes = []
  constant_names = [node.name for node in node_list if node.op == 'Const']
  for node in node_list:
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    nodes.append(new_node)
    if node.op == 'Const':
      new_node.name = prefix + '/' + node.name
    else:
      for i, input_node in enumerate(new_node.input):
        for name in constant_names:
          if input_node.startswith(name):
            new_node.input[i] = prefix + '/' + input_node
  return nodes

def get_output_node_names(node_map, target):
  output_node_names = []
  for name, node in node_map.items():
    for input_name in node.input:
      if target == input_name:
        output_node_names.append(name)
  return output_node_names
