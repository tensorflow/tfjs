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
"""Convert Tensorflow SavedModel to TensorFlow.js web format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import re
import os

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model.load import load
from tensorflow.python.training.saver import export_meta_graph
from google.protobuf.json_format import MessageToDict
import tensorflow_hub as hub

from tensorflowjs import write_weights
from tensorflowjs.converters import common

# enable eager execution for v2 APIs
tf.compat.v1.enable_eager_execution()

INPUT_ORDER = {
    # Order of inputs for BatchNormWithGlobalNormalization.
    "BatchNormWithGlobalNormalization": [
        "conv_op", "mean_op", "var_op", "beta_op", "gamma_op"
    ],
    # Order of inputs for FusedBatchNorm.
    "FusedBatchNorm": ["conv_op", "gamma_op", "beta_op", "mean_op", "var_op"],
    # Order of inputs for FusedBatchNormV3.
    "FusedBatchNormV3": ["conv_op", "gamma_op", "beta_op", "mean_op", "var_op"]
}
# Name of the attribute epsilon value is stored in.
EPSILON_ATTR = {
    "BatchNormWithGlobalNormalization": "variance_epsilon",
    "FusedBatchNorm": "epsilon",
    "FusedBatchNormV3": "epsilon"
}

CLEARED_TENSOR_FIELDS = (
    'tensor_content', 'half_val', 'float_val', 'double_val', 'int_val',
    'string_val', 'scomplex_val', 'int64_val', 'bool_val',
    'resource_handle_val', 'variant_val', 'uint32_val', 'uint64_val')

_HUB_V1_MODULE_PB = "tfhub_module.pb"

def load_graph(graph_filename):
  """Loads GraphDef. Returns Python Graph object.

  Args:
    graph_filename: string File name for the frozen graph.
  """
  with tf.compat.v1.gfile.Open(graph_filename, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    # Set name to empty to avoid using the default name 'import'.
    tf.import_graph_def(graph_def, name='')

  return graph

def get_cluster():
  """Grappler optimization configuration for GPU."""
  named_device = device_properties_pb2.NamedDevice()
  named_device.name = '/GPU:0'
  named_device.properties.type = 'GPU'
  named_device.properties.environment['architecture'] = '4'
  cluster = gcluster.Cluster(devices=[named_device])
  return cluster

def validate(nodes, skip_op_check, strip_debug_ops):
  """Validate if the node's op is compatible with TensorFlow.js.

  Args:
    nodes: tf.NodeDef TensorFlow NodeDef objects from GraphDef.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to allow unsupported debug ops.
  """
  if skip_op_check:
    return set()
  ops = []
  op_list_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), '../op_list/')
  for filename in os.listdir(op_list_path):
    if os.path.splitext(filename)[1] == '.json':
      with open(os.path.join(op_list_path, filename)) as json_data:
        ops += json.load(json_data)

  names = {x['tfOpName'] for x in ops}
  if strip_debug_ops:
    names = names.union({'Assert', 'CheckNumerics', 'Print'})
  not_supported = {x.op for x in [x for x in nodes if x.op not in names]}
  return not_supported

def optimize_graph(graph, output_node_names, output_graph, tf_version,
                   quantization_dtype=None, skip_op_check=False,
                   strip_debug_ops=False):
  """Takes a Python Graph object and optimizes the graph.

  Args:
    graph: The frozen graph to optimize.
    output_node_names: List of output node names.
    output_graph: The location of the output graph.
    tf_version: Tensorflow version of the input graph.
    quantization_dtype: An optional numpy dtype to quantize weights to for
      compression. Only np.uint8 and np.uint16 are supported.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
  """

  # Add a collection 'train_op' so that Grappler knows the outputs.
  for output in output_node_names:
    graph.add_to_collection('train_op', graph.get_operation_by_name(output))

  graph_def = graph.as_graph_def()

  unsupported = validate(graph_def.node, skip_op_check,
                         strip_debug_ops)
  if unsupported:
    raise ValueError('Unsupported Ops in the model before optimization\n' +
                     ', '.join(unsupported))

  config = config_pb2.ConfigProto()
  rewriter_config = config.graph_options.rewrite_options
  rewriter_config.optimizers[:] = [
      'pruning', 'constfold', 'arithmetic', 'dependency', 'pruning',
      'constfold', 'arithmetic', 'dependency'
  ]
  if strip_debug_ops:
    rewriter_config.optimizers.insert(0, 'debug_stripper')
  meta_graph = export_meta_graph(
      graph_def=graph_def, graph=graph)

  optimized_graph = tf_optimizer.OptimizeGraph(
      config, meta_graph, cluster=get_cluster())

  # batch norm folding
  optimized_graph = fold_batch_norms(optimized_graph)

  # set the device to CPU for all Conv2d nodes
  for node in optimized_graph.node:
    if node.op == 'Conv2D':
      node.device = '/device:CPU:0'

  # rerun grappler to fuse conv2d
  config.graph_options.rewrite_options.optimizers[:] = [
      'remap',
      'constfold', 'arithmetic', 'dependency'
  ]
  meta_graph = export_meta_graph(
      graph_def=optimized_graph, graph=graph)

  optimized_graph = tf_optimizer.OptimizeGraph(
      config, meta_graph, cluster=get_cluster())
  unsupported = validate(optimized_graph.node, skip_op_check,
                         strip_debug_ops)

  if unsupported:
    raise ValueError('Unsupported Ops in the model after optimization\n' +
                     ', '.join(unsupported))

  extract_weights(
      optimized_graph, output_graph, tf_version, quantization_dtype)
  return optimize_graph


def extract_weights(graph_def,
                    output_graph,
                    tf_version,
                    quantization_dtype=None):
  """Takes a Python GraphDef object and extract the weights.

  Args:
    graph_def: tf.GraphDef TensorFlow GraphDef proto object, which represents
      the model topology.
    tf_version: Tensorflow version of the input graph.
    quantization_dtype: An optional numpy dtype to quantize weights to for
        compression. Only np.uint8 and np.uint16 are supported.
  """
  constants = [node for node in graph_def.node if node.op == 'Const']
  const_inputs = {}
  # removed the conditional inputs for constants
  for const in constants:
    const_inputs[const.name] = const.input[:]
    del const.input[:]

  print('Writing weight file ' + output_graph + '...')
  const_manifest = []

  graph = tf.Graph()
  with tf.compat.v1.Session(graph=graph) as sess:
    tf.import_graph_def(graph_def, name='')
    for const in constants:
      tensor = graph.get_tensor_by_name(const.name + ':0')
      value = tensor.eval(session=sess)
      if not isinstance(value, np.ndarray):
        value = np.array(value)

      const_manifest.append({'name': const.name, 'data': value})

      # Restore the conditional inputs
      const.input[:] = const_inputs[const.name]

      # Remove the binary array from tensor and save it to the external file.
      for field_name in CLEARED_TENSOR_FIELDS:
        const.attr["value"].tensor.ClearField(field_name)

  write_artifacts(MessageToDict(graph_def), [const_manifest], output_graph,
                  tf_version, quantization_dtype=quantization_dtype)


def write_artifacts(topology,
                    weights,
                    output_graph,
                    tf_version,
                    quantization_dtype=None):
  """Writes weights and topology to the output_dir.

  If `topology` is Falsy (e.g., `None`), only emit weights to output_dir.

  Args:
    topology: tf.GraphDef TensorFlow GraphDef proto object, which represents
      the model topology.
    weights: an array of weight groups (as defined in tfjs write_weights).
    output_graph: the output file name to hold all the contents.
    tf_version: Tensorflow version of the input graph.
    quantization_dtype: An optional numpy dtype to quantize weights to for
      compression. Only np.uint8 and np.uint16 are supported.
  """
  model_json = {
      common.FORMAT_KEY: common.TFJS_GRAPH_MODEL_FORMAT,
      # TODO(piyu): Add tensorflow version below by using `meta_info_def`.
      common.GENERATED_BY_KEY: tf_version,
      common.CONVERTED_BY_KEY: common.get_converted_by(),
  }

  model_json[common.ARTIFACT_MODEL_TOPOLOGY_KEY] = topology or None
  weights_manifest = write_weights.write_weights(
      weights, os.path.dirname(output_graph), write_manifest=False,
      quantization_dtype=quantization_dtype)
  assert isinstance(weights_manifest, list)
  model_json[common.ARTIFACT_WEIGHTS_MANIFEST_KEY] = weights_manifest

  with open(output_graph, 'wt') as f:
    json.dump(model_json, f)


def _check_signature_in_model(saved_model, signature_name):
  if signature_name not in saved_model.signatures:
    raise ValueError("Signature '%s' does not exist. The following signatures "
                     "are available: %s" % (signature_name,
                                            saved_model.signatures.keys()))


def _freeze_saved_model_v1(graph, output_node_names):
  frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
      tf.compat.v1.Session(), graph.as_graph_def(), output_node_names)

  frozen_graph = tf.Graph()
  with frozen_graph.as_default():
    tf.import_graph_def(frozen_graph_def, name='')

  return frozen_graph

def _freeze_saved_model_v2(concrete_func):
  return convert_to_constants.convert_variables_to_constants_v2(
      concrete_func).graph

def convert_tf_saved_model(saved_model_dir,
                           output_dir, signature_def='serving_default',
                           saved_model_tags='serve',
                           quantization_dtype=None,
                           skip_op_check=False,
                           strip_debug_ops=False):
  """Freeze the SavedModel and check the model compatibility with Tensorflow.js.

  Optimize and convert the model to Tensorflow.js format, when the model passes
  the compatiblity check.

  Args:
    saved_model_dir: string The saved model directory.
    : string The names of the output nodes, comma separated.
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'model.json'
      - possibly sharded binary weight files.
    signature_def: string Tagset of the SignatureDef to load. Defaults to
      'serving_default'.
    saved_model_tags: tags of the GraphDef to load. Defaults to 'serve'.
    quantization_dtype: An optional numpy dtype to quantize weights to for
      compression. Only np.uint8 and np.uint16 are supported.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
  """
  if signature_def is None:
    signature_def = 'serving_default'

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  output_graph = os.path.join(
      output_dir, common.ARTIFACT_MODEL_JSON_FILE_NAME)

  saved_model_tags = saved_model_tags.split(', ')
  model = load(saved_model_dir, saved_model_tags)

  _check_signature_in_model(model, signature_def)

  concrete_func = model.signatures[signature_def]
  output_node_names = []
  for output_tensor in concrete_func.outputs:
    output_node_names.append(output_tensor.name.split(':')[0])

  # TensorFlow doesn't encode the saved model version in the graph in a reliable
  # way. Try to freeze the graph using V2 utils. If that fails, freeze the
  # graph using V1 utils.
  try:
    frozen_graph = _freeze_saved_model_v2(concrete_func)
  except BaseException:
    frozen_graph = _freeze_saved_model_v1(
        concrete_func.graph, output_node_names)

  optimize_graph(frozen_graph, output_node_names, output_graph,
                 model.tensorflow_version,
                 quantization_dtype=quantization_dtype,
                 skip_op_check=skip_op_check,
                 strip_debug_ops=strip_debug_ops)

def load_and_initialize_hub_module(module_path, signature='default'):
  """Loads graph of a TF-Hub module and initializes it into a session.

  Args:
    module_path: string Path to TF-Hub module.
    signature: string Signature to use when creating the apply graph.

  Return:
    graph: tf.Graph Graph of the module.
    session: tf.Session Session with initialized variables and tables.
    inputs: dict Dictionary of input tensors.
    outputs: dict Dictionary of output tensors.

  Raises:
    ValueError: If signature contains a SparseTensor on input or output.
  """
  graph = tf.Graph()
  with graph.as_default():
    tf.compat.v1.logging.info('Importing %s', module_path)
    module = hub.Module(module_path)

    signature_inputs = module.get_input_info_dict(signature)
    signature_outputs = module.get_output_info_dict(signature)
    # First check there are no SparseTensors in input or output.
    for key, info in list(signature_inputs.items()) + list(
        signature_outputs.items()):
      if info.is_sparse:
        raise ValueError(
            'Signature "%s" has a SparseTensor on input/output "%s".'
            ' SparseTensors are not supported.' % (signature, key))

    # Create placeholders to represent the input of the provided signature.
    inputs = {}
    for input_key, input_info in signature_inputs.items():
      inputs[input_key] = tf.compat.v1.placeholder(
          shape=input_info.get_shape(), dtype=input_info.dtype, name=input_key)

    outputs = module(inputs=inputs, signature=signature, as_dict=True)

    session = tf.compat.v1.Session(graph=graph)
    session.run(tf.compat.v1.global_variables_initializer())
    session.run(tf.compat.v1.tables_initializer())

  return graph, session, inputs, outputs


def convert_tf_hub_module_v1(module_path, output_dir,
                             signature='default', quantization_dtype=None,
                             skip_op_check=False, strip_debug_ops=False):
  """Freeze the TF-Hub module and check compatibility with Tensorflow.js.

  Optimize and convert the TF-Hub module to Tensorflow.js format, if it passes
  the compatiblity check.

  Args:
    module_path: string Path to the module.
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'model.json'
      - possibly sharded binary weight files.
    signature: string Signature to load.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
  """

  if signature is None:
    signature = 'default'

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  graph, sess, inputs, outputs = load_and_initialize_hub_module(
      module_path, signature)

  input_node_names = []
  output_node_names = []

  for _, input_tensor in inputs.items():
    input_node_names.append(input_tensor.name.split(':')[0])
  for _, output_tensor in outputs.items():
    output_node_names.append(output_tensor.name.split(':')[0])

  print('Creating a model with inputs %s and outputs %s.' % (input_node_names,
                                                             output_node_names))

  frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), output_node_names)

  output_graph = os.path.join(output_dir, common.ARTIFACT_MODEL_JSON_FILE_NAME)
  frozen_file = output_graph + '.frozen'
  try:
    with tf.compat.v1.gfile.GFile(frozen_file, 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

    frozen_graph = load_graph(frozen_file)
    optimize_graph(frozen_graph, output_node_names, output_graph,
                   tf.__version__, quantization_dtype=quantization_dtype,
                   skip_op_check=skip_op_check, strip_debug_ops=strip_debug_ops)
  finally:
    # Clean up the temp files.
    if os.path.exists(frozen_file):
      os.remove(frozen_file)


def convert_tf_hub_module(module_handle, output_dir,
                          signature='default', saved_model_tags='serve',
                          quantization_dtype=None, skip_op_check=False,
                          strip_debug_ops=False):
  """Conversion for TF Hub modules V1 and V2.

  See convert_tf_hub_module and convert_tf_saved_model.

  Args:
    module_path: string Path to the module.
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'model.json'
      - possibly sharded binary weight files.
    signature: string Signature to load.
    saved_model_tags: tags of the GraphDef to load. Defaults to ''.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
  """
  module_path = hub.resolve(module_handle)
  # TODO(vbardiovskyg): We can remove this v1 code path once loading of all v1
  # modules is fixed on the TF side, or once the modules we cannot load become
  # replaced with newer versions.
  if tf.io.gfile.exists(os.path.join(module_path, _HUB_V1_MODULE_PB)):
    print("Loading the module using TF 1.X interface from %s." % module_path)
    convert_tf_hub_module_v1(module_path, output_dir, signature,
                             quantization_dtype, skip_op_check, strip_debug_ops)
  else:
    print("Loading the module using TF 2.X interface from %s." % module_path)
    if signature is None:
      signature = 'default'
    convert_tf_saved_model(saved_model_dir=module_path,
                           output_dir=output_dir,
                           signature_def=signature,
                           saved_model_tags=saved_model_tags,
                           quantization_dtype=None,
                           skip_op_check=False,
                           strip_debug_ops=False)

# pylint: disable=R0915
def fold_batch_norms(input_graph_def):
  """Removes batch normalization ops by folding them into convolutions.

  Batch normalization during training has multiple dynamic parameters that are
  updated, but once the graph is finalized these become constants. That means
  there's an opportunity to reduce the computations down to a scale and
  addition, rather than the more expensive multiple ops, and even bake the
  scaling into the convolution weights. This function identifies the typical
  pattern of batch normalization subgraphs, and performs the transformation to
  fold the computations down into a simpler form. It currently only supports
  batch normalization that's performed by the BatchNormWithGlobalNormalization
  FusedBatchNorm and FusedBatchNormV3 ops, and will need to be extended in the
  future to handle the newer style.

  Args:
    input_graph_def: A GraphDef containing a model.

  Returns:
    Modified graph with BN ops removed, and modified weights.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """
  input_node_map = {}
  for node in input_graph_def.node:
    if node.name not in input_node_map:
      input_node_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected for ", node.name)

  nodes_to_skip = {}
  new_ops = []
  for node in input_graph_def.node:
    if (node.op not in ("BatchNormWithGlobalNormalization",
                        "FusedBatchNorm", "FusedBatchNormV3")):
      continue

    conv_op = node_from_map(input_node_map,
                            node.input[INPUT_ORDER[node.op].index("conv_op")])
    if conv_op.op != "Conv2D" and conv_op.op != "DepthwiseConv2dNative":
      tf_logging.warning("Didn't find expected Conv2D or DepthwiseConv2dNative"
                         " input to '%s'" % node.name)
      continue

    weights_op = node_from_map(input_node_map, conv_op.input[1])
    if weights_op.op != "Const":
      tf_logging.warning("Didn't find expected conv Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (conv_op.name, weights_op))
      continue
    weights = values_from_const(weights_op)
    if conv_op.op == "Conv2D":
      channel_count = weights.shape[3]
    elif conv_op.op == "DepthwiseConv2dNative":
      channel_count = weights.shape[2] * weights.shape[3]

    mean_op = node_from_map(input_node_map,
                            node.input[INPUT_ORDER[node.op].index("mean_op")])
    if mean_op.op != "Const":
      tf_logging.warning("Didn't find expected mean Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (node.name, mean_op))
      continue
    mean_value = values_from_const(mean_op)
    if mean_value.shape != (channel_count,):
      tf_logging.warning("Incorrect shape for mean, found %s, expected %s,"
                         " for node %s" % (str(mean_value.shape), str(
                             (channel_count,)), node.name))
      continue

    var_op = node_from_map(input_node_map,
                           node.input[INPUT_ORDER[node.op].index("var_op")])
    if var_op.op != "Const":
      tf_logging.warning("Didn't find expected var Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (node.name, var_op))
      continue
    var_value = values_from_const(var_op)
    if var_value.shape != (channel_count,):
      tf_logging.warning("Incorrect shape for var, found %s, expected %s,"
                         " for node %s" % (str(var_value.shape), str(
                             (channel_count,)), node.name))
      continue

    beta_op = node_from_map(input_node_map,
                            node.input[INPUT_ORDER[node.op].index("beta_op")])
    if beta_op.op != "Const":
      tf_logging.warning("Didn't find expected beta Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (node.name, beta_op))
      continue
    beta_value = values_from_const(beta_op)
    if beta_value.shape != (channel_count,):
      tf_logging.warning("Incorrect shape for beta, found %s, expected %s,"
                         " for node %s" % (str(beta_value.shape), str(
                             (channel_count,)), node.name))
      continue

    gamma_op = node_from_map(input_node_map,
                             node.input[INPUT_ORDER[node.op].index("gamma_op")])
    if gamma_op.op != "Const":
      tf_logging.warning("Didn't find expected gamma Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (node.name, gamma_op))
      continue
    gamma_value = values_from_const(gamma_op)
    if gamma_value.shape != (channel_count,):
      tf_logging.warning("Incorrect shape for gamma, found %s, expected %s,"
                         " for node %s" % (str(gamma_value.shape), str(
                             (channel_count,)), node.name))
      continue

    variance_epsilon_value = node.attr[EPSILON_ATTR[node.op]].f
    nodes_to_skip[node.name] = True
    nodes_to_skip[weights_op.name] = True
    nodes_to_skip[conv_op.name] = True

    if scale_after_normalization(node):
      scale_value = (
          (1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value)) *
          gamma_value)
    else:
      scale_value = (
          1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value))
    offset_value = (-mean_value * scale_value) + beta_value
    scaled_weights = np.copy(weights)
    it = np.nditer(
        scaled_weights, flags=["multi_index"], op_flags=["readwrite"])
    if conv_op.op == "Conv2D":
      while not it.finished:
        current_scale = scale_value[it.multi_index[3]]
        it[0] *= current_scale
        it.iternext()
    elif conv_op.op == "DepthwiseConv2dNative":
      channel_multiplier = weights.shape[3]
      while not it.finished:
        current_scale = scale_value[it.multi_index[2] * channel_multiplier +
                                    it.multi_index[3]]
        it[0] *= current_scale
        it.iternext()
    scaled_weights_op = node_def_pb2.NodeDef()
    scaled_weights_op.op = "Const"
    scaled_weights_op.name = weights_op.name
    scaled_weights_op.attr["dtype"].CopyFrom(weights_op.attr["dtype"])
    scaled_weights_op.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            scaled_weights, weights.dtype.type, weights.shape)))
    new_conv_op = node_def_pb2.NodeDef()
    new_conv_op.CopyFrom(conv_op)
    offset_op = node_def_pb2.NodeDef()
    offset_op.op = "Const"
    offset_op.name = conv_op.name + "_bn_offset"
    offset_op.attr["dtype"].CopyFrom(mean_op.attr["dtype"])
    offset_op.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            offset_value, mean_value.dtype.type, offset_value.shape)))
    bias_add_op = node_def_pb2.NodeDef()
    bias_add_op.op = "BiasAdd"
    bias_add_op.name = node.name
    bias_add_op.attr["T"].CopyFrom(conv_op.attr["T"])
    bias_add_op.attr["data_format"].CopyFrom(conv_op.attr["data_format"])
    bias_add_op.input.extend([new_conv_op.name, offset_op.name])
    new_ops.extend([scaled_weights_op, new_conv_op, offset_op, bias_add_op])

  result_graph_def = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if node.name in nodes_to_skip:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    result_graph_def.node.extend([new_node])

  result_graph_def.node.extend(new_ops)
  return result_graph_def

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
