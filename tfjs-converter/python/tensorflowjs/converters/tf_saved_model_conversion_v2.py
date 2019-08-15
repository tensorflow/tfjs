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
import os

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.saved_model.load import load
from tensorflow.python.training.saver import export_meta_graph
from google.protobuf.json_format import MessageToDict
import tensorflow_hub as hub

from tensorflowjs import write_weights
from tensorflowjs.converters import common

# enable eager execution for v2 APIs
tf.compat.v1.enable_eager_execution()

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
      'pruning', 'constfold', 'arithmetic', 'dependency', 'pruning', 'remap',
      'constfold', 'arithmetic', 'dependency'
  ]
  if strip_debug_ops:
    rewriter_config.optimizers.insert(0, 'debug_stripper')
  meta_graph = export_meta_graph(
      graph_def=graph_def, graph=graph)

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
